"""Instinct-style virtual edge volumes and an additional toe/riser cost.

The cylinder distance and sum(depth * (point_speed + 1e-6)) match the local
InstinctLab implementation. Geometry extraction is specialized to the exposed
upper lips of this project's square box stairs, not arbitrary mesh terrain.
Virtual volumes never change collision geometry or policy observations.
"""
import numpy as np
import torch
from isaaclab.utils.math import quat_apply, yaw_quat
from .stair_edge_guidance import StairEdgeApproach
from .style_state import tensor


def neighboring_rings(points, rings):
    radius = points[..., :2].abs().amax(-1)
    idx = torch.searchsorted(rings[..., 0].contiguous(), radius.contiguous())
    for offset in (-1, 0):
        chosen = (idx + offset).clamp(0, rings.shape[1] - 1)
        yield rings[..., 0].gather(1, chosen), rings[..., 1].gather(1, chosen), chosen


def cylinder_depth(points, rings, radius=.05):
    """[N,P,3], [N,K,2] -> [N,P]. Finite cylinders, not rounded capsules.

    Adjacent stair rings are 30 cm apart, much wider than two cylinder radii.
    Two neighboring rings therefore cover all possible penetration candidates.
    Outside an edge's endpoint planes has zero penetration (as in Instinct).
    """
    if not 0 < radius < .15:
        raise ValueError('Cylinder radius must be positive and below half the 30 cm tread')
    x, y, z = points.unbind(-1)
    depth = torch.zeros_like(x)
    for r, h, _ in neighboring_rings(points, rings):
        dx = (radius - torch.sqrt((x.abs()-r).square() + (z-h).square())).clamp_min(0.)
        dy = (radius - torch.sqrt((y.abs()-r).square() + (z-h).square())).clamp_min(0.)
        dx = torch.where(y.abs() <= r, dx, 0.)
        dy = torch.where(x.abs() <= r, dy, 0.)
        depth = torch.maximum(depth, torch.maximum(dx, dy))
    return depth


def instinct_penetration_cost(depth, velocities):
    return (depth * (velocities.norm(dim=-1) + 1e-6)).sum(-1)


def stair_surface_height(points, rings, lower, high_outside):
    """Exact top height of concentric square treads, including the end platforms.

    Input [N,P,3]; rings must be sorted, extracted from the actual box meshes.
    This analytic query is only for this course, not arbitrary terrain meshes.
    """
    radius = points[..., :2].abs().amax(-1)
    idx = torch.searchsorted(rings[..., 0].contiguous(), radius.contiguous())
    chosen = idx.clamp_max(rings.shape[1] - 1)
    upper = rings[..., 1].gather(1, chosen)
    low = lower.gather(1, chosen)
    outside = high_outside.gather(1, chosen)
    inner_height = torch.where(outside, low, upper)
    outer_height = torch.where(outside, upper, low)
    return torch.where(idx < rings.shape[1], inner_height, outer_height)


def loaded_overhang_cost(sole, heights, forces):
    """[N,F,P,3] physical sole samples; penalize loaded overhang, not proximity.

    Only samples over a lower tread and at least 2.5 cm above it count.
    Tilt/toe-off on a single plane is not overhang. The vertical load ramp
    removes swing/light unloading contacts and softens load transfer. There is
    no velocity multiplier: standing on an edge cannot erase this cost.
    """
    valid = torch.isfinite(heights) & torch.isfinite(sole).all(-1)
    highest = torch.where(valid, heights, -torch.inf).amax(-1, keepdim=True)
    lower_tread = valid & (highest - heights > .05)
    gap = torch.nan_to_num(sole[..., 2] - heights, nan=0., posinf=0., neginf=0.)
    unsupported = lower_tread.float() * ((gap - .025) / .05).clamp(0., 1.)
    fraction = unsupported.sum(-1) / valid.sum(-1).clamp_min(1)
    load = ((forces[..., 2] - 50.) / 150.).clamp(0., 1.)
    return (fraction * load).sum(-1), fraction * load


def swing_clearance_cost(points, rings, lower, high_outside, forces, air_time, desired_world,
                         margin=.12, clearance=.025):
    """Clear the upcoming upper lip by 2.5 cm during an ascending swing only.

    Command direction gates ascent, so slowing an obstructed swing to zero
    does not remove the signal. No penalty for stance, descending, retreating,
    a distant riser or a toe already above the clearance target.
    """
    n, feet, count, _ = points.shape
    p = points.reshape(n, -1, 3)
    penalty = torch.zeros_like(p[..., 0])
    swing = ((forces[..., 2] < 20.) & (air_time > .02))
    for r, upper, idx in neighboring_rings(p, rings):
        low = lower.gather(1, idx)
        orientation = torch.where(high_outside.gather(1, idx), 1., -1.)
        for axis in (0, 1):
            sign = torch.where(p[..., axis] >= 0, 1., -1.)
            d = -orientation * (p[..., axis].abs() - r)
            approaching = desired_world[:, axis, None] * sign * orientation > .1
            active = (approaching & (p[..., 1-axis].abs() <= r) & (d >= -.02)
                      & (d < margin) & (p[..., 2] >= low - .02))
            deficit = ((upper + clearance - p[..., 2]) / .12).clamp(0., 1.)
            penalty = torch.maximum(penalty, active * (1.-d/margin).clamp(0., 1.) * deficit)
    return (penalty.reshape(n, feet, count).amax(-1) * swing).sum(-1)


def toe_riser_cost(points, velocities, rings, lower, high_outside, forces, margin=.10):
    """Toe points [N,F,P,3], forces [N,F,3]. Return cost and near-riser flag.

    Only the vertical face near the toe, below its upper lip, participates.
    Penalize motion towards the high side and horizontal normal contact force
    resisting that motion. Walking on a tread, high clearance and moving away
    without a collision do not incur the cost. Supports ascending either way.
    """
    n, feet, count, _ = points.shape
    p, v = points.reshape(n,-1,3), velocities.reshape(n,-1,3)
    force = forces[:,:,None].expand(-1,-1,count,-1).reshape(n,-1,3)
    penalty, proximity = torch.zeros_like(p[...,0]), torch.zeros_like(p[...,0])
    for r, upper, idx in neighboring_rings(p, rings):
        low = lower.gather(1,idx)
        outward_high = high_outside.gather(1,idx)
        orientation = torch.where(outward_high, 1., -1.)
        for axis in (0,1):
            tangent = 1-axis
            sign = torch.where(p[...,axis] >= 0, 1., -1.)
            # d > 0 on the low side, d < 0 in the higher solid.
            d = -orientation * (p[...,axis].abs()-r)
            active = ((p[...,tangent].abs() <= r) & (p[...,2] >= low-.01)
                      & (p[...,2] < upper) & (d >= -.02) & (d < margin))
            band = (1.-d/margin).clamp(0.,1.) * active
            closing = (v[...,axis]*sign*orientation).clamp(0.,3.)
            impact = ((-force[...,axis]*sign*orientation-20.)/100.).clamp(0.,3.)
            penalty = torch.maximum(penalty, band*(closing+impact))
            proximity = torch.maximum(proximity, band)
    return (penalty.reshape(n,feet,count).amax(-1).sum(-1),
            proximity.reshape(n,feet,count).amax(-1) > 0.)


class StairVirtualSafety(StairEdgeApproach):
    def __init__(self, env):
        super().__init__(None, env)
        # Instinct parkour's 10x5x2 foot-volume sampling, in ankle-roll frame.
        x,y,z = torch.meshgrid(torch.linspace(-.025,.12,10,device=env.device),
                               torch.linspace(-.03,.03,5,device=env.device),
                               torch.linspace(-.04,0.,2,device=env.device),indexing='ij')
        self.local_points = torch.stack((x.flatten(),y.flatten(),z.flatten()),-1)
        self.toe_mask = self.local_points[:,0] >= .10
        # Physical sole anchors verified against this G1 USD (not shoe-equipped G1).
        sx, sy = torch.meshgrid(torch.linspace(-.05,.12,10,device=env.device),
                                torch.linspace(-.025,.025,5,device=env.device),indexing='ij')
        self.sole_points = torch.stack((sx.flatten(),sy.flatten(),torch.full_like(sx.flatten(),-.035)),-1)
        # Recover the lower side and orientation from actual exposed mesh tops.
        self.lower = torch.zeros_like(self.rings[...,0])
        self.high_outside = torch.zeros_like(self.lower,dtype=torch.bool)
        gen = env.scene.terrain.cfg.terrain_generator
        for direction,name in enumerate(('pyramid_stairs_inv','pyramid_stairs')):
            cfg = gen.sub_terrains[name].copy();cfg.size=gen.size
            for row in range(1,10):
                meshes,origin=cfg.function((row+.5)/10.,cfg)
                bounds=np.array([m.bounds for m in meshes])
                for k,(r,_) in enumerate(self.rings[direction,row].cpu().numpy()):
                    tops=[]
                    for offset in (-1e-4,1e-4):
                        px=origin[0]+float(r)+offset
                        hit=((bounds[:,0,0]<=px)&(bounds[:,1,0]>=px)
                             &(bounds[:,0,1]<=origin[1])&(bounds[:,1,1]>=origin[1]))
                        tops.append(bounds[hit,1,2].max()-origin[2])
                    self.lower[direction,row,k]=float(min(tops))
                    self.high_outside[direction,row,k]=bool(tops[1]>tops[0])
        print('[VirtualSafety] 5 cm finite edge cylinders; 100 volume points per foot; Instinct depth*speed cost; toe/riser approach and normal-impact penalty', flush=True)
        self.cached_step = None
        self.edge_cost = torch.zeros(env.num_envs,device=env.device)
        self.toe_cost = torch.zeros_like(self.edge_cost)
        self.support_cost = torch.zeros_like(self.edge_cost)
        self.clearance_cost = torch.zeros_like(self.edge_cost)
        self.loaded_overhang = torch.zeros(env.num_envs,2,device=env.device)
        self.depth = torch.zeros(env.num_envs,2,100,device=env.device)
        self.toe_near = torch.zeros(env.num_envs,2,device=env.device,dtype=torch.bool)
        self.points_w = torch.zeros(env.num_envs,2,100,3,device=env.device)

    def update(self,env,force=False):
        if not force and self.cached_step == env.common_step_counter:
            return self
        self.cached_step=env.common_step_counter
        self.edge_cost.zero_();self.toe_cost.zero_();self.depth.zero_();self.toe_near.zero_()
        self.support_cost.zero_();self.clearance_cost.zero_();self.loaded_overhang.zero_()
        c=self.command
        ids=torch.where(c.stairs & ~c._skip_progress_transition)[0]
        if not len(ids):return self
        data=self.robot.data
        pos=tensor(data.body_pos_w)[ids][:,self.feet]
        q=tensor(data.body_quat_w)[ids][:,self.feet]
        local=self.local_points[None,None].expand(len(ids),2,-1,-1)
        offsets=quat_apply(q[:,:,None].expand(-1,-1,100,-1),local)
        world=pos[:,:,None]+offsets
        points=world-tensor(c.terrain.env_origins)[ids,None,None]
        velocity=tensor(data.body_lin_vel_w)[ids][:,self.feet,None]
        omega=tensor(data.body_ang_vel_w)[ids][:,self.feet,None].expand_as(offsets)
        velocity=velocity+torch.cross(omega,offsets,dim=-1)
        rows=tensor(c.terrain.terrain_levels)[ids];directions=c.group[ids]-1
        rings=self.rings[directions,rows]
        depth=cylinder_depth(points.reshape(len(ids),-1,3),rings)
        self.edge_cost[ids]=instinct_penetration_cost(depth,velocity.reshape(len(ids),-1,3))
        self.depth[ids]=depth.reshape(-1,2,100);self.points_w[ids]=world
        forces=tensor(self.contact.data.net_normal_forces_w)[ids][:,self.contact_feet]
        cost,near=toe_riser_cost(points[:,:,self.toe_mask],velocity[:,:,self.toe_mask],rings,
                               self.lower[directions,rows],self.high_outside[directions,rows],forces)
        self.toe_cost[ids]=cost;self.toe_near[ids]=near
        sole_local=self.sole_points[None,None].expand(len(ids),2,-1,-1)
        sole=pos[:,:,None]+quat_apply(q[:,:,None].expand(-1,-1,50,-1),sole_local)
        sole=sole-tensor(c.terrain.env_origins)[ids,None,None]
        heights=stair_surface_height(sole.reshape(len(ids),-1,3),rings,
                                    self.lower[directions,rows],self.high_outside[directions,rows])
        self.support_cost[ids],self.loaded_overhang[ids]=loaded_overhang_cost(sole,heights.reshape(-1,2,50),forces)
        air_time=tensor(self.contact.data.current_air_time)[ids][:,self.contact_feet]
        desired=c.vel_command_b[ids].clone();desired[:,2]=0.
        desired_world=quat_apply(yaw_quat(tensor(data.root_quat_w)[ids]),desired)
        self.clearance_cost[ids]=swing_clearance_cost(points[:,:,self.toe_mask],rings,
            self.lower[directions,rows],self.high_outside[directions,rows],forces,air_time,desired_world)
        return self


def get_stair_virtual_safety(env):
    if not hasattr(env,'_stair_virtual_safety'):
        env._stair_virtual_safety=StairVirtualSafety(env)
    return env._stair_virtual_safety


def virtual_stair_reward(env,kind):
    safety=get_stair_virtual_safety(env).update(env)
    if kind=='edge':return safety.edge_cost
    if kind=='toe':return safety.toe_cost
    if kind=='support':return safety.support_cost
    if kind=='clearance':return safety.clearance_cost
    raise ValueError(f'Unknown stair safety reward {kind}')
