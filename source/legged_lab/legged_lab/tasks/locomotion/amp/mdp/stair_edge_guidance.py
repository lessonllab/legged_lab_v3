"""Training-only pre-contact guidance around exposed square-stair upper lips.

Inspired by Hiking in the Wild's virtual edge regions and dense unsafe stepping
penalties. Specialized to this project's axis-aligned, square pyramid stairs;
not a full reimplementation of either paper or a general mesh edge detector.
"""
import numpy as np
import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply
from .style_state import tensor


def exposed_stair_rings(meshes, origin, tile_size):
    """Extract height discontinuities of the actual box meshes, ignoring seams.

The square symmetry lets a centerline profile describe all four sides. Compare
the visible top envelope on both sides of each box boundary, so buried boxes,
internal seams, and bottom edges never become virtual obstacles.
    """
    bounds = np.array([mesh.bounds for mesh in meshes])
    center = np.asarray(origin)
    for mesh in meshes:
        normals = np.abs(mesh.face_normals)
        if not np.allclose(normals.max(axis=1), 1., atol=1e-6):
            raise ValueError('Stair edge guidance requires axis-aligned box geometry')
    # Adjacent boxes can represent the same boundary a few ULPs apart.
    candidates = np.unique(np.round(bounds[:, :, 0],6))
    candidates = candidates[(candidates > center[0]+1e-4) & (candidates < tile_size[0]-1e-4)]
    def top(x):
        hit = ((bounds[:, 0, 0] <= x) & (bounds[:, 1, 0] >= x)
               & (bounds[:, 0, 1] <= center[1]) & (bounds[:, 1, 1] >= center[1]))
        return bounds[hit, 1, 2].max() if hit.any() else np.nan
    rings = []
    for x in candidates:
        a, b = top(x-1e-4), top(x+1e-4)
        if np.isfinite(a+b) and abs(a-b) > .05:
            rings.append((x-center[0], max(a,b)-center[2]))
    return np.array(rings, dtype=np.float32).reshape(-1,2)


def edge_approach_cost(points, velocities, rings, unloaded, band=.04):
    """Points/velocities [N,feet,P,3], sorted (radius,z) rings [N,K,2].

Only the two neighboring square rings can enter a narrow safety band. No
all-points/all-edges distance tensor is allocated. Motion away from the edge
receives zero cost, including normal toe-off. Ring padding uses radius 1e6.
    """
    if band <= 0.:
        raise ValueError('Edge safety band must be positive')
    n, feet, count, _ = points.shape
    p = points.reshape(n,-1,3)
    v = velocities.reshape(n,-1,3)
    radius = p[..., :2].abs().amax(-1)
    idx = torch.searchsorted(rings[...,0].contiguous(), radius.contiguous())
    costs = []
    for offset in (-1,0):
        chosen = (idx+offset).clamp(0,rings.shape[1]-1)
        r = rings[...,0].gather(1,chosen)
        z = rings[...,1].gather(1,chosen)
        x,y = p[...,0],p[...,1]
        sx,sy = torch.where(x>=0,1.,-1.),torch.where(y>=0,1.,-1.)
        # Closest point on vertical/horizontal sides of a square ring.
        ax,ay = sx*r,torch.maximum(torch.minimum(y,r),-r)
        bx,by = torch.maximum(torch.minimum(x,r),-r),sy*r
        first = (x-ax).square()+(y-ay).square() <= (x-bx).square()+(y-by).square()
        closest = torch.stack((torch.where(first,ax,bx),torch.where(first,ay,by),z),-1)
        delta = p-closest
        distance = delta.norm(dim=-1)
        closing = (-(v*delta).sum(-1)/distance.clamp_min(1e-6)).clamp(0.,3.)
        costs.append((1.-distance/band).clamp_min(0.)*closing)
    cost = torch.maximum(*costs).reshape(n,feet,count).amax(-1)
    return (cost*unloaded).sum(-1)


class StairEdgeApproach(ManagerTermBase):
    """Privileged geometric reward only: no policy inputs or physical obstacles."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.command = env.command_manager.get_term('base_velocity')
        self.robot = env.scene['robot']
        self.contact = env.scene['contact_forces']
        names = [f'{side}_ankle_roll_link' for side in ('left','right')]
        self.feet = [self.robot.body_names.index(name) for name in names]
        self.contact_feet = [self.contact.body_names.index(name) for name in names]
        generator = env.scene.terrain.cfg.terrain_generator
        if generator.size[0] != generator.size[1]:
            raise ValueError('Stair edge guidance requires square tiles')
        all_rings = []
        expected_steps = getattr(self.command, 'expected_stair_steps', 33)
        for name in ('pyramid_stairs_inv','pyramid_stairs'):
            rows = [np.empty((0,2),dtype=np.float32)]
            terrain = generator.sub_terrains[name].copy()
            terrain.size = generator.size
            for row in range(1,10):
                meshes,origin = terrain.function((row+.5)/10.,terrain)
                rings = exposed_stair_rings(meshes,origin,generator.size)
                if len(rings) != expected_steps:
                    raise ValueError(f'Expected {expected_steps} exposed stair rings, got {len(rings)}')
                rows.append(rings)
            all_rings.append(rows)
        table = np.zeros((2,10,expected_steps,2),dtype=np.float32)
        table[...,0] = 1e6
        for d,rows in enumerate(all_rings):
            for row,rings in enumerate(rows): table[d,row,:len(rings)] = rings
        self.rings = torch.tensor(table,device=env.device)
        y,x = torch.meshgrid(torch.linspace(-.025,.025,3,device=env.device),
                             torch.linspace(-.05,.12,9,device=env.device),indexing='ij')
        self.local_points = torch.stack((x.flatten(),y.flatten(),torch.full_like(x.flatten(),-.035)),-1)

    def __call__(self,env,band=.04):
        c = self.command
        ids = torch.where(c.stairs & ~c._skip_progress_transition)[0]
        result = torch.zeros(env.num_envs,device=env.device)
        if not len(ids): return result
        data = self.robot.data
        pos = tensor(data.body_pos_w)[ids][:,self.feet]
        q = tensor(data.body_quat_w)[ids][:,self.feet]
        local = self.local_points[None,None].expand(len(ids),2,-1,-1)
        offsets = quat_apply(q[:,:,None].expand(-1,-1,len(self.local_points),-1),local)
        points = pos[:,:,None]+offsets-tensor(c.terrain.env_origins)[ids,None,None]
        velocity = tensor(data.body_lin_vel_w)[ids][:,self.feet,None]
        omega = tensor(data.body_ang_vel_w)[ids][:,self.feet,None].expand_as(offsets)
        velocity = velocity+torch.cross(omega,offsets,dim=-1)
        forces = tensor(self.contact.data.net_normal_forces_w)[ids][:,self.contact_feet,2]
        unloaded = ((20.-forces)/20.).clamp(0.,1.)
        rows = tensor(c.terrain.terrain_levels)[ids]
        rings = self.rings[c.group[ids]-1,rows]
        result[ids] = edge_approach_cost(points,velocity,rings,unloaded,band)
        return result
