"""V12 motion and foothold shaping. Terrain probes are training-only supervision.

The additions are local hypotheses, not a reproduction of a published method.
No fixed gait phase: reward useful completed steps, not time spent holding a foot up.
"""
import torch
from isaaclab.utils.math import quat_apply, yaw_quat
from .style_state import tensor
from .locomotion_progress import motion_state


def progress_tracking(command, velocity, yaw, std=.5, angular=False):
    speed = command[:, :2].norm(dim=-1)
    along = (velocity * command[:, :2]).sum(-1) / speed.square().clamp_min(1e-6)
    rotation = yaw * command[:, 2] / command[:, 2].square().clamp_min(1e-6)
    if angular:
        gate = torch.where(speed > .08, along.clamp(0., 1.),
                           torch.where(command[:, 2].abs() > .15, rotation.clamp(0., 1.), 1.))
        error = (command[:, 2]-yaw).square()
    else:
        gate = torch.where(speed > .08, along.clamp(0., 1.), 1.)
        error = (command[:, :2]-velocity).square().sum(-1)
    return torch.exp(-error/std**2)*gate


def useful_tracking(env, command_name='base_velocity', std=.5, angular=False):
    return progress_tracking(*motion_state(env, command_name), std=std, angular=angular)


def path_ray_pattern(cfg, device):
    # Both forward and backward lookahead, with a center ray under each ankle.
    x, y = torch.meshgrid(torch.tensor([-.24,-.16,-.08,0.,.08,.16,.24,.32],device=device),
                          torch.tensor([-.025,0.,.025],device=device),indexing='ij')
    starts=torch.stack((x.flatten(),y.flatten(),torch.full_like(x.flatten(),.5)),-1)
    directions=torch.zeros_like(starts);directions[:,2]=-1.
    return starts,directions


def lookahead_height(heights, forward):
    """[N,feet,8,3] world heights; ignore misses, flat ground, downhill and smooth slopes.

    An upward discontinuity >4 cm across an 8 cm interval enables clearance
    supervision. A 10% quantile is not used: thin blocks must remain visible.
    """
    strip=heights.amax(-1)
    valid=torch.isfinite(heights).all(-1)
    inc=strip[...,1:]-strip[...,:-1]
    pair=valid[...,1:] & valid[...,:-1]
    front_edges=pair[...,3:] & (inc[...,3:]>.04)
    back_edges=pair[...,:3] & (inc[...,:3]<-.04)
    front=front_edges.any(-1);back=back_edges.any(-1)
    # Nearest lip only: never demand clearing two successive steps at once.
    front_index=front_edges.long().argmax(-1)+4
    back_index=2-back_edges.flip(-1).long().argmax(-1)
    front_h=strip.gather(-1,front_index[...,None]).squeeze(-1)
    back_h=strip.gather(-1,back_index[...,None]).squeeze(-1)
    enabled=torch.where(forward,front,back) & valid[...,3]
    target=torch.where(forward,front_h,back_h)
    return torch.where(enabled & torch.isfinite(target),target,0.), enabled


def loaded_point_speed(linear, angular, offsets, load):
    """A rotating loaded foot can scuff even if its ankle center does not translate."""
    velocities=linear[:,:,None,:]+torch.cross(angular[:,:,None,:].expand_as(offsets),offsets,dim=-1)
    return (velocities[...,:2].norm(dim=-1).mean(-1).clamp(max=2.)*load).sum(-1)


def step_credit(touchdown, air_duration, yaw_progress, peak_clearance, slip, last_foot, turning):
    feet=torch.arange(2,device=touchdown.device)[None,:]
    good=(touchdown & (air_duration>=.08) & (air_duration<=.8) & (yaw_progress>.04)
          & (peak_clearance>.025) & (slip<.25) & (feet!=last_foot[:,None]))
    good &= turning[:,None]
    # Never pay twice on a simultaneous bilateral landing.
    good &= (touchdown.sum(-1)==1)[:,None]
    return good


class FootholdSignals:
    def __init__(self,env):
        self.env=env;self.robot=env.scene['robot'];self.sensor=env.scene['contact_forces']
        self.feet=self.robot.find_bodies(['left_ankle_roll_link','right_ankle_roll_link'],preserve_order=True)[0]
        self.contacts=[self.sensor.body_names.index(f'{s}_ankle_roll_link') for s in ('left','right')]
        n=env.num_envs;device=env.device
        self.ema=torch.ones(n,device=device)
        self.prev_contact=torch.ones(n,2,dtype=torch.bool,device=device)
        self.air=torch.zeros(n,2,device=device);self.yaw_progress=torch.zeros_like(self.air)
        self.clearance=torch.zeros_like(self.air);self.lift_z=torch.zeros_like(self.air)
        self.last_foot=torch.full((n,),-1,device=device,dtype=torch.long)
        self.turn_swing=torch.zeros(n,2,device=device,dtype=torch.bool)
        self.local=torch.tensor([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.025,-.035],[.12,.025,-.035]],device=device)
        self.values={};self.cached_step=None

    def update(self):
        e=self.env
        if self.cached_step==e.common_step_counter:return
        self.cached_step=e.common_step_counter
        d=self.robot.data;command,velocity,yaw=motion_state(e)
        speed=command[:,:2].norm(dim=-1);moving=speed>.08
        turning=(speed<=.08)&(command[:,2].abs()>.15)
        fresh=tensor(e.episode_length_buf)<=1
        force=tensor(self.sensor.data.net_normal_forces_w)[:,self.contacts]
        contact=force[:,:,2]>30.
        pos=tensor(d.body_pos_w)[:,self.feet];quat=tensor(d.body_quat_w)[:,self.feet]
        lin=tensor(d.body_link_lin_vel_w)[:,self.feet];ang=tensor(d.body_ang_vel_w)[:,self.feet]
        offsets=quat_apply(quat[:,:,None,:].expand(-1,-1,4,-1),self.local[None,None].expand(e.num_envs,2,-1,-1))
        sole_z=(pos[:,:,None,:]+offsets)[...,2].amin(-1)
        self.ema[fresh]=1.;self.air[fresh]=0.;self.yaw_progress[fresh]=0.;self.clearance[fresh]=0.
        self.prev_contact[fresh]=contact[fresh];self.last_foot[fresh]=-1;self.turn_swing[fresh]=False
        ratio=((velocity*command[:,:2]).sum(-1)/speed.square().clamp_min(1e-6)).clamp(-1.,1.)
        # ~1 s filter avoids punishing the ordinary near-zero velocity in each stance transition.
        self.ema=torch.where(moving,self.ema+(e.step_dt/(1.+e.step_dt))*(ratio-self.ema),torch.ones_like(ratio))
        stall=((.25-self.ema)/.25).clamp(0.,2.)*moving*(tensor(e.episode_length_buf)*e.step_dt>1.)
        load=((force[:,:,2]-30.)/120.).clamp(0.,1.)
        scuff=loaded_point_speed(lin,ang,offsets,load)*turning
        # Integrate each swing; pay only after a useful, alternating touchdown.
        liftoff=self.prev_contact & ~contact
        self.lift_z=torch.where(liftoff,sole_z,self.lift_z)
        self.turn_swing=torch.where(liftoff,turning[:,None]&contact.flip(-1),self.turn_swing)
        self.air=torch.where(~contact,self.air+e.step_dt,self.air)
        self.yaw_progress+=torch.where(~contact,(yaw*command[:,2].sign())[:,None]*e.step_dt,0.)
        self.clearance=torch.maximum(self.clearance,torch.where(~contact,sole_z-self.lift_z,0.))
        touchdown=~self.prev_contact & contact & ~fresh[:,None]
        point_velocity=lin[:,:,None]+torch.cross(ang[:,:,None].expand_as(offsets),offsets,dim=-1)
        slip=point_velocity[...,:2].norm(dim=-1).mean(-1)
        credit=step_credit(touchdown & self.turn_swing,self.air,self.yaw_progress,self.clearance,slip,self.last_foot,turning)
        self.last_foot=torch.where(credit.any(-1),credit.long().argmax(-1),self.last_foot)
        self.air[contact]=0.;self.yaw_progress[contact]=0.;self.clearance[contact]=0.
        self.turn_swing[contact]=False
        self.prev_contact=contact
        c=e.command_manager.get_term('base_velocity')
        names=c.column_names;types=tensor(c.terrain.terrain_types)
        rough_cols=torch.tensor([name in ('boxes','random_rough') for name in names],device=e.device)
        nonflat=tensor(c.terrain.terrain_levels)>0
        obstacle=(c.stairs | rough_cols[types]) & nonflat & moving
        heights=torch.stack([tensor(e.scene[f'{s}_path_scanner'].data.ray_hits_w)[...,2].reshape(e.num_envs,8,3) for s in ('left','right')],1)
        axis=quat_apply(yaw_quat(quat),torch.tensor([1.,0.,0.],device=e.device).expand(e.num_envs,2,3))
        from isaaclab.utils.math import quat_apply as rotate
        world_cmd=rotate(yaw_quat(tensor(d.root_quat_w)),command*torch.tensor([1.,1.,0.],device=e.device))
        foot_direction=(axis*world_cmd[:,None]).sum(-1)
        upper,near=lookahead_height(heights,foot_direction>=0)
        swing=(~contact)&(self.air>.04)&(self.air<.8)
        deficit=((upper+.035-sole_z)/.15).clamp(0.,1.)
        path_cost=(deficit*near*swing*(foot_direction.abs()>.08)).sum(-1)*obstacle
        # Generalize impact feedback beyond analytically parameterized stair rings.
        impact=((force[:,:,:2].norm(dim=-1)-3.*force[:,:,2].abs()-30.)/100.).clamp(0.,2.).sum(-1)
        self.values={'stall':stall,'turn_scuff':scuff,'turn_step':credit.float().sum(-1)/e.step_dt,
                     'path_clearance':path_cost,'rough_impact':impact*rough_cols[types]*nonflat*moving}


def foothold_signal(env,kind):
    if not hasattr(env,'_foothold_signals'):env._foothold_signals=FootholdSignals(env)
    env._foothold_signals.update()
    return env._foothold_signals.values[kind]


def rough_support(env,asset_cfg,contact_cfg,sensor_names):
    from .foot_support import foot_support_penalty
    c=env.command_manager.get_term('base_velocity')
    columns=torch.tensor([n in ('boxes','random_rough') for n in c.column_names],device=env.device)
    active=columns[tensor(c.terrain.terrain_types)] & (tensor(c.terrain.terrain_levels)>0)
    return foot_support_penalty(env,asset_cfg,contact_cfg,sensor_names,'support',support_target=.8,contact_ramp=.08)*active


def walking_air_time(env,command_name,vel_threshold,sensor_cfg):
    from .rewards import instinct_feet_air_time
    command=env.command_manager.get_command(command_name)
    reward=instinct_feet_air_time(env,command_name,vel_threshold,sensor_cfg).clamp(max=.3)
    return reward*(command[:,:2].norm(dim=-1)>.08)


def navigation_heading_error(env,command_name='base_velocity'):
    c=env.command_manager.get_term(command_name)
    command=c.command
    # Exempt only the dedicated rate-command pool. Turning to a rear target
    # still has a heading error, even while the controller commands zero vx.
    rate_pool=getattr(c,'turn_pool',torch.zeros(command.shape[0],dtype=torch.bool,device=command.device))
    pure_rate=rate_pool & ~c.manual_target
    return command[:,2].abs()*(~pure_rate)
