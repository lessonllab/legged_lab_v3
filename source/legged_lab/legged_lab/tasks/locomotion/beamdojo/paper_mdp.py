"""Optional paper-aligned profile; no periodic leg targets or gait clock."""
import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from ..unitree_official import mdp as native
from .paper_terrain import geometry_tables,sample_surface
from .geometry import g1_sole_points,transform_sole_points


def ensure_state(env):
    if hasattr(env,'beamdojo_levels'):
        return
    device=env.device
    env._paper_tables=geometry_tables(env.cfg.beamdojo_stage,env.cfg.seed,device)
    env.beamdojo_levels=torch.full((env.num_envs,),env.cfg.beamdojo_initial_level,device=device,dtype=torch.long)
    env.beamdojo_success_streak=torch.zeros_like(env.beamdojo_levels)
    env.beamdojo_direction=torch.ones(env.num_envs,device=device)
    env.beamdojo_qualified_success=torch.zeros(env.num_envs,device=device,dtype=torch.bool)
    env._paper_unsupported=torch.zeros(env.num_envs,device=device)
    env._paper_contacts=torch.zeros(env.num_envs,device=device)
    env._paper_finish_time=torch.zeros(env.num_envs,device=device)
    env._paper_arrived=torch.zeros(env.num_envs,device=device,dtype=torch.bool)
    env._paper_visited=torch.zeros((env.num_envs,env._paper_tables[0].shape[1]),device=device,dtype=torch.bool)
    env._paper_stats_tick=-1
    env._paper_action_tick=-1
    env._paper_previous_previous_action=None
    env._paper_last_action=None
    env._paper_action_curvature=torch.zeros(env.num_envs,device=device)
    refresh_geometry(env)


def refresh_geometry(env):
    tables=env._paper_tables
    env.beamdojo_geometry=tables[0][env.beamdojo_levels]
    env.beamdojo_rectangle_mask=tables[1][env.beamdojo_levels]
    env.beamdojo_stone_mask=tables[2][env.beamdojo_levels]
    env.beamdojo_start_platform=env.beamdojo_geometry[:,0]
    last=env.beamdojo_rectangle_mask.sum(-1)-1 if env.cfg.beamdojo_stage==2 else torch.zeros_like(env.beamdojo_levels)
    env.beamdojo_end_platform=env.beamdojo_geometry[torch.arange(env.num_envs,device=env.device),last]
    if env.cfg.beamdojo_stage==2:
        terrain=env.scene.terrain
        terrain.terrain_levels[:]=env.beamdojo_levels
        terrain.env_origins[:]=terrain.terrain_origins[env.beamdojo_levels,terrain.terrain_types]
        env.scene.env_origins[:]=terrain.env_origins


def surface_height(env,points):
    if (getattr(env.cfg,'beamdojo_upstream_aligned',False)
            or getattr(env.cfg,'beamdojo_unitree_motion_aligned',False)):
        from .upstream_terrain import surface_height as upstream_surface
        return upstream_surface(env,points)
    ensure_state(env)
    return sample_surface(points,native.tensor(env.scene.env_origins),env.beamdojo_levels,
                          env._paper_tables,env.cfg.beamdojo_stage)


def foot_state(env):
    ensure_state(env)
    if not hasattr(env,'_paper_feet_ids'):
        env._paper_feet_ids=env.scene['robot'].find_bodies(['left_ankle_roll_link','right_ankle_roll_link'],preserve_order=True)[0]
        env._paper_contact_ids=env.scene.sensors['contact_forces'].find_bodies(['left_ankle_roll_link','right_ankle_roll_link'],preserve_order=True)[0]
    data=env.scene['robot'].data
    pos=native.tensor(data.body_pos_w)[:,env._paper_feet_ids]
    quat=native.tensor(data.body_quat_w)[:,env._paper_feet_ids][...,[3,0,1,2]]
    soles=transform_sole_points(pos,quat,g1_sole_points(device=env.device))
    force=native.tensor(env.scene.sensors['contact_forces'].data.net_forces_w)[:,env._paper_contact_ids]
    return soles,force.norm(dim=-1)>5.


def foothold(env):
    soles,contact=foot_state(env)
    _,safe=surface_height(env,soles)
    # Paper depth-threshold rule: a virtual +/-.05 m stone is safe even though
    # Stage-1 physical contacts are at z=0. Do not compare sole z to virtual z.
    return -((~safe)&contact[...,None]).sum(dim=(-1,-2)).float()


def action_smoothness(env):
    ensure_state(env)
    if env._paper_action_tick!=env.common_step_counter:
        action=env.action_manager.action
        if env._paper_last_action is None:
            env._paper_last_action=torch.zeros_like(action)
            env._paper_previous_previous_action=torch.zeros_like(action)
        env._paper_action_curvature=(action-2*env._paper_last_action+env._paper_previous_previous_action).square().sum(-1)
        env._paper_previous_previous_action=env._paper_last_action.clone()
        env._paper_last_action=action.clone()
        env._paper_action_tick=env.common_step_counter
    return env._paper_action_curvature


def stand_still(env):
    data=env.scene['robot'].data
    deviation=native.tensor(data.joint_pos)-native.tensor(data.default_joint_pos)
    stop=env.command_manager.get_command('base_velocity')[:,:2].square().sum(-1)<.1
    # G1 adaptation: theta is displacement from the valid bent-knee rest pose.
    return deviation.square().sum(-1)*stop


def joint_velocity_limits(env):
    data=env.scene['robot'].data
    return (native.tensor(data.joint_vel).abs()-native.tensor(data.joint_vel_limits)).clamp_min(0.).sum(-1)


def joint_power(env):
    data=env.scene['robot'].data
    power=(native.tensor(data.applied_torque).abs()*native.tensor(data.joint_vel).abs()).sum(-1)
    speed=native.tensor(data.root_lin_vel_w).square().sum(-1)+.2*native.tensor(data.root_ang_vel_w).square().sum(-1)
    # The paper denominator is singular at standing; finite .1 lower bound is
    # an explicit numerical G1 adaptation, not an unreported paper constant.
    return power/speed.clamp_min(.1)


def feet_parallel(env):
    soles,_=foot_state(env)
    return soles[...,2].var(-1,unbiased=False).sum(-1)


def feet_distance(env):
    data=env.scene['robot'].data
    soles,_=foot_state(env)
    centers=soles.mean(-2)-native.tensor(data.root_pos_w)[:,None,:]
    q=native.tensor(data.root_quat_w)
    from isaaclab.utils.math import quat_apply_inverse
    centers=quat_apply_inverse(q[:,None,:].expand(-1,2,-1).reshape(-1,4),centers.reshape(-1,3)).reshape(-1,2,3)
    # Published positive hinge with bounded excess to prevent reward hacking
    # by progressively wider splits; .18 m threshold is the paper value.
    return ((centers[:,0,1]-centers[:,1,1]).abs()-.18).clamp(0.,.12)


def feet_air_time(env):
    foot_state(env)
    sensor=env.scene.sensors['contact_forces']
    first=native.tensor(sensor.compute_first_contact(env.step_dt))[:,env._paper_contact_ids]
    air=native.tensor(sensor.data.last_air_time)[:,env._paper_contact_ids]
    moving=env.command_manager.get_command('base_velocity')[:,:2].norm(dim=-1)>.1
    return ((air-.5)*first).sum(-1)*moving


def feet_clearance(env):
    soles,_=foot_state(env)
    data=env.scene['robot'].data
    pos=native.tensor(data.body_pos_w)[:,env._paper_feet_ids]
    linear=native.tensor(data.body_link_lin_vel_w)[:,env._paper_feet_ids]
    angular=native.tensor(data.body_ang_vel_w)[:,env._paper_feet_ids]
    velocity=linear[...,None,:]+torch.cross(angular[...,None,:].expand_as(soles),soles-pos[...,None,:],dim=-1)
    # Stage 1 locomotion uses physical flat-floor z, never the virtual void.
    relative=soles[...,2]-native.tensor(env.scene.env_origins)[:,None,2:3]
    return ((relative-.1).square()*velocity[...,:2].norm(dim=-1)).mean(-1).sum(-1)


def natural_arm_swing(env,asset_cfg,hip_cfg):
    data=env.scene['robot'].data
    arm=native.tensor(data.joint_pos)[:,asset_cfg.joint_ids]
    rest=native.tensor(data.default_joint_pos)[:,asset_cfg.joint_ids]
    hip=native.tensor(data.joint_pos)[:,hip_cfg.joint_ids]-native.tensor(data.default_joint_pos)[:,hip_cfg.joint_ids]
    moving=env.command_manager.get_command('base_velocity')[:,:2].norm(dim=-1)>.1
    # G1 user extension: use actual opposite-leg motion, not a forced leg phase.
    target=(.6*hip.flip(-1)).clamp(-.25,.25)*moving[:,None]
    return torch.exp(-(arm-rest-target).square().sum(-1)/.08)


def update_statistics(env):
    ensure_state(env)
    if env._paper_stats_tick==env.common_step_counter:
        return
    soles,contact=foot_state(env)
    _,safe=surface_height(env,soles)
    env._paper_unsupported+=((~safe)&contact[...,None]).sum(dim=(-1,-2))
    env._paper_contacts+=contact.sum(-1)*soles.shape[-2]
    data=env.scene['robot'].data
    relative=native.tensor(data.root_pos_w)-native.tensor(env.scene.env_origins)
    upright=native.tensor(data.projected_gravity_b)[:,2]<-.85
    accurate=env._paper_unsupported/env._paper_contacts.clamp_min(1.)<=.15
    if env.cfg.beamdojo_stage==1:
        traversal=relative[:,:2].norm(dim=-1)>=3.
        env.beamdojo_qualified_success[:]=traversal&upright&accurate&(env._paper_contacts>0)
    else:
        local=soles-native.tensor(env.scene.env_origins)[:,None,None,:]
        for i in range(env.beamdojo_geometry.shape[1]):
            r=env.beamdojo_geometry[:,i]
            hit=((local[...,:2]-r[:,None,None,:2]).abs()<=r[:,None,None,2:4]/2).all(-1)
            hit &= (local[...,2]-r[:,None,None,4]).abs()<.06
            qualified=(hit.float().mean(-1)>=.85)&contact
            env._paper_visited[:,i]|=qualified.any(-1)&env.beamdojo_stone_mask[:,i]
        end_index=env.beamdojo_rectangle_mask.sum(-1)-1
        end=env.beamdojo_geometry[torch.arange(env.num_envs,device=env.device),end_index]
        target=torch.where((env.beamdojo_direction>0)[:,None],end,env.beamdojo_geometry[:,0])
        center=target[:,:2]
        reached=((relative[:,:2]-center).abs()<torch.tensor([.7,.7],device=env.device)).all(-1)
        env._paper_arrived |= reached
        env.command_manager.get_command('base_velocity')[env._paper_arrived]=0.
        feet_inside=((local[...,:2]-center[:,None,None,:]).abs()<target[:,None,None,2:4]/2-.01).all(dim=(-1,-2,-3))
        feet_inside &= (local[...,2]-target[:,None,None,4]).abs().amax(dim=(-1,-2))<.06
        slow=(native.tensor(data.root_lin_vel_w).norm(dim=-1)<.12)&(native.tensor(data.root_ang_vel_w).norm(dim=-1)<.25)
        all_visited=(env._paper_visited|~env.beamdojo_stone_mask).all(-1)
        stable=env._paper_arrived&feet_inside&contact.all(-1)&upright&slow&all_visited&accurate
        env._paper_finish_time[:]=torch.where(stable,env._paper_finish_time+env.step_dt,0.)
        env.beamdojo_qualified_success[:]=env._paper_finish_time>=.6-1e-6
    env._paper_stats_tick=env.common_step_counter


def finished(env):
    update_statistics(env)
    return env.beamdojo_qualified_success


def left_course(env):
    ensure_state(env)
    p=native.tensor(env.scene['robot'].data.root_pos_w)-native.tensor(env.scene.env_origins)
    if env.cfg.beamdojo_stage==1:
        return (p[:,:2].abs().amax(-1)>3.9)|(p[:,2]<.25)
    end_id=env.beamdojo_rectangle_mask.sum(-1)-1
    end=env.beamdojo_geometry[torch.arange(env.num_envs,device=env.device),end_id]
    return (p[:,0]<-1.)|(p[:,0]>end[:,0]+1.)|(p[:,1].abs()>1.)|(p[:,2]<.25)


class PaperVelocityCommand(UniformVelocityCommand):
    def _resample_command(self,env_ids):
        super()._resample_command(env_ids)
        ensure_state(self._env)
        if self._env.cfg.beamdojo_stage==2:
            self.vel_command_b[env_ids,0]=self.vel_command_b[env_ids,0].abs()*self._env.beamdojo_direction[env_ids]

    def _update_command(self):
        super()._update_command()
        if self._env.cfg.beamdojo_stage==2:
            ensure_state(self._env)
            self.vel_command_b[self._env._paper_arrived]=0.


class PaperResetRoot(native.reset_root_state_uniform):
    """Extend Sim6's stateful reset term after it caches ranges at construction."""
    def __call__(self,env,env_ids,pose_range,velocity_range):
        ensure_state(env)
        super().__call__(env,env_ids,pose_range,velocity_range)
        if env.cfg.beamdojo_stage==2:
            ids=env_ids[env.beamdojo_direction[env_ids]<0]
            if ids.numel():
                data=env.scene['robot'].data
                pose=torch.cat((native.tensor(data.root_pos_w)[ids],native.tensor(data.root_quat_w)[ids]),dim=-1).clone()
                end_id=env.beamdojo_rectangle_mask[ids].sum(-1)-1
                end=env.beamdojo_geometry[ids,end_id]
                pose[:,0]+=end[:,0]
                env.scene['robot'].write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)


def before_reset(env,env_ids):
    ensure_state(env)
    experienced=env.episode_length_buf[env_ids]>0
    skip=env._paper_skip_curriculum_once[env_ids] if hasattr(env,'_paper_skip_curriculum_once') else torch.zeros_like(experienced)
    failed=torch.zeros_like(experienced)
    if hasattr(env,'termination_manager'):
        for name in env.termination_manager.active_terms:
            if name not in ('finished','time_out'):
                failed |= env.termination_manager.get_term(name)[env_ids].bool()
    wins=env.beamdojo_qualified_success[env_ids]&experienced&~failed&~skip
    rate=wins.float().sum()/(experienced&~skip).sum().clamp_min(1)
    if env.cfg.beamdojo_curriculum:
        streak=torch.where(wins,env.beamdojo_success_streak[env_ids]+1,0)
        streak=torch.where(skip,env.beamdojo_success_streak[env_ids],streak)
        promotion=(streak>=3)&(env.beamdojo_levels[env_ids]<8)&~skip
        env.beamdojo_levels[env_ids]+=promotion.long()
        env.beamdojo_success_streak[env_ids]=torch.where(promotion,0,streak.clamp_max(3))
    if hasattr(env,'_paper_skip_curriculum_once'):
        env._paper_skip_curriculum_once[env_ids]=False
    env.beamdojo_direction[env_ids]=torch.where(torch.rand(len(env_ids),device=env.device)<.5,-1.,1.)
    refresh_geometry(env)
    env._paper_unsupported[env_ids]=0.
    env._paper_contacts[env_ids]=0.
    env._paper_finish_time[env_ids]=0.
    env._paper_arrived[env_ids]=False
    env._paper_visited[env_ids]=False
    env.beamdojo_qualified_success[env_ids]=False
    if env._paper_last_action is not None:
        env._paper_last_action[env_ids]=0.
        env._paper_previous_previous_action[env_ids]=0.
        env._paper_action_curvature[env_ids]=0.

    return {'Metrics/course_success_rate':rate}
