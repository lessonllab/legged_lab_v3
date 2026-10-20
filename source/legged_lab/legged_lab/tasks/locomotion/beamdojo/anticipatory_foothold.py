"""Opt-in reward-only constant-velocity swing foothold cost; no actor inputs."""
import torch


def projected_cost(soles, point_velocity, contact_fz, tile_top, command_xy, query, horizon=.08):
    """All shapes N,2,35,3 except forces N,2/top N/command N,2.

    Query returns physical raster heights and validity at predicted world XY.
    This is a kinematic proxy, not a learned future-contact predictor.
    """
    height=soles[...,2].mean(-1)-tile_top[:,None]
    descending=point_velocity[...,2].mean(-1)<-.05
    gate=(contact_fz<=5.) & (height>=.02) & (height<=.20) & descending
    gate &= (command_xy.norm(dim=-1)>.2)[:,None]
    predicted=soles.clone();predicted[...,:2]+=horizon*point_velocity[...,:2]
    heights,valid=query(predicted)
    unsafe=(~valid) | ((heights-tile_top[:,None,None])<-.1)
    fraction=unsafe.float().mean(-1)
    return -(gate.float()*fraction).sum(-1)


def anticipatory_foothold(env):
    from ..unitree_official.mdp import tensor
    from .geometry import g1_sole_points, transform_sole_points
    from .upstream_terrain import surface_height
    if not getattr(env.cfg,'beamdojo_anticipatory_foothold',False):
        raise ValueError('Anticipatory foothold requires explicit research opt-in')
    robot=env.scene['robot'];sensor=env.scene.sensors['contact_forces']
    if not hasattr(env,'_anticipatory_foothold_ids'):
        names=['left_ankle_roll_link','right_ankle_roll_link']
        env._anticipatory_foothold_ids=(robot.find_bodies(names,preserve_order=True)[0],sensor.find_bodies(names,preserve_order=True)[0],g1_sole_points(device=env.device))
    feet,contacts,points=env._anticipatory_foothold_ids;data=robot.data
    pos=tensor(data.body_pos_w)[:,feet]
    soles=transform_sole_points(pos,tensor(data.body_quat_w)[:,feet][...,[3,0,1,2]],points)
    linear=tensor(data.body_link_lin_vel_w)[:,feet]
    angular=tensor(data.body_ang_vel_w)[:,feet]
    velocity=linear[:,:,None,:]+torch.cross(angular[:,:,None,:].expand_as(soles),soles-pos[:,:,None,:],dim=-1)
    return projected_cost(soles,velocity,tensor(sensor.data.net_forces_w)[:,contacts,2],
        tensor(env.scene.env_origins)[:,2],env.command_manager.get_command('base_velocity')[:,:2],
        lambda p:surface_height(env,p,exact=True))
