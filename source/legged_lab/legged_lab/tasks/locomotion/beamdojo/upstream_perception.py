"""Current-frame virtual/physical elevation samples, without paper noise.

custom/humanoid_env.py compute_observations and base/legged_robot.py
_get_heights: yaw-aligned 5cm grid, three-neighbour minimum, no map noise or
10Hz hold. New runs use the BeamDojo paper's base-centered 15x15, 10cm grid.
Legacy checkpoints retain their original 187-point or 5cm-grid footprint.
The value transform and virtual terrain query remain BeamDojo adaptations.
"""
import torch
from .upstream_terrain import surface_height


def scanner_spec(cfg):
    if (getattr(cfg, 'beamdojo_unitree_motion_aligned', False)
            and getattr(cfg, 'beamdojo_scanner_profile', 'beamdojo_paper') == 'beamdojo_paper'):
        return {'profile': 'beamdojo_paper', 'shape_xy': [15, 15], 'size_xy_m': [1.4, 1.4],
                'resolution_m': .1, 'count': 225, 'ordering': 'ij', 'frame': 'root'}
    official = (getattr(cfg, 'beamdojo_unitree_motion_aligned', False)
                and getattr(cfg, 'beamdojo_scanner_profile', 'unitree_g1') == 'unitree_g1')
    return ({'profile': 'unitree_g1', 'shape_xy': [17, 11], 'size_xy_m': [1.6, 1.0],
             'resolution_m': .1, 'count': 187, 'ordering': 'xy', 'frame': 'torso_link'}
            if official else
            {'profile': 'legacy_upstream', 'shape_xy': [15, 15], 'size_xy_m': [.7, .7],
             'resolution_m': .05, 'count': 225, 'ordering': 'ij', 'frame': 'root'})


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def sampling_points_world(env):
    """Versioned scanner footprint/order, shared by observations and visualization.

    The 20 m offset matches Unitree's downward ray starts. Surface queries use
    XY only, retaining virtual gaps that do not exist in stage-1 collision.
    """
    spec = scanner_spec(getattr(env, 'cfg', None))
    robot = env.scene['robot']
    data = robot.data
    if spec['frame'] == 'torso_link':
        if not hasattr(env, '_beamdojo_scanner_body_id'):
            ids, _ = robot.find_bodies('torso_link', preserve_order=True)
            if len(ids) != 1:
                raise ValueError('Unitree scanner requires exactly one torso_link')
            env._beamdojo_scanner_body_id = ids[0]
        body = env._beamdojo_scanner_body_id
        pos = _tensor(data.body_pos_w)[:, body]
        quat = _tensor(data.body_quat_w)[:, body]
    else:
        pos, quat = _tensor(data.root_pos_w), _tensor(data.root_quat_w)
    yaw = torch.atan2(2*(quat[:,3]*quat[:,2]+quat[:,0]*quat[:,1]),
                      1-2*(quat[:,1].square()+quat[:,2].square()))
    if getattr(env, '_beamdojo_scanner_grid_profile', None) != spec['profile']:
        nx, ny = spec['shape_xy']
        sx, sy = spec['size_xy_m']
        x = -sx/2 + torch.arange(nx, device=env.device)*spec['resolution_m']
        y = -sy/2 + torch.arange(ny, device=env.device)*spec['resolution_m']
        x, y = torch.meshgrid(x, y, indexing=spec['ordering'])
        env._upstream_heightmap_grid = (x.flatten(), y.flatten())
        env._beamdojo_scanner_grid_profile = spec['profile']
    x, y = env._upstream_heightmap_grid
    points = pos[:, None, :].expand(-1, spec['count'], -1).clone()
    points[:, :, 0] += yaw.cos()[:, None]*x-yaw.sin()[:, None]*y
    points[:, :, 1] += yaw.sin()[:, None]*x+yaw.cos()[:, None]*y
    if spec['frame'] == 'torso_link':
        points[:, :, 2] += 20.
    return points


def heightmap(env):
    tick = getattr(env, 'common_step_counter', None)
    cached = getattr(env, '_upstream_heightmap_cache', None)
    if tick is not None and cached is not None and cached[0] == tick:
        return cached[1]
    pos = _tensor(env.scene['robot'].data.root_pos_w)
    points = sampling_points_world(env)
    heights,_=surface_height(env,points)
    result = (pos[:,2:3]-.5-heights).clamp(-1.,1.)*5.
    env._upstream_heightmap_cache = (tick, result)
    return result
