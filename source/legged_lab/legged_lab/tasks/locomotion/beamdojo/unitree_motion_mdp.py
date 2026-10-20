"""Only the BeamDojo sparse foothold term added to the official Unitree MDP.

No humanoid-gym dense motion rewards, phase reference or action smoothing enter
this profile. Contact threshold and sparse coefficient follow the repository;
the 35-point sole envelope is the existing G1 geometry adaptation. The explicit
physical-alignment profile adds a soft 30% overhang allowance per contacting foot.
Opt-in strict foothold rewards retain the virtual task's full unsafe-point count
on the same physically aligned surface query, without an extra sole-height cost.
"""
from ..unitree_official.mdp import tensor
from .geometry import g1_sole_points, transform_sole_points
from .upstream_terrain import surface_height


def foothold(env):
    robot = env.scene['robot']
    sensor = env.scene.sensors['contact_forces']
    if not hasattr(env, '_unitree_motion_foot_query'):
        names = ['left_ankle_roll_link', 'right_ankle_roll_link']
        env._unitree_motion_foot_query = (
            robot.find_bodies(names, preserve_order=True)[0],
            sensor.find_bodies(names, preserve_order=True)[0],
            g1_sole_points(device=env.device))
    feet, contacts, points = env._unitree_motion_foot_query
    data = robot.data
    soles = transform_sole_points(tensor(data.body_pos_w)[:, feet],
        tensor(data.body_quat_w)[:, feet][..., [3, 0, 1, 2]], points)
    heights, valid = surface_height(env, soles)
    unsafe = heights - tensor(env.scene.env_origins)[:, None, 2:3] < -.1
    contact = tensor(sensor.data.net_forces_w)[:, contacts, 2] > 5.
    if (getattr(getattr(env, 'cfg', None), 'beamdojo_physical_aligned', False)
            and not getattr(env.cfg, 'beamdojo_strict_foothold', False)):
        # This soft reward deadband is per contacting foot, not a physical
        # success criterion. Partial sole overhang can still be a valid step.
        tolerance = getattr(env.cfg, 'beamdojo_physical_surface_tolerance', .06)
        supported = valid & ((soles[..., 2] - heights).abs() <= tolerance)
        missing = (~supported).sum(-1).float()
        allowance = getattr(env.cfg, 'beamdojo_allowed_unsupported_fraction', .30)
        excess = (missing - allowance * points.shape[0]).clamp_min(0.)
        return -(excess * contact).sum(-1)
    return -(unsafe & contact[..., None]).sum(dim=(-1, -2)).float()
