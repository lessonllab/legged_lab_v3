"""Native G1 BeamDojo terms; fixed forward-course adaptation, ideal height map.

Physics uses IsaacLab's XYZW quaternions. Sole sampling utilities explicitly
receive WXYZ. The actor has no route index, next-foot label or target position.
"""
import math
import numpy as np
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from ..unitree_official import mdp as official
from .geometry import g1_sole_points, transform_sole_points, foothold_support
from .perception import grid_points_world, ElevationMapHold
from .map_noise import IdealMapNoise, MapNoiseCfg


def __getattr__(name):
    return getattr(official, name)


def course_rectangles(stone_count=2):
    """(cx, cy, sx, sy, top_z) relative to reset origin.

    Staggered-course dimensions are retained, but entry is shortened: spawn is
    .35 m before the start edge (legacy course: 1 m). Platform-edge gaps .03 m.
    """
    if stone_count not in (2, 4, 6):
        raise ValueError("BeamDojo supports 2, 4 or 6 stepping stones")
    rectangles = [(-.65, 0., 2., 1.4, 0.)]  # start ends at x=.35
    for i in range(stone_count):
        rectangles.append((.545 + .285 * i, .15 * (-1)**i, .33, .21, 0.))
    end_start = .74 + .285 * (stone_count - 1)
    rectangles.append((end_start + 1.2, 0., 2.4, 1.4, 0.))
    return rectangles


def course_terrain(difficulty, cfg):
    """Identical physical/virtual support geometry; Stage 2 has no ground cover."""
    import trimesh
    origin = np.array((2., cfg.size[1] / 2, 0.))
    if cfg.beamdojo_stage == 1:
        mesh = trimesh.creation.box(extents=(cfg.size[0], cfg.size[1], .1))
        mesh.apply_translation((cfg.size[0]/2, cfg.size[1]/2, -.05))
        return [mesh], origin
    meshes = []
    for x, y, sx, sy, top in course_rectangles(cfg.beamdojo_num_stones):
        mesh = trimesh.creation.box(extents=(sx, sy, .5))
        mesh.apply_translation((origin[0]+x, origin[1]+y, top-.25))
        meshes.append(mesh)
    return meshes, origin


def surface_height(env, points):
    """Separate virtual perception/safety surface, including explicit voids."""
    if getattr(env.cfg,'beamdojo_paper_aligned',False):
        from .paper_mdp import surface_height as paper_surface
        return paper_surface(env,points)
    origins = official.tensor(env.scene.env_origins)
    view = (env.num_envs,) + (1,) * (points.ndim - 2) + (3,)
    local = points - origins.reshape(view)
    heights = torch.zeros_like(local[..., 0]) + origins[:, 2].reshape(view[:-1]) - 1.
    valid = torch.zeros_like(heights, dtype=torch.bool)
    for x, y, sx, sy, top in course_rectangles(env.cfg.beamdojo_num_stones):
        inside = (local[..., 0]-x).abs() <= sx/2
        inside &= (local[..., 1]-y).abs() <= sy/2
        heights = torch.where(inside, top + origins[:, 2].reshape(view[:-1]), heights)
        valid |= inside
    return heights, valid


def ideal_heightmap(env):
    """15x15, 0.1 m ideal map held at 10 Hz; not a calibrated LiDAR result."""
    robot = env.scene['robot'].data
    pos = official.tensor(robot.root_pos_w)
    quat = official.tensor(robot.root_quat_w)  # Sim6 XYZW
    yaw = torch.atan2(2*(quat[:,3]*quat[:,2]+quat[:,0]*quat[:,1]),
                      1-2*(quat[:,1].square()+quat[:,2].square()))
    if not hasattr(env, '_beamdojo_map'):
        env._beamdojo_map = ElevationMapHold(env.num_envs, env.device, period=.1)
        env._beamdojo_map_tick = -1
        env._beamdojo_map_clock_tick = -1
        noise_cfg=MapNoiseCfg(**env.cfg.beamdojo_paper_noise) if getattr(env.cfg,'beamdojo_paper_aligned',False) else MapNoiseCfg()
        env._beamdojo_map_noise = IdealMapNoise(env.num_envs, env.device,cfg=noise_cfg)
    tick = env.common_step_counter
    if env._beamdojo_map_tick != tick:
        measured_yaw = yaw + env._beamdojo_map_noise.yaw if env.cfg.beamdojo_map_noise else yaw
        points = grid_points_world(pos, measured_yaw, size=15, resolution=.1)
        heights, support = surface_height(env, points)
        # Analytic void is known empty space, not a missing sensor return.
        # Hold the complete sampled observation (including its sampled pose).
        dt = env.step_dt if env._beamdojo_map_clock_tick != tick else 0.
        hold = env._beamdojo_map
        due = hold.due | (hold._elapsed+dt >= hold.period-1e-7)
        ids = due.nonzero().flatten()
        valid = torch.ones_like(support)
        repeat = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        if ids.numel() and env.cfg.beamdojo_map_noise:
            heights[ids], valid[ids] = env._beamdojo_map_noise(
                heights[ids], valid[ids], support[ids], env_ids=ids, rotate_map=False)
            repeat[ids] = env._beamdojo_map_noise.last_repeated[ids]
        relative = torch.clamp(pos[:, 2:3] - heights - .78, -1., 1.)
        hold.update(relative, valid, dt, repeat=repeat)
        env._beamdojo_map_tick = tick
        env._beamdojo_map_clock_tick = tick
    return env._beamdojo_map.heights.clone()


def gait_cycle(env):
    """One continuous phase shared by observations and every gait reference.

    The nominal .57 m same-foot stride follows the fixed staggered geometry.
    No next-foot target or privileged route state enters the phase generator.
    """
    if not hasattr(env, '_beamdojo_gait_cycle'):
        env._beamdojo_gait_cycle = torch.zeros(env.num_envs, device=env.device)
        env._beamdojo_gait_tick = env.common_step_counter
    if env._beamdojo_gait_tick != env.common_step_counter:
        speed = env.command_manager.get_command('base_velocity')[:,0].clamp_min(0.)
        period = (.57/speed.clamp_min(.2)).clamp(1.,1.9)
        increment = env.step_dt/period * (speed>.1)
        env._beamdojo_gait_cycle = (env._beamdojo_gait_cycle+increment).remainder(1.)
        env._beamdojo_gait_tick = env.common_step_counter
    return env._beamdojo_gait_cycle


def phase(env):
    angle = gait_cycle(env) * (2*math.pi)
    return torch.stack((angle.sin(), angle.cos()), dim=-1)


def course_gait(env, sensor_cfg, threshold=.5):
    cycle = gait_cycle(env)
    expected = torch.stack((cycle.remainder(1.)<threshold,
                            (cycle+.5).remainder(1.)<threshold), dim=-1)
    force = official.tensor(env.scene.sensors[sensor_cfg.name].data.net_forces_w)[:,sensor_cfg.body_ids]
    contact = force.norm(dim=-1)>5.
    return (expected==contact).float().sum(-1) * moving(env)


def low_speed_flight(env, sensor_cfg):
    command = env.command_manager.get_command('base_velocity')
    force = official.tensor(env.scene.sensors[sensor_cfg.name].data.net_forces_w)[:,sensor_cfg.body_ids]
    airborne = ~(force.norm(dim=-1)>5.).any(-1)
    return (airborne & (command[:,:2].norm(dim=-1)<1.2)).float()


def foothold(env, asset_cfg, sensor_cfg):
    robot = env.scene[asset_cfg.name].data
    positions = official.tensor(robot.body_pos_w)[:, asset_cfg.body_ids]
    quats = official.tensor(robot.body_quat_w)[:, asset_cfg.body_ids][..., [3,0,1,2]]
    points = transform_sole_points(positions, quats, g1_sole_points().to(env.device))
    heights, valid = surface_height(env, points)
    force = official.tensor(env.scene.sensors[sensor_cfg.name].data.net_forces_w)
    contact = force[:, sensor_cfg.body_ids].norm(dim=-1) > 5.
    _, unsupported = foothold_support(points, heights, contact, tolerance=.04, surface_valid=valid)
    # Original style contact-only safety penalty, normalized by 35 samples/foot.
    return -unsupported.float() / points.shape[-2]


def natural_arm_swing(env, asset_cfg, amplitude=.25):
    """Bounded shoulder pitch follows the opposite leg's gait phase."""
    data = env.scene[asset_cfg.name].data
    q = official.tensor(data.joint_pos)[:, asset_cfg.joint_ids]
    default = official.tensor(data.default_joint_pos)[:, asset_cfg.joint_ids]
    wave = phase(env)[:, 0] * moving(env)
    # Forward shoulder pitch is negative. During left swing, right arm advances.
    targets = torch.stack((-amplitude*wave, amplitude*wave), dim=1)
    return torch.exp(-((q-default-targets).square().sum(-1))/.08)


def relative_base_height(env, target_height=.78):
    data = env.scene['robot'].data
    z = official.tensor(data.root_pos_w)[:,2] - official.tensor(env.scene.env_origins)[:,2]
    return (z-target_height).square()


def feet_air_time(env, sensor_cfg, threshold=.25):
    sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = official.tensor(sensor.compute_first_contact(env.step_dt))[:, sensor_cfg.body_ids]
    air = official.tensor(sensor.data.last_air_time)[:, sensor_cfg.body_ids]
    return ((air-threshold).clamp(max=.4) * first_contact).sum(-1) * moving(env)


def feet_reference(env, asset_cfg):
    pos = official.tensor(env.scene[asset_cfg.name].data.body_pos_w)[:, asset_cfg.body_ids,2]
    pos = pos - official.tensor(env.scene.env_origins)[:,2:3]
    t = gait_cycle(env)
    swing = torch.stack((-torch.sin(2*math.pi*t), -torch.sin(2*math.pi*(t+.5))),-1).clamp_min(0)
    target = .035 + .10*swing * moving(env)[:, None]
    return torch.exp(-(pos-target).square().sum(-1)/.01)


def _update_arrival(env):
    if not hasattr(env, '_beamdojo_arrived'):
        env._beamdojo_arrived = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    position = official.tensor(env.scene['robot'].data.root_pos_w) - official.tensor(env.scene.env_origins)
    end = course_rectangles(env.cfg.beamdojo_num_stones)[-1]
    # Travel .4 m onto the platform before requesting a stop, leaving room
    # for the trailing foot to clear the final stone/platform gap.
    env._beamdojo_arrived |= ((position[:, 0] > end[0]-end[2]/2+.4)
                              & (position[:, 1].abs() < end[3]/2))
    return env._beamdojo_arrived


class CourseVelocityCommand(UniformVelocityCommand):
    def _update_command(self):
        super()._update_command()
        self.vel_command_b[_update_arrival(self._env)] = 0.


def moving(env):
    return env.command_manager.get_command('base_velocity')[:, :2].norm(dim=-1) > .1


def terrain_observation(env):
    heights = ideal_heightmap(env)
    return torch.cat((heights, env._beamdojo_map.valid.to(heights.dtype),
                      env._beamdojo_map.age[:, None]), dim=-1)


def arm_excess_speed(env, asset_cfg, limit=3.):
    velocity = official.tensor(env.scene[asset_cfg.name].data.joint_vel)[:, asset_cfg.joint_ids]
    return (velocity.abs()-limit).clamp_min(0.).square().sum(-1)


def arm_excess_amplitude(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    q = official.tensor(asset.data.joint_pos)[:, asset_cfg.joint_ids]
    defaults = official.tensor(asset.data.default_joint_pos)[:, asset_cfg.joint_ids]
    pitch = torch.tensor(['shoulder_pitch' in asset.joint_names[i] for i in asset_cfg.joint_ids], device=env.device)
    centers = torch.where(pitch, torch.zeros_like(defaults), defaults)
    bounds = torch.where(pitch, .65, .5)
    return ((q-centers).abs()-bounds).clamp_min(0.).square().sum(-1)


def finished(env, asset_cfg, sensor_cfg):
    if not hasattr(env, '_beamdojo_finish_time'):
        env._beamdojo_finish_time = torch.zeros(env.num_envs, device=env.device)
        env._beamdojo_finish_tick = -1
    if env._beamdojo_finish_tick == env.common_step_counter:
        return env._beamdojo_finish_time >= .6-1e-6
    data = env.scene[asset_cfg.name].data
    arrived = _update_arrival(env)
    # Apply the stop before this transition's rewards. The command term also
    # reapplies it after command resampling, before observations are produced.
    env.command_manager.get_command('base_velocity')[arrived] = 0.
    positions = official.tensor(data.body_pos_w)[:, asset_cfg.body_ids]
    quats = official.tensor(data.body_quat_w)[:, asset_cfg.body_ids][..., [3,0,1,2]]
    soles = transform_sole_points(positions, quats, g1_sole_points(device=env.device))
    local = soles - official.tensor(env.scene.env_origins)[:,None,None,:]
    x,y,sx,sy,z = course_rectangles(env.cfg.beamdojo_num_stones)[-1]
    on_platform = ((local[...,0]-x).abs() < sx/2-.01) & ((local[...,1]-y).abs() < sy/2-.01)
    on_platform &= (local[...,2]-z).abs() < .04
    forces = official.tensor(env.scene.sensors[sensor_cfg.name].data.net_forces_w)[:,sensor_cfg.body_ids]
    both_contact = (forces.norm(dim=-1)>5.).all(-1)
    upright = official.tensor(data.projected_gravity_b)[:,2] < -.85
    slow = official.tensor(data.root_lin_vel_w).norm(dim=-1) < .12
    slow &= official.tensor(data.root_ang_vel_w).norm(dim=-1) < .25
    stable = arrived & on_platform.all(dim=(-1,-2)) & both_contact & upright & slow
    env._beamdojo_finish_time = torch.where(stable, env._beamdojo_finish_time+env.step_dt, 0.)
    env._beamdojo_finish_tick = env.common_step_counter
    return env._beamdojo_finish_time >= .6-1e-6


def left_course(env):
    pos = official.tensor(env.scene['robot'].data.root_pos_w) - official.tensor(env.scene.env_origins)
    return (pos[:,1].abs() > .75) | (pos[:,0] < -.9) | (pos[:,2] < .25)
