"""Opt-in two-column training: episode-fixed references and terminal pass gate.

Simulator-free tensor logic. Geometry metadata is injected by the profile; no
actor observations are added and speed resampling never renews the reference.
"""
import math
import torch

BOUNDARY_INSET_M = .75


def enabled(env):
    return bool(getattr(env.cfg, 'beamdojo_corridor_training', False))


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def boundary_mask(env, local_root=None):
    """Version 2 stops along X before the ends of actual retained support.

    Return None for legacy callers, which retain their original map boundary.
    Lateral escape never creates a boundary completion: the persistent lane
    gate remains responsible for rejecting it. Contact and hard-failure checks
    remain owned by the upstream physical-course qualification.
    """
    if getattr(env.cfg, 'beamdojo_corridor_boundary_version', 1) != 2:
        return None
    if local_root is None:
        local_root = (_tensor(env.scene['robot'].data.root_pos_w)
                      - _tensor(env.scene.env_origins))
    if not hasattr(env, '_beamdojo_corridor_safe_x_bounds'):
        bounds = []
        for geometry in env.cfg.beamdojo_corridor_geometry_by_level:
            low, high = geometry['supported_x_bounds_m']
            if not (math.isfinite(low) and math.isfinite(high)
                    and low + BOUNDARY_INSET_M < high - BOUNDARY_INSET_M):
                raise ValueError('Corridor support must contain the 0.75 m boundary inset')
            bounds.append([low + BOUNDARY_INSET_M, high - BOUNDARY_INSET_M])
        env._beamdojo_corridor_safe_x_bounds = torch.tensor(
            bounds, device=local_root.device, dtype=local_root.dtype)
    selected = env._beamdojo_corridor_safe_x_bounds[env.beamdojo_levels]
    return (local_root[:, 0] <= selected[:, 0]) | (local_root[:, 0] >= selected[:, 1])


def _yaw(env):
    x, y, z, w = _tensor(env.scene['robot'].data.root_quat_w).unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def _ids(env, ids):
    all_ids = torch.arange(env.num_envs, device=env.device)
    return all_ids if ids is None else all_ids[ids] if isinstance(ids, slice) else torch.as_tensor(ids, device=env.device, dtype=torch.long)


def initialize(env):
    if hasattr(env, '_beamdojo_corridor_state'):
        return env._beamdojo_corridor_state
    position = _tensor(env.scene['robot'].data.root_pos_w)
    vector = lambda: torch.zeros_like(position[:, 0])
    flag = lambda: torch.zeros(env.num_envs, dtype=torch.bool, device=position.device)
    state = {name: vector() for name in ('reference_yaw', 'heading_error', 'lateral_error',
             'signed_progress_m', 'heading_square_sum', 'heading_peak_rad', 'samples',
             'lateral_peak_m')}
    state.update(reference_xy=torch.zeros_like(position[:, :2]),
                 initialized=flag(), departed_platform=flag(), left_lane=flag(), failed=flag(),
                 tick=torch.full((env.num_envs,), -1, dtype=torch.long, device=position.device))
    geometry = env.cfg.beamdojo_corridor_geometry_by_level
    state['geometry'] = torch.tensor([
        [g['recommended_spawn_y_m'], *g['corridor_y_bounds_m'],
         *g['platform_x_bounds_m'], *g['platform_y_bounds_m']]
        for g in geometry], dtype=position.dtype, device=position.device)
    env._beamdojo_corridor_state = state
    return state


def reset(env, env_ids=None):
    """Capture reference only after all native events and command resets finish."""
    state = initialize(env)
    ids = _ids(env, env_ids)
    state['reference_xy'][ids] = _tensor(env.scene['robot'].data.root_pos_w)[ids, :2]
    state['reference_yaw'][ids] = _yaw(env)[ids]
    for name in ('heading_error', 'lateral_error', 'signed_progress_m', 'heading_square_sum',
                 'heading_peak_rad', 'samples', 'lateral_peak_m'):
        state[name][ids] = 0
    for name in ('departed_platform', 'left_lane', 'failed'):
        state[name][ids] = False
    state['initialized'][ids] = True
    # Suppress repeated diagnostics on the reset frame: it belongs to the old
    # episode and its pre-reset snapshot is already retained by the env hook.
    state['tick'][ids] = env.common_step_counter
    return state


def update(env):
    state = initialize(env)
    if getattr(env, '_beamdojo_corridor_update_tick', None) == env.common_step_counter:
        return state
    xy = _tensor(env.scene['robot'].data.root_pos_w)[:, :2]
    current_yaw = _yaw(env)
    missing = ~state['initialized']
    state['reference_xy'].copy_(torch.where(missing[:, None], xy, state['reference_xy']))
    state['reference_yaw'].copy_(torch.where(missing, current_yaw, state['reference_yaw']))
    fresh = (state['tick'] != env.common_step_counter) & ~missing
    state['initialized'].fill_(True)
    delta = xy - state['reference_xy']
    yaw = state['reference_yaw']
    error = current_yaw - yaw
    heading = torch.atan2(error.sin(), error.cos())
    lateral = -yaw.sin() * delta[:, 0] + yaw.cos() * delta[:, 1]
    progress = yaw.cos() * delta[:, 0] + yaw.sin() * delta[:, 1]
    # Do not overwrite final-transition measurements after a same-tick reset.
    for key, value in (('heading_error', heading), ('lateral_error', lateral),
                       ('signed_progress_m', progress)):
        state[key].copy_(torch.where(fresh, value, state[key]))
    geometry = state['geometry'][env.beamdojo_levels]
    local = xy - _tensor(env.scene.env_origins)[:, :2]
    outside_platform = ((local[:, 0] < geometry[:, 3]) | (local[:, 0] > geometry[:, 4])
                        | (local[:, 1] < geometry[:, 5]) | (local[:, 1] > geometry[:, 6]))
    state['departed_platform'] |= fresh & outside_platform
    outside_lane = (local[:, 1] < geometry[:, 1]) | (local[:, 1] > geometry[:, 2])
    state['left_lane'] |= fresh & state['departed_platform'] & outside_lane
    finite = torch.isfinite(xy).all(-1) & torch.isfinite(heading)
    failed = getattr(env, 'beamdojo_course_failed', torch.zeros_like(finite)) | ~finite
    manager = getattr(env, 'termination_manager', None)
    if manager is not None and hasattr(manager, 'terminated'):
        failed = failed | _tensor(manager.terminated)
    state['failed'] |= fresh & failed
    state['heading_square_sum'] += torch.where(fresh, heading.square(), 0.)
    state['samples'] += fresh
    state['heading_peak_rad'].copy_(torch.where(fresh,
        torch.maximum(state['heading_peak_rad'], heading.abs()), state['heading_peak_rad']))
    state['lateral_peak_m'].copy_(torch.where(fresh,
        torch.maximum(state['lateral_peak_m'], lateral.abs()), state['lateral_peak_m']))
    state['tick'].fill_(env.common_step_counter)
    env._beamdojo_corridor_update_tick = env.common_step_counter
    return state


def qualified_mask(env, original_pass=None):
    """Gate the original physical pass; no intermediate milestone is a pass."""
    state = update(env)
    original = env.beamdojo_qualified_success if original_pass is None else original_pass
    manager = getattr(env, 'termination_manager', None)
    if manager is not None and hasattr(manager, 'terminated'):
        # Termination terms may finish computing after the first same-tick
        # geometry update; hard failures must still veto the final pass.
        # Native termination buffers can still describe the previous episode
        # while same-tick reset diagnostics run. No new sample means no veto.
        state['failed'] |= _tensor(manager.terminated) & (state['samples'] > 0)
    timeout = env.episode_length_buf >= env.max_episode_length
    boundary = getattr(env, '_course_boundary', torch.zeros_like(timeout))
    rms = (state['heading_square_sum'] / state['samples'].clamp_min(1)).sqrt()
    return (original & (timeout | boundary) & (state['signed_progress_m'] >= 3.)
            & state['departed_platform'] & ~state['left_lane'] & ~state['failed']
            & (rms <= math.radians(15)) & (state['heading_peak_rad'] <= math.radians(30)))


def _fixed_velocity(env, asset_cfg=None):
    state = update(env)
    robot = env.scene['robot' if asset_cfg is None else asset_cfg.name]
    velocity = _tensor(robot.data.root_lin_vel_w)
    yaw = state['reference_yaw']
    return torch.stack((yaw.cos() * velocity[:, 0] + yaw.sin() * velocity[:, 1],
                        -yaw.sin() * velocity[:, 0] + yaw.cos() * velocity[:, 1]), -1)


def track_lin_vel_xy_yaw_frame_exp(env, std, command_name, asset_cfg=None):
    """Official reward call signature, using the immutable reset yaw frame."""
    if std <= 0:
        raise ValueError('Velocity tracking std must be positive')
    command = _tensor(env.command_manager.get_command(command_name))
    error = (command[:, :2] - _fixed_velocity(env, asset_cfg)).square().sum(-1)
    return torch.exp(-error / std ** 2)



def track_lin_vel_xy_dual_frame_exp(env, std, command_name, asset_cfg=None):
    """Require velocity tracking in both reset and current body-yaw frames.

    The maximum squared error preserves the original reward when both frames
    coincide, while preventing either a turned trajectory or sideways walking
    along the desired world axis from receiving full tracking credit.
    """
    if std <= 0:
        raise ValueError('Velocity tracking std must be positive')
    command = _tensor(env.command_manager.get_command(command_name))[:, :2]
    fixed = _fixed_velocity(env, asset_cfg)
    robot = env.scene['robot' if asset_cfg is None else asset_cfg.name]
    velocity = _tensor(robot.data.root_lin_vel_w)
    yaw = _yaw(env)
    current = torch.stack((yaw.cos() * velocity[:, 0] + yaw.sin() * velocity[:, 1],
                           -yaw.sin() * velocity[:, 0] + yaw.cos() * velocity[:, 1]), -1)
    reference_error = (command - fixed).square().sum(-1)
    current_error = (command - current).square().sum(-1)
    return torch.exp(-torch.maximum(reference_error, current_error) / std ** 2)


def heading_error_l2(env, scale=.5):
    if scale <= 0:
        raise ValueError('Heading scale must be positive')
    return (update(env)['heading_error'] / scale).clamp(-2., 2.).square()


def lateral_error_l2(env, scale=.3):
    if scale <= 0:
        raise ValueError('Lateral scale must be positive')
    return (update(env)['lateral_error'] / scale).clamp(-2., 2.).square()


def signed_progress(env, normalization_speed=1., command_name='base_velocity'):
    """Upper-capped signed velocity; full backward debit prevents excursion credit."""
    if normalization_speed <= 0:
        raise ValueError('Progress normalization speed must be positive')
    moving = _tensor(env.command_manager.get_command(command_name))[:, 0].abs() >= .1
    return (_fixed_velocity(env)[:, 0] / normalization_speed).clamp_max(1.) * moving


def signed_progress_velocity(env, command_name='base_velocity'):
    return signed_progress(env, command_name=command_name)


def transition_metrics(env):
    state = update(env)
    metrics = {name: state[name].detach().clone() for name in (
        'heading_error', 'lateral_error', 'signed_progress_m', 'heading_peak_rad',
        'lateral_peak_m', 'departed_platform', 'left_lane', 'failed', 'samples')}
    metrics['heading_rms_rad'] = (state['heading_square_sum'] / state['samples'].clamp_min(1)).sqrt().detach().clone()
    metrics['qualified'] = qualified_mask(env).detach().clone()
    return metrics


def diagnostics(env):
    metrics = transition_metrics(env)
    return {'Corridor/' + name: value.float().mean() for name, value in metrics.items()}


def configure_reset(cfg):
    """Wrap native reset: choose 0/pi and source-pair-centered Y before observations."""
    if getattr(cfg, '_beamdojo_corridor_reset_configured', False):
        return
    base = cfg.events.reset_base.func

    class CorridorReset(base):
        def __call__(self, env, env_ids, pose_range, velocity_range):
            limited = dict(pose_range)
            limited['y'] = (-.05, .05)
            super().__call__(env, env_ids, limited, velocity_range)
            ids = _ids(env, env_ids)
            robot = env.scene['robot']
            position = _tensor(robot.data.root_pos_w)[ids].clone()
            quat = _tensor(robot.data.root_quat_w)[ids].clone()
            geometry = initialize(env)['geometry'][env.beamdojo_levels[ids]]
            origins = _tensor(env.scene.env_origins)[ids]
            position[:, 1] = origins[:, 1] + geometry[:, 0] + (position[:, 1] - origins[:, 1]).clamp(-.05, .05)
            target = torch.randint(0, 2, (len(ids),), device=env.device) * math.pi
            x, y, z, w = quat.unbind(-1)
            old = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
            half = (target - old) / 2
            c, s = half.cos(), half.sin()
            quat = torch.stack((c*x - s*y, s*x + c*y, c*z + s*w, c*w - s*z), -1)
            robot.write_root_pose_to_sim_index(root_pose=torch.cat((position, quat), -1), env_ids=ids)

    cfg.events.reset_base.func = CorridorReset
    cfg._beamdojo_corridor_reset_configured = True
