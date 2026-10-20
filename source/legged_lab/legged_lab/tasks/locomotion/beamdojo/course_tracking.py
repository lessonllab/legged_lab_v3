"""Persistent straight-command references, independent of Isaac Sim imports.

The reference is the robot's world position and heading when a straight segment
starts. It is never corrected toward a drifting robot. Straight command speed
changes (including forward/reverse) preserve it; turn, lateral, and stand
commands end it. Call ``update_state`` after command updates and before rewards.
Call ``reset_state`` after physics AND command reset, including spawn relocation.
References are episode-local runtime state and must not enter policy checkpoints.
"""

import torch


def _tensor(value):
    """Bridge native Sim6 WarpArray data without copying device buffers."""
    return value if isinstance(value, torch.Tensor) else value.torch


def _random(env, reference, *, normal=False):
    sampler = torch.randn if normal else torch.rand
    return sampler(reference.shape, device=reference.device, dtype=reference.dtype,
                   generator=env._beamdojo_tracking_generator)


def _yaw(env):
    # Native Sim6 quaternions are XYZW, not the older Isaac Lab WXYZ layout.
    x, y, z, w = _tensor(env.scene['robot'].data.root_quat_w).unbind(-1)
    return torch.atan2(2. * (w * z + x * y), 1. - 2. * (y.square() + z.square()))


def _straight(env):
    command = _tensor(env.command_manager.get_command('base_velocity'))
    return ((command[:, 0].abs() >= .1)
            & (command[:, 1].abs() <= 1e-4)
            & (command[:, 2].abs() <= 1e-4))


def ensure_state(env):
    """Allocate tensor-only state once, without capturing an early spawn pose."""
    if not hasattr(env, '_beamdojo_course_tracking'):
        position = _tensor(env.scene['robot'].data.root_pos_w)
        # Keep sensor augmentation independent of reset/event/policy RNG streams.
        env._beamdojo_tracking_generator = torch.Generator(device=position.device)
        env._beamdojo_tracking_generator.manual_seed(int(getattr(env.cfg, 'seed', 0) or 0) + 190731)
        env._beamdojo_course_tracking = {
            'reference_xy': torch.zeros_like(position[:, :2]),
            'reference_yaw': torch.zeros_like(position[:, 0]),
            'active': torch.zeros(env.num_envs, device=position.device, dtype=torch.bool),
            'heading_error': torch.zeros_like(position[:, 0]),
            'lateral_error': torch.zeros_like(position[:, 0]),
            'imu_heading': torch.zeros_like(position[:, 0]),
            'gyro_bias': torch.zeros_like(position[:, 0]),
            'imu_tick': torch.full((env.num_envs,), -1, device=position.device, dtype=torch.long),
        }
    return env._beamdojo_course_tracking


def update_state(env):
    """Refresh errors; repeated calls never move an existing segment reference.

    Truth errors refresh on every call; IMU heading integrates once per control
    tick. Commands and resets may change within a tick. IMU inputs alone control
    the deployed heading estimate; world pose only controls privileged rewards.
    """
    state = ensure_state(env)
    active = _straight(env)
    starting = active & ~state['active']
    tick = env.common_step_counter
    fresh = state['imu_tick'] != tick
    cfg = env.cfg
    bias_bound = getattr(cfg, 'tracking_gyro_bias', .001)
    noise_std = getattr(cfg, 'tracking_gyro_noise_std', .01)
    # Draw only once each control tick, not once per observation/reward term.
    if getattr(env, '_beamdojo_tracking_noise_tick', None) != tick:
        env._beamdojo_tracking_noise_tick = tick
        env._beamdojo_tracking_gyro_noise = _random(env, state['imu_heading'], normal=True) * noise_std
        env._beamdojo_tracking_bias_sample = (_random(env, state['imu_heading']) * 2. - 1.) * bias_bound
    state['gyro_bias'].copy_(torch.where(starting, env._beamdojo_tracking_bias_sample, state['gyro_bias']))
    robot = env.scene['robot'].data
    gravity = _tensor(robot.projected_gravity_b)
    omega = _tensor(robot.root_ang_vel_b)
    # Euler yaw rate from body gyro and projected gravity; XYZW pose is not an
    # input here. The denominator clamp bounds gain near the pitch singularity.
    rate = -(omega[:, 1] * gravity[:, 1] + omega[:, 2] * gravity[:, 2]) / (
        gravity[:, 1].square() + gravity[:, 2].square()).clamp_min(.25)
    rate = rate + state['gyro_bias'] + env._beamdojo_tracking_gyro_noise
    integrate = fresh & active & ~starting
    estimate = state['imu_heading'] + torch.where(integrate, rate * env.step_dt, 0.)
    estimate = torch.atan2(torch.sin(estimate), torch.cos(estimate))
    state['imu_heading'].copy_(torch.where(active & ~starting, estimate, 0.))
    state['imu_tick'].fill_(tick)
    xy = _tensor(env.scene['robot'].data.root_pos_w)[:, :2]
    yaw = _yaw(env)
    state['reference_xy'].copy_(torch.where(starting[:, None], xy, state['reference_xy']))
    state['reference_yaw'].copy_(torch.where(starting, yaw, state['reference_yaw']))
    state['active'].copy_(active)
    delta_yaw = yaw - state['reference_yaw']
    heading = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))
    delta_xy = xy - state['reference_xy']
    # Positive cross-track means left of the initial body-forward direction.
    lateral = (-torch.sin(state['reference_yaw']) * delta_xy[:, 0]
               + torch.cos(state['reference_yaw']) * delta_xy[:, 1])
    state['heading_error'].copy_(torch.where(active, heading, 0.))
    state['lateral_error'].copy_(torch.where(active, lateral, 0.))
    return state


def reset_state(env, env_ids=None):
    """Capture only reset environments after their final spawn/command reset."""
    state = ensure_state(env)
    ids = slice(None) if env_ids is None else env_ids
    state['reference_xy'][ids] = _tensor(env.scene['robot'].data.root_pos_w)[ids, :2]
    state['reference_yaw'][ids] = _yaw(env)[ids]
    state['active'][ids] = _straight(env)[ids]
    state['heading_error'][ids] = 0.
    state['lateral_error'][ids] = 0.
    state['imu_heading'][ids] = 0.
    state['imu_tick'][ids] = env.common_step_counter
    bias_bound = getattr(env.cfg, 'tracking_gyro_bias', .001)
    state['gyro_bias'][ids] = (_random(env, state['gyro_bias'][ids]) * 2. - 1.) * bias_bound


def observation(env):
    """Deployable inputs: sin(IMU relative yaw), cos(yaw)-1, active.

    No world pose, absolute heading, cross-track distance, or privileged reference
    enters these values. Deployment must reproduce reset/segment logic, control
    dt, and filtered, bias-calibrated gyro plus projected gravity inputs.
    """
    state = update_state(env)
    return torch.stack((torch.sin(state['imu_heading']),
                        torch.cos(state['imu_heading']) - 1.,
                        state['active'].to(state['heading_error'].dtype)), dim=-1)


def heading_error_l2(env, scale=.5):
    """Nonnegative normalized squared heading error, capped at 4; use weight < 0.

    ``scale`` is radians (default .5). Recommended initial reward weight: -.5.
    RewardManager applies control dt; this function returns a reward rate term.
    """
    if scale <= 0.:
        raise ValueError('Heading error scale must be positive')
    return (update_state(env)['heading_error'] / scale).clamp(-2., 2.).square()


def lateral_error_l2(env, scale=.3):
    """Nonnegative normalized squared cross-track error, capped at 4.

    ``scale`` is meters (default .3). Recommended initial reward weight: -.5.
    Both penalties are zero for intentional turning, lateral motion, or standing.
    """
    if scale <= 0.:
        raise ValueError('Lateral error scale must be positive')
    return (update_state(env)['lateral_error'] / scale).clamp(-2., 2.).square()


def diagnostics(env):
    """Current straight-segment errors; sample before autoreset for logging."""
    state = update_state(env)
    count = state['active'].sum().clamp_min(1)
    return {
        'CourseTracking/active_fraction': state['active'].float().mean(),
        'CourseTracking/heading_abs_rad': state['heading_error'].abs().sum() / count,
        'CourseTracking/lateral_abs_m': state['lateral_error'].abs().sum() / count,
        'CourseTracking/imu_heading_abs_rad': state['imu_heading'].abs().sum() / count,
    }


def snapshots(env):
    """Detached copies for diagnostics/tests, never checkpoint curriculum data."""
    return {name: value.detach().clone() for name, value in update_state(env).items()}
