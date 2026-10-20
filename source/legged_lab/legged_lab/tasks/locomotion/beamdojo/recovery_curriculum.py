"""Training-only recovery curriculum; never constrains preview/evaluation tiles."""
import math
from copy import deepcopy, copy
import torch
from . import course_schedule


def enabled(cfg):
    return bool(getattr(cfg, 'beamdojo_recovery', False) and getattr(cfg, 'beamdojo_curriculum', False))


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def _yaw_frame_velocity(data):
    """Equivalent to official quat_apply_inverse(yaw_quat(q), world_velocity).

    Native Sim6 quaternions are XYZW. Project only world yaw, so body roll/pitch
    and world vertical motion cannot contaminate planar command qualification.
    Expressing this rotation in torch keeps the curriculum CPU-testable without
    importing the simulator while matching the official reward's coordinates.
    """
    x, y, z, w = _tensor(data.root_quat_w).unbind(-1)
    yaw = torch.atan2(2. * (w * z + x * y), 1. - 2. * (y.square() + z.square()))
    velocity = _tensor(data.root_lin_vel_w)
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * velocity[:, 0] + s * velocity[:, 1],
                        -s * velocity[:, 0] + c * velocity[:, 1]), dim=-1)


def _settings(cfg):
    cap = int(getattr(cfg, 'recovery_max_level', 2))
    yaw = float(getattr(cfg, 'recovery_yaw_half_range', math.pi / 12))
    replay = float(getattr(cfg, 'recovery_easy_replay_fraction', .2))
    if not 0 <= cap <= course_schedule.max_level(cfg) or not 0. <= yaw <= math.pi or not 0. <= replay <= 1.:
        raise ValueError('Invalid recovery level cap, yaw range, or replay fraction')
    return cap, yaw, replay


def ensure_state(env):
    if not hasattr(env, '_beamdojo_recovery_state'):
        zero = torch.zeros(env.num_envs, device=env.device)
        env._beamdojo_recovery_state = {name: zero.clone() for name in (
            'moving_seconds', 'expected_distance', 'directed_distance', 'xy_squared_error', 'single_seconds')}
        state = env._beamdojo_recovery_state
        state['invalid'] = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        state['tick'] = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        state['target_levels'] = env.beamdojo_levels.clone().clamp(0, _settings(env.cfg)[0])
        state['episode_challenge'] = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        state['level_episode_count'] = torch.zeros(course_schedule.level_count(env.cfg), device=env.device, dtype=torch.long)
        state['challenge_episode_count'] = torch.zeros_like(state['level_episode_count'])
        env._beamdojo_recovery_generator = torch.Generator(device=env.device)
        env._beamdojo_recovery_generator.manual_seed(int(getattr(env.cfg, 'seed', 0) or 0) + 6700417)
    return env._beamdojo_recovery_state


def _challenge_settings(cfg):
    fraction = float(getattr(cfg, 'recovery_challenge_fraction', 0.))
    levels = tuple(getattr(cfg, 'recovery_challenge_levels', (7, 8, 9, 10)))
    cap, _, replay = _settings(cfg)
    if (not math.isfinite(fraction) or not 0. <= fraction <= 1. - replay
            or (fraction > 0 and (not levels or len(set(levels)) != len(levels)
                or any(type(level) is not int or not 0 <= level <= cap for level in levels)))):
        raise ValueError('Invalid challenge fraction/levels or overlapping sampling probabilities')
    return fraction, levels


def apply_challenge_sampling(env, fraction=.2, levels=(7, 8, 9, 10)):
    """Explicit migration after restore, before reset; changes only settings.

    The normal upstream reset always invokes before_reset, even when its
    skip-once flag makes scoring ineligible. That reset samples the first
    challenge episodes and refreshes their geometry without grading old data.
    """
    if not enabled(env.cfg):
        raise ValueError('Challenge sampling requires training recovery curriculum')
    candidate = copy(env.cfg)
    candidate.recovery_challenge_fraction = fraction
    candidate.recovery_challenge_levels = tuple(levels)
    _challenge_settings(candidate)
    env.cfg.recovery_challenge_fraction = float(fraction)
    env.cfg.recovery_challenge_levels = tuple(levels)


def resample_levels(env, ids):
    """Select the next tile only; caller must refresh geometry before spawning."""
    state = ensure_state(env)
    cap, _, replay_fraction = _settings(env.cfg)
    challenge_fraction, levels = _challenge_settings(env.cfg)
    target = state['target_levels'][ids]
    generator = env._beamdojo_recovery_generator
    draw = torch.rand(target.shape, device=env.device, generator=generator)
    easy = torch.randint(min(2, cap) + 1, target.shape, device=env.device, generator=generator)
    next_level = torch.where(draw < replay_fraction, torch.minimum(easy, target), target)
    challenge = (draw >= replay_fraction) & (draw < replay_fraction + challenge_fraction)
    if challenge_fraction > 0:
        choices = torch.randint(len(levels), target.shape, device=env.device, generator=generator)
        selected = torch.tensor(levels, device=env.device, dtype=torch.long)[choices]
        next_level = torch.where(challenge, selected, next_level)
    env.beamdojo_levels[ids] = next_level
    state['episode_challenge'][ids] = challenge


def reset_state(env, env_ids=None):
    if not enabled(env.cfg):
        return
    state = ensure_state(env)
    ids = slice(None) if env_ids is None else env_ids
    for key in ('moving_seconds', 'expected_distance', 'directed_distance', 'xy_squared_error', 'single_seconds', 'invalid'):
        state[key][ids] = 0
    # Never count stale pre-reset velocity/contact data in a new episode.
    state['tick'][ids] = env.common_step_counter


def update(env):
    """Accumulate each moving-command transition once, before command resampling."""
    if not enabled(env.cfg):
        return
    state = ensure_state(env)
    fresh = state['tick'] != env.common_step_counter
    command = _tensor(env.command_manager.get_command('base_velocity'))[:, :2]
    velocity = _yaw_frame_velocity(env.scene['robot'].data)
    speed = command.norm(dim=-1)
    moving = (speed > .2 if getattr(env.cfg, 'recovery_neutral_unattempted', False)
              else speed >= .2)
    finite = torch.isfinite(command).all(-1) & torch.isfinite(velocity).all(-1)
    state['invalid'] |= fresh & moving & ~finite
    active = fresh & moving & finite
    if hasattr(env, '_course_boundary'):
        active &= ~env._course_boundary
    sensor = env.scene.sensors['contact_forces']
    if not hasattr(env, '_recovery_contact_ids'):
        env._recovery_contact_ids = sensor.find_bodies(
            ['left_ankle_roll_link', 'right_ankle_roll_link'], preserve_order=True)[0]
    contact = _tensor(sensor.data.net_forces_w)[:, env._recovery_contact_ids, 2] > 5.
    duration = active.to(velocity.dtype) * env.step_dt
    directed = (velocity * command).sum(-1) / speed.clamp_min(1e-8)
    squared_error = (velocity - command).square().sum(-1)
    state['moving_seconds'] += duration
    state['expected_distance'] += torch.where(active, speed * env.step_dt, 0.)
    state['directed_distance'] += torch.where(active, directed * env.step_dt, 0.)
    state['xy_squared_error'] += torch.where(active, squared_error * env.step_dt, 0.)
    state['single_seconds'] += duration * (contact.sum(-1) == 1)
    state['tick'].fill_(env.common_step_counter)


def motion_pass(env):
    state = ensure_state(env)
    time = state['moving_seconds'].clamp_min(1e-8)
    finite = torch.stack([torch.isfinite(state[name]) for name in (
        'moving_seconds', 'expected_distance', 'directed_distance', 'xy_squared_error', 'single_seconds')]).all(0)
    if getattr(env.cfg, 'beamdojo_recovery_speed_gate', True):
        progress_ok = state['directed_distance'] / state['expected_distance'].clamp_min(1e-8) >= .6
        speed_ok = state['xy_squared_error'] / time <= .4 ** 2 + 1e-8
    else:
        progress_ok = state['directed_distance'] >= 3. - 1e-6
        speed_ok = finite
    return ((state['moving_seconds'] >= 1. - 1e-6)
            & progress_ok
            & speed_ok & finite
            & (state['single_seconds'] / time >= .10 - 1e-6) & ~state['invalid'])


def remap_curriculum_schedule(saved, source_schedule, target_schedule):
    """Explicit old-topology migration; never reset progression or RNG state."""
    source = course_schedule.profile_name(source_schedule)
    target = course_schedule.profile_name(target_schedule)
    result = deepcopy(saved)
    recorded = saved.get('course_schedule', source)
    if course_schedule.profile_name(recorded) != source:
        raise ValueError('Saved curriculum schedule differs from explicit source schedule')
    if source != target:
        if source != course_schedule.profile_name(None) or target != 'smooth_top_v1':
            raise ValueError('Unsupported curriculum schedule migration')
        def remap(values):
            if any(v < 0 or v > course_schedule.max_level(source) for v in values):
                raise ValueError('Saved levels outside source schedule')
            return [10 if v == 8 else v for v in values]
        result['levels'] = remap(result['levels'])
        recovery = result.get('recovery')
        if recovery is not None:
            recovery['target_levels'] = remap(recovery['target_levels'])
            recovery['max_level'] = remap([recovery['max_level']])[0]
    result['course_schedule'] = target
    return result


def demote_once(env):
    """Explicit performance recovery; preserve cap, command state and RNG."""
    if not enabled(env.cfg):
        raise ValueError('Explicit recovery demotion requires training curriculum')
    state = ensure_state(env)
    def histogram(value):
        return torch.bincount(value, minlength=course_schedule.level_count(env.cfg)).cpu().tolist()
    summary = {'before_actual': histogram(env.beamdojo_levels),
               'before_target': histogram(state['target_levels'])}
    env.beamdojo_levels.sub_(1).clamp_(min=0)
    state['target_levels'].sub_(1).clamp_(min=0)
    env.beamdojo_success_streak.zero_()
    summary.update(after_actual=histogram(env.beamdojo_levels),
                   after_target=histogram(state['target_levels']))
    env.beamdojo_curriculum_demotion_summary = summary
    return summary


def _translation_attempted(state):
    return ((state['moving_seconds'] >= 1. - 1e-6)
            & (state['expected_distance'] >= 3. - 1e-6))


def _true_failure(env, state):
    failed = state['invalid'].clone()
    for name in ('beamdojo_course_failed', 'reset_terminated'):
        if hasattr(env, name):
            failed |= _tensor(getattr(env, name)).reshape(-1).bool()
    return failed


def initialize(env):
    """Explicit recovery transfer: reset curriculum to genuine level-0–2 tiles."""
    if not enabled(env.cfg):
        return
    from . import upstream_terrain
    upstream_terrain.ensure_state(env)
    state = ensure_state(env)
    cap, _, _ = _settings(env.cfg)
    levels = torch.randint(min(2, cap) + 1, (env.num_envs,), device=env.device,
                           generator=env._beamdojo_recovery_generator)
    env.beamdojo_levels[:] = levels
    state['target_levels'][:] = levels
    env.beamdojo_success_streak.zero_()
    env._upstream_skip_curriculum_once = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    reset_state(env)
    upstream_terrain.refresh_geometry(env)


def before_reset(env, ids, course_wins, eligible):
    """Apply joint passage/motion qualification and select the next training tile.

    Easy replay preserves the intended progression level; replay successes do
    not earn promotion at an unattempted harder level. Any eligible failure
    lowers the next progression target one below the level actually attempted.
    """
    if not enabled(env.cfg):
        return course_wins
    state = ensure_state(env)
    cap, _, replay_fraction = _settings(env.cfg)
    if _challenge_settings(env.cfg)[0] > 0:
        # Native before_reset runs before episode_length_buf is cleared.
        # First initialization/reset is not a completed training episode.
        ended = env.episode_length_buf[ids] > 0 if hasattr(env, 'episode_length_buf') else eligible
        levels = env.beamdojo_levels[ids]
        count = course_schedule.level_count(env.cfg)
        state['level_episode_count'] += torch.bincount(levels[ended], minlength=count)
        state['challenge_episode_count'] += torch.bincount(
            levels[ended & state['episode_challenge'][ids]], minlength=count)
    wins = course_wins & motion_pass(env)[ids]
    # Challenge results are diagnostic/training samples, never progression
    # evidence. In particular a failed hard probe cannot raise a low target.
    eligible = eligible & ~state['episode_challenge'][ids]
    if getattr(env.cfg, 'recovery_neutral_unattempted', False):
        attempted = _translation_attempted(state)[ids]
        failed = _true_failure(env, state)[ids]
        # No translational assignment is not a locomotion failure. A real fall
        # remains a failure even before sufficient command exposure accumulates.
        eligible = eligible & (attempted | failed)
        wins &= attempted & ~failed
    actual = env.beamdojo_levels[ids]
    target = state['target_levels'][ids].clamp(max=cap)
    scored = eligible & (actual == target)
    streak = env.beamdojo_success_streak[ids]
    streak = torch.where(scored & wins, streak + 1, streak)
    streak = torch.where(eligible & ~wins, 0, streak)
    promote = scored & wins & (streak >= 3)
    target = torch.where(promote, (target + 1).clamp(max=cap), target)
    target = torch.where(eligible & ~wins, (actual - 1).clamp(min=0, max=cap), target)
    state['target_levels'][ids] = target
    env.beamdojo_success_streak[ids] = torch.where(promote, 0, streak)
    resample_levels(env, ids)
    return wins


def yaw_pose_range(env, pose_range):
    if not enabled(env.cfg):
        return pose_range
    half_range = _settings(env.cfg)[1]
    return {**pose_range, 'yaw': (-half_range, half_range)}


def snapshot(env):
    if not enabled(env.cfg):
        return None
    state = ensure_state(env)
    cap, yaw, replay = _settings(env.cfg)
    result = {'version': 1, 'num_envs': env.num_envs, 'max_level': cap,
            'yaw_half_range': yaw, 'easy_replay_fraction': replay,
            'target_levels': state['target_levels'].cpu().tolist(),
            'generator_state': env._beamdojo_recovery_generator.get_state().tolist()}
    if getattr(env.cfg, 'recovery_neutral_unattempted', False):
        result['neutral_unattempted'] = True
    fraction, levels = _challenge_settings(env.cfg)
    if fraction > 0:
        result.update(challenge_fraction=fraction, challenge_levels=list(levels),
                      episode_challenge=state['episode_challenge'].cpu().tolist(),
                      level_episode_count=state['level_episode_count'].cpu().tolist(),
                      challenge_episode_count=state['challenge_episode_count'].cpu().tolist())
    return result


def restore(env, saved):
    """Restore persistent progression only; episode accumulators start empty."""
    if not enabled(env.cfg) or saved is None:
        return
    if saved.get('version') != 1 or saved.get('num_envs') != env.num_envs:
        raise ValueError('Recovery curriculum version/environment count mismatch')
    target = torch.as_tensor(saved['target_levels'], device=env.device, dtype=torch.long)
    cap, yaw, replay = saved['max_level'], saved['yaw_half_range'], saved['easy_replay_fraction']
    if (not 0 <= cap <= course_schedule.max_level(env.cfg) or not 0. <= yaw <= math.pi or not 0. <= replay <= 1.
            or target.shape != (env.num_envs,) or ((target < 0) | (target > cap)).any()):
        raise ValueError('Invalid saved recovery curriculum')
    env.cfg.recovery_max_level = cap
    env.cfg.recovery_yaw_half_range = yaw
    env.cfg.recovery_easy_replay_fraction = replay
    env.cfg.recovery_challenge_fraction = saved.get('challenge_fraction', 0.)
    env.cfg.recovery_challenge_levels = tuple(saved.get('challenge_levels', (7, 8, 9, 10)))
    _challenge_settings(env.cfg)
    challenge = saved.get('episode_challenge', [False] * env.num_envs)
    if (not isinstance(challenge, list) or len(challenge) != env.num_envs
            or any(type(value) is not bool for value in challenge)
            or (any(challenge) and env.cfg.recovery_challenge_fraction == 0)):
        raise ValueError('Invalid saved challenge episode markers')
    if saved.get('neutral_unattempted', False):
        env.cfg.recovery_neutral_unattempted = True
    state = ensure_state(env)
    state['target_levels'][:] = target
    state['episode_challenge'][:] = torch.tensor(challenge, device=env.device, dtype=torch.bool)
    for name in ('level_episode_count', 'challenge_episode_count'):
        values = saved.get(name, [0] * course_schedule.level_count(env.cfg))
        if (len(values) != course_schedule.level_count(env.cfg)
                or any(type(v) is not int or v < 0 for v in values)):
            raise ValueError('Invalid saved per-level episode counts')
        state[name][:] = torch.tensor(values, device=env.device, dtype=torch.long)
    if (state['challenge_episode_count'] > state['level_episode_count']).any():
        raise ValueError('Challenge episode counts exceed all episode counts')
    env._beamdojo_recovery_generator.set_state(torch.tensor(saved['generator_state'], dtype=torch.uint8))
    reset_state(env)


def diagnostics(env):
    state = ensure_state(env)
    time = state['moving_seconds'].clamp_min(1e-8)
    result = {'Recovery/motion_pass_fraction': motion_pass(env).float().mean(),
            'Recovery/command_progress_ratio': (state['directed_distance'] / state['expected_distance'].clamp_min(1e-8)).mean(),
            'Recovery/xy_velocity_rmse': (state['xy_squared_error'] / time).sqrt().mean(),
            'Recovery/single_support_fraction': (state['single_seconds'] / time).mean()}
    if _challenge_settings(env.cfg)[0] > 0:
        result['Recovery/challenge_fraction'] = state['episode_challenge'].float().mean()
        for level in range(course_schedule.level_count(env.cfg)):
            mask = env.beamdojo_levels == level
            result[f'Recovery/level_{level}_fraction'] = mask.float().mean()
            result[f'Recovery/challenge_level_{level}_fraction'] = (mask & state['episode_challenge']).float().mean()
            result[f'Recovery/level_{level}_episode_count'] = state['level_episode_count'][level].float()
            result[f'Recovery/challenge_level_{level}_episode_count'] = state['challenge_episode_count'][level].float()
    if getattr(env.cfg, 'recovery_neutral_unattempted', False):
        attempted = _translation_attempted(state)
        result.update({
            'Recovery/translation_attempted_fraction': attempted.float().mean(),
            'Recovery/neutral_possible_fraction': (~attempted & ~_true_failure(env, state)).float().mean(),
            'Recovery/moving_command_seconds': state['moving_seconds'].mean(),
            'Recovery/expected_translation_m': state['expected_distance'].mean()})
    return result
