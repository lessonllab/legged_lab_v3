"""Gait-preserving square course with finite-time terminal braking (CPU-safe).

This module leaves motion rewards, policy observations and control gains intact.
Install after binding the extended geometry. Flat motor-skill environments never
participate in the ring task or its curriculum; their map exits are truncations.
"""
import math

import torch

import beamdojo_square_mixed_runtime as mixed
import beamdojo_square_extended_runtime as previous
from beamdojo_flat_rehearsal import flat_mask

REVISION = 'gait_preserved12m_time_brake_v1'
tensor = mixed.tensor
base = mixed.base
_ORIGINAL_ENSURE_STATE = previous._ORIGINAL_ENSURE_STATE
_ORIGINAL_METRICS = previous._ORIGINAL_METRICS
_ORIGINAL_SNAPSHOT = previous._ORIGINAL_SNAPSHOT
_STAGE_COUNTS = ('arrivals', 'parkings', 'stable_parks', 'arrived_then_failed',
                 'arrived_then_timeout', 'timeout_episodes')
_DIAGNOSTIC_SUMS = ('timeout_progress_s_sum', 'timeout_planar_speed_sum',
                    'timeout_contact_count_sum', 'timeout_full_support_count',
                    'maximum_stand_seconds_sum')


def params(env):
    values = dict(inner_half=4., outer_half=6., center_half=.7,
                  diagonal_corner_infill=.5, terminal_runout=1.,
                  terminal_entry_clearance=.20, boundary_margin=.10,
                  route_half_width=.42, end_half_width=.60,
                  terminal_min_speed=.10, terminal_max_deceleration=.8, terminal_min_braking_seconds=.8,
                  stand_seconds=1., flat_boundary_margin=.5)
    values.update(getattr(env.cfg, 'beamdojo_square_extended_params', {}))
    for key, value in values.items():
        if key == 'route_half_width' and value is None:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid extended course parameter: {key}')
    if not values['center_half'] < values['inner_half'] < values['outer_half']:
        raise ValueError('Extended center, inner and outer boundaries must be ordered')
    if values['diagonal_corner_infill'] >= values['inner_half'] - values['center_half']:
        raise ValueError('Corner infill overlaps the central start')
    if (values['terminal_entry_clearance'] + values['terminal_runout'] + .25
            >= values['outer_half'] - values['inner_half']):
        raise ValueError('Terminal runout leaves insufficient stopping room')
    if values['flat_boundary_margin'] >= values['outer_half']:
        raise ValueError('Flat boundary margin exceeds map width')
    return values


def ensure_state(env):
    state = _ORIGINAL_ENSURE_STATE(env)
    if 'terminal_entered' not in state:
        params(env)
        state['terminal_entered'] = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        state['terminal_braking_start_s'] = torch.zeros(env.num_envs, device=env.device)
        state['terminal_speed'] = torch.zeros(env.num_envs, device=env.device)
        state['terminal_braking'] = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        state['terminal_braking_elapsed'] = torch.zeros(env.num_envs, device=env.device)
        state['terminal_braking_cruise'] = torch.zeros(env.num_envs, device=env.device)
        state['terminal_braking_tick'] = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        state['arrived'] = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for name in ('last_progress_s', 'last_planar_speed', 'last_contact_count',
                     'last_full_support', 'maximum_stand_time'):
            state[name] = torch.zeros(env.num_envs, device=env.device)
        env.beamdojo_square_arrived = state['arrived']
        env._square_extended_counts = {
            name: torch.zeros(8, 11, device=env.device,
                              dtype=torch.long if name in _STAGE_COUNTS else torch.float)
            for name in _STAGE_COUNTS + _DIAGNOSTIC_SUMS}
    if 'terminal_braking' not in state:
        state['terminal_braking'] = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        state['terminal_braking_elapsed'] = torch.zeros(env.num_envs, device=env.device)
        state['terminal_braking_cruise'] = torch.zeros(env.num_envs, device=env.device)
        state['terminal_braking_tick'] = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    return state


def goal_start(env):
    p = params(env)
    diagonal = (2 * p['inner_half'] - p['diagonal_corner_infill']) / math.sqrt(2)
    return torch.where(env.square_family == 0, p['inner_half'], diagonal)


def stop_position(env):
    p = params(env)
    return goal_start(env) + p['terminal_runout']


def _geometry(env, measurements):
    p = params(env)
    root = mixed.route_frame(env, measurements['root'])
    corners = mixed.route_frame(env, measurements['corners'])
    goal = goal_start(env)
    full = ((corners[..., 0] >= goal[:, None, None])
            & (corners[..., 1].abs() <= p['end_half_width'])
            & (corners[..., 2].abs() <= .035)
            & (measurements['corners'][..., :2].abs() < p['outer_half']).all(-1)).all((-1, -2))
    full &= measurements['supported'].all(-1) & measurements['corner_supported'].all((-1, -2))
    full &= ~flat_mask(env)
    return root, corners, full


def parking_command(env, cruise, root_s, full_ring):
    """Finite-time smooth braking after a safe full-entry latch.

    Cruise is retained until the speed-dependent braking distance to the 1m
    depth target. Once braking starts, elapsed control time forces exact zero,
    independently of tracking error or the phase of the last swinging foot.
    Full support remains mandatory for stable-success certification.
    """
    p = params(env)
    state = ensure_state(env)
    stone = ~flat_mask(env)
    c = cruise.clamp(.5, 1.)
    goal = goal_start(env)
    entered = stone & full_ring & (root_s >= goal+p['terminal_entry_clearance'])
    state['terminal_entered'] |= entered
    duration = torch.maximum(c.new_full(c.shape, p['terminal_min_braking_seconds']),
                             1.5*c/p['terminal_max_deceleration'])
    braking_distance = .5*c*duration
    start_line = stop_position(env)-braking_distance
    begin = stone & state['terminal_entered'] & (root_s >= start_line)
    first = begin & ~state['terminal_braking']
    state['terminal_braking'][first] = True
    state['terminal_braking_elapsed'][first] = 0.
    state['terminal_braking_cruise'][first] = c[first]
    state['terminal_braking_start_s'][first] = root_s[first]
    tick = int(env.common_step_counter)
    fresh = (state['terminal_braking_tick'] != tick) & state['terminal_braking'] & ~first
    state['terminal_braking_elapsed'][fresh] += env.step_dt
    state['terminal_braking_tick'][state['terminal_braking']] = tick
    bc = state['terminal_braking_cruise'].clamp(.5, 1.)
    duration = torch.maximum(bc.new_full(bc.shape, p['terminal_min_braking_seconds']),
                             1.5*bc/p['terminal_max_deceleration'])
    phase = (state['terminal_braking_elapsed']/duration).clamp(0., 1.)
    blend = phase.square()*(3.-2.*phase)
    taper = bc*(1.-blend)
    state['parked'] |= stone & state['terminal_braking'] & (phase >= 1.-1e-6)
    speed = torch.where(state['terminal_braking'], taper, c)
    speed = torch.where(state['parked'] | ~torch.isfinite(root_s), 0., speed)
    state['parked'][~stone] = False
    state['terminal_entered'][~stone] = False
    state['terminal_braking'][~stone] = False
    state['terminal_speed'].copy_(speed)
    return torch.stack((speed, torch.zeros_like(speed), torch.zeros_like(speed)), -1)


def update_course(env, measurements=None):
    state = ensure_state(env)
    p = params(env)
    tick = int(env.common_step_counter)
    active = state['last_tick'] != tick
    if not bool(active.any()):
        return state
    m = mixed.read_measurements(env) if measurements is None else measurements
    root, corners, full = _geometry(env, m)
    flat = flat_mask(env)
    stone = ~flat
    contact = m['contact_fz'] > 5
    finite = torch.isfinite(root).all(-1) & torch.isfinite(corners).all((-1, -2, -3))
    for key in ('lin', 'ang', 'gravity', 'contact_fz'):
        finite &= torch.isfinite(m[key]).all(-1)
    tilt = torch.acos((-m['gravity'][:, 2]).clamp(-1, 1))
    terms = getattr(env.cfg, 'terminations', None)
    minimum_height = getattr(getattr(terms, 'base_height', None), 'params', {}).get('minimum_height', .2)
    maximum_tilt = getattr(getattr(terms, 'bad_orientation', None), 'params', {}).get('limit_angle', .8)
    central = torch.where(env.square_family == 0, p['center_half'], p['center_half'] * math.sqrt(2))
    departed = state['departed'] | (stone & (root[:, 0] > central))
    boundary = stone & (m['root'][:, :2].abs() >= p['outer_half'] - p['boundary_margin']).any(-1)
    if p['route_half_width'] is not None:
        boundary |= stone & departed & ~(state['arrived'] | full) & (root[:, 1].abs() > p['route_half_width'])
    physical_failure = (root[:, 2] < minimum_height) | (tilt > maximum_tilt) | ~finite
    failure = physical_failure | boundary | env.beamdojo_course_failed
    # The old mixed updater's lane and terminal boundary cannot affect flat modes.
    failure = torch.where(flat, physical_failure | env.beamdojo_course_failed, failure)
    expired = tensor(env.episode_length_buf) >= env.max_episode_length
    command = tensor(env.command_manager.get_command('base_velocity'))
    zero = torch.isfinite(command).all(-1) & (command[:, :2].norm(dim=-1) < 1e-4) & (command[:, 2].abs() < 1e-4)
    stable = full & contact.all(-1) & (m['lin'].norm(dim=-1) < .15) & (m['ang'].norm(dim=-1) < .3) & (tilt < .35)
    stable &= state['parked'] & zero & ~failure & ~expired
    duration = torch.where(stable, env.beamdojo_course_stand_time + env.step_dt, 0.)
    success = stone & (duration >= p['stand_seconds'] - 1e-6) & ~failure & ~expired
    # Arrival is a separate, absorbing curriculum qualification. Reaching the
    # ring does not depend on the later stopping/standing test or speed error.
    state['arrived'][active] |= (full & ~failure)[active]
    first = ~state['initialized']
    delta = (root[:, :2] - state['last_xy']).norm(dim=-1)
    delta = torch.where(first, 0., delta)
    unsafe = contact & ~m['supported']
    safe = contact.any(-1) & ~unsafe.any(-1) & ~failure
    values = {
        'beamdojo_course_stand_time': duration,
        'beamdojo_course_failed': failure,
        'beamdojo_qualified_success': success,
        '_course_finished': success,
        '_course_boundary': boundary,
        '_course_milestone': full,
        'beamdojo_course_distance': env.beamdojo_course_distance + torch.where(safe, delta, 0.),
        'beamdojo_course_contact_samples': env.beamdojo_course_contact_samples + contact.sum(-1),
        'beamdojo_course_unsafe_samples': env.beamdojo_course_unsafe_samples + unsafe.sum(-1),
        'beamdojo_course_valid_contact_seconds': env.beamdojo_course_valid_contact_seconds + safe * env.step_dt,
    }
    state['start_xy'][active & first] = root[active & first, :2]
    values['beamdojo_course_signed_progress'] = root[:, 0] - state['start_xy'][:, 0]
    values['beamdojo_course_net_displacement'] = (root[:, :2] - state['start_xy']).norm(dim=-1)
    for name, value in values.items():
        getattr(env, name)[active] = value[active]
    state['full_ring'][active] = full[active]
    state['last_progress_s'][active] = torch.nan_to_num(root[active, 0], nan=0., posinf=0., neginf=0.)
    state['last_planar_speed'][active] = torch.nan_to_num(m['lin'][active, :2].norm(dim=-1), nan=0., posinf=0., neginf=0.)
    state['last_contact_count'][active] = contact[active].sum(-1).float()
    state['last_full_support'][active] = full[active].float()
    state['maximum_stand_time'][active] = torch.maximum(
        state['maximum_stand_time'][active], duration[active])
    state['departed'][active] = departed[active]
    state['last_xy'][active] = root[active, :2]
    state['initialized'][active] = True
    state['last_tick'][active] = tick
    env._upstream_started[active] = True
    state['parked'][flat] = False
    state['terminal_entered'][flat] = False
    state['arrived'][flat] = False
    return state


def flat_neutral_boundary(env):
    flat = flat_mask(env)
    if not bool(flat.any()):
        return flat
    local = tensor(env.scene['robot'].data.root_pos_w) - tensor(env.scene.env_origins)
    limit = params(env)['outer_half'] - params(env)['flat_boundary_margin']
    return flat & (local[:, :2].abs() >= limit).any(-1)


def time_out(env):
    update_course(env)
    expired = tensor(env.episode_length_buf) >= env.max_episode_length
    return (expired | flat_neutral_boundary(env)) & ~env.beamdojo_course_failed



def before_reset(env, ids):
    """Upgrade on 3 arrivals; downgrade after 2 failures to reach the ring.

    An arrival remains a curriculum win when the robot subsequently times out or
    falls while braking. Those parking failures remain visible in separate stage
    counters, and the physical episode still terminates as a failure.
    """
    course = ensure_state(env)
    ids = mixed._ids(env, ids).unique()
    counters = env._square_mixed_state
    tick = int(env.common_step_counter)
    ids = ids[counters['last_reset_tick'][ids] != tick]
    if ids.numel() == 0:
        return {}
    levels = env.beamdojo_levels[ids].clone()
    bins = env.square_family[ids] * 4 + env.square_route[ids]
    experienced = env._upstream_started[ids]
    expired = tensor(env.episode_length_buf)[ids] >= env.max_episode_length
    failed = env.beamdojo_course_failed[ids]
    arrived = course['arrived'][ids]
    eligible = experienced & (arrived | failed | expired) & ~flat_mask(env)[ids]
    skip = getattr(env, '_square_skip_curriculum_once', None)
    if skip is not None:
        eligible &= ~skip[ids]
        skip[ids] = False
    wins = eligible & arrived
    losses = eligible & ~arrived
    win_streak = torch.where(wins, env.square_win_streak[ids] + 1, 0)
    fail_streak = torch.where(wins, 0, env.square_fail_streak[ids] + 1)
    win_streak = torch.where(eligible, win_streak, env.square_win_streak[ids])
    fail_streak = torch.where(eligible, fail_streak, env.square_fail_streak[ids])
    promote = wins & (win_streak >= 3) & (levels < 10)
    demote = losses & (fail_streak >= 2) & (levels > 0)
    env.beamdojo_levels[ids] = (levels + promote.long() - demote.long()).clamp(0, 10)
    env.square_win_streak[ids] = torch.where(promote | demote, 0, win_streak.clamp_max(3))
    env.square_fail_streak[ids] = torch.where(promote | demote, 0, fail_streak.clamp_max(2))
    values = dict(episodes=eligible, successes=wins, failures=losses & failed,
                  timeouts=losses & expired & ~failed, promotions=promote, demotions=demote)
    indices = bins * 11 + levels
    for name, value in values.items():
        counters[name].view(-1).scatter_add_(0, indices, value.long())
    stages = dict(arrivals=wins, parkings=eligible & course['parked'][ids],
                  stable_parks=eligible & env.beamdojo_qualified_success[ids],
                  arrived_then_failed=wins & failed, arrived_then_timeout=wins & expired & ~failed,
                  timeout_episodes=eligible & expired & ~failed)
    for name, value in stages.items():
        env._square_extended_counts[name].view(-1).scatter_add_(0, indices, value.long())
    timeout = (eligible & expired & ~failed).float()
    diagnostics = dict(
        timeout_progress_s_sum=timeout * course['last_progress_s'][ids],
        timeout_planar_speed_sum=timeout * course['last_planar_speed'][ids],
        timeout_contact_count_sum=timeout * course['last_contact_count'][ids],
        timeout_full_support_count=timeout * course['last_full_support'][ids],
        maximum_stand_seconds_sum=eligible.float() * course['maximum_stand_time'][ids])
    for name, value in diagnostics.items():
        env._square_extended_counts[name].view(-1).scatter_add_(0, indices, value)
    counters['last_reset_tick'][ids] = tick
    choose = getattr(env, 'square_choose_variants', None)
    if choose is not None:
        choose(ids)
    mixed.refresh_angles(env)
    env.square_refresh_geometry(env, ids)
    base.reset_course(env, ids)
    stone = ~flat_mask(env)
    return {
        'Metrics/course_success_rate': wins.float().sum() / eligible.sum().clamp_min(1),
        'Metrics/course_arrival_rate': wins.float().sum() / eligible.sum().clamp_min(1),
        'Curriculum/square_promotions': promote.float().sum(),
        'Curriculum/square_demotions': demote.float().sum(),
        'Curriculum/square_mean_level': (env.beamdojo_levels.float() * stone).sum() / stone.sum().clamp_min(1),
    }


def course_metrics(env):
    ensure_state(env)
    metrics = _ORIGINAL_METRICS(env)
    counts = env._square_extended_counts
    episodes = env._square_mixed_state['episodes'].sum().clamp_min(1)
    timeouts = counts['timeout_episodes'].sum().clamp_min(1)
    for stage in ('arrivals', 'parkings', 'stable_parks'):
        metrics['Extended/' + stage + '_rate'] = counts[stage].sum().float() / episodes
    for stage in ('arrived_then_failed', 'arrived_then_timeout'):
        metrics['Extended/' + stage + '_rate'] = counts[stage].sum().float() / counts['arrivals'].sum().clamp_min(1)
    diagnostic_names = dict(timeout_progress_s_sum='timeout_progress_m_mean',
                            timeout_planar_speed_sum='timeout_planar_speed_mps_mean',
                            timeout_contact_count_sum='timeout_contact_feet_mean',
                            timeout_full_support_count='timeout_full_support_fraction')
    for key, label in diagnostic_names.items():
        metrics['Extended/' + label] = counts[key].sum() / timeouts
    metrics['Extended/maximum_stand_seconds_mean'] = counts['maximum_stand_seconds_sum'].sum() / episodes
    return metrics


def snapshot(env):
    saved = _ORIGINAL_SNAPSHOT(env)
    saved['extended_counters'] = {name: value.detach().cpu().tolist()
                                  for name, value in env._square_extended_counts.items()}
    saved['extended_curriculum'] = 'arrival_3wins_2losses_v1'
    return saved


def restore(env, saved):
    """Restore only same-geometry curriculum; caller validates its contract first."""
    ensure_state(env)
    if saved.get('version') != 1 or saved.get('num_envs') != env.num_envs:
        raise ValueError('Extended curriculum checkpoint version/environment count mismatch')
    specs = getattr(env.cfg, 'square_mixed_column_specs', [(0, 0, 3)])
    maximum_variant = max(int(spec[2]) for spec in specs)
    maximum_variant = int(getattr(env.cfg, 'beamdojo_square_extended_params', {}).get(
        'variant_max', maximum_variant))
    bounds = dict(beamdojo_levels=10, square_family=1, square_route=3,
                  square_variant=maximum_variant, square_win_streak=3, square_fail_streak=2)
    staged = {}
    for name, maximum in bounds.items():
        value = torch.as_tensor(saved['assignments'][name], device=env.device, dtype=torch.long)
        if value.shape != (env.num_envs,) or bool(((value < 0) | (value > maximum)).any()):
            raise ValueError(f'Invalid extended curriculum assignment: {name}')
        staged[name] = value
    staged_counts = {}
    for name, target in env._square_mixed_state.items():
        if name == 'last_reset_tick':
            continue
        value = torch.as_tensor(saved['counters'][name], device=env.device, dtype=target.dtype)
        if value.shape != (8, 11) or bool((value < 0).any()):
            raise ValueError(f'Invalid mixed curriculum counter: {name}')
        staged_counts[name] = value
    extended = {}
    for name, target in env._square_extended_counts.items():
        value = torch.as_tensor(saved.get('extended_counters', {}).get(name, torch.zeros_like(target)),
                                device=env.device, dtype=target.dtype)
        if value.shape != (8, 11) or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError(f'Invalid extended curriculum counter: {name}')
        extended[name] = value
    for name, value in staged.items():
        getattr(env, name).copy_(value)
    for name, value in staged_counts.items():
        env._square_mixed_state[name].copy_(value)
    for name, value in extended.items():
        env._square_extended_counts[name].copy_(value)
    env._square_mixed_state['last_reset_tick'].fill_(-1)
    env._square_skip_curriculum_once = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    mixed.refresh_angles(env)
    env.square_refresh_geometry(env, None)
    base.reset_course(env, None)


def install(runtime=mixed):
    """Bind extended course hooks; caller must rebind already-created manager refs.

    Old command and curriculum classes resolve these module globals at call time.
    Base reset zeroes all state tensors, including the three new terminal fields.
    """
    for name in ('ensure_state', 'goal_start', '_geometry', 'parking_command', 'update_course', 'time_out',
                 'before_reset', 'course_metrics', 'snapshot', 'restore'):
        setattr(runtime, name, globals()[name])
    return runtime
