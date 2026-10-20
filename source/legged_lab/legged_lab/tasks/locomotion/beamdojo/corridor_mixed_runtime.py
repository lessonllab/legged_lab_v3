"""Fixed environment quotas; short failed episodes never resample difficulty."""
import torch

WEIGHTS = (12, 9, 9, 9, 9, 3, 3, 3, 1, 1, 1)
VERSION = 1
COUNTERS = ('env_steps', 'departed_platform_steps', 'supported_contact_steps',
            'outside_platform_supported_steps',
            'episodes', 'qualified_episodes', 'failed_episodes')


def spec():
    return {'version': VERSION, 'weights': list(WEIGHTS),
            'assignment': 'largest remainder quotas stratified by terrain type; weighted deficit interleaving within each type',
            'episode_resampling': False}


def assignment_counts(num_envs):
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError('num_envs must be a positive integer')
    total = sum(WEIGHTS)
    counts = [num_envs * weight // total for weight in WEIGHTS]
    order = sorted(range(len(WEIGHTS)), key=lambda level: (-(num_envs * WEIGHTS[level] % total), level))
    for level in order[:num_envs - sum(counts)]:
        counts[level] += 1
    return counts


def enabled(env):
    return bool(getattr(env.cfg, 'beamdojo_corridor_mixed_levels', False))


def _state(env):
    if hasattr(env, '_beamdojo_corridor_mixed'):
        return env._beamdojo_corridor_mixed
    counts = assignment_counts(env.num_envs)
    terrain = getattr(env.scene, 'terrain', None)
    types = getattr(terrain, 'terrain_types', torch.zeros(env.num_envs, dtype=torch.long))
    types = types if isinstance(types, torch.Tensor) else types.torch
    types_list = types.cpu().tolist()
    groups = {kind: [] for kind in sorted(set(types_list))}
    for index, kind in enumerate(types_list):
        groups[kind].append(index)
    # Allocate global quotas proportionally within each physical terrain type,
    # then interleave levels inside that type. Global interleaving alone can
    # alias a level's periodicity with the number of terrain columns.
    remaining = counts.copy()
    remaining_envs = env.num_envs
    assignment = [0] * env.num_envs
    for group in groups.values():
        n = len(group)
        quota = [count * n // remaining_envs for count in remaining]
        priority = sorted(range(len(WEIGHTS)),
            key=lambda i: (-(remaining[i] * n % remaining_envs), i))
        for level in priority[:n - sum(quota)]:
            quota[level] += 1
        used = [0] * len(WEIGHTS)
        for step, index in enumerate(group):
            level = max((i for i in range(len(WEIGHTS)) if used[i] < quota[i]),
                        key=lambda i: ((step + 1) * quota[i] - used[i] * n, -i))
            assignment[index] = level
            used[level] += 1
        remaining = [a - b for a, b in zip(remaining, quota)]
        remaining_envs -= n
    state = {name: torch.zeros(len(WEIGHTS), dtype=torch.long, device=env.device) for name in COUNTERS}
    state.update(assignment=torch.tensor(assignment, dtype=torch.long, device=env.device),
                 terrain_types=types_list, tick=None,
                 last_end_tick=torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
                 last_contact_seconds=torch.zeros(env.num_envs, device=env.device))
    env._beamdojo_corridor_mixed = state
    return state


def assign_levels(env, ids=None):
    if not enabled(env):
        return
    state = _state(env)
    selected = slice(None) if ids is None else ids
    env.beamdojo_levels[selected] = state['assignment'][selected]
    env.beamdojo_success_streak[selected] = 0


def update(env):
    if not enabled(env):
        return
    state = _state(env)
    if state['tick'] == env.common_step_counter:
        return
    active = env.episode_length_buf > 0
    levels = env.beamdojo_levels
    state['env_steps'].scatter_add_(0, levels, active.long())
    corridor = getattr(env, '_beamdojo_corridor_state', None)
    if corridor is not None:
        state['departed_platform_steps'].scatter_add_(0, levels,
            (active & corridor['departed_platform']).long())
    contact = getattr(env, 'beamdojo_course_valid_contact_seconds', None)
    if contact is not None:
        supported = active & (contact > state['last_contact_seconds'])
        state['supported_contact_steps'].scatter_add_(0, levels, supported.long())
        if corridor is not None and 'geometry' in corridor:
            position = env.scene['robot'].data.root_pos_w
            origins = env.scene.env_origins
            position = position if isinstance(position, torch.Tensor) else position.torch
            origins = origins if isinstance(origins, torch.Tensor) else origins.torch
            local = position[:, :2] - origins[:, :2]
            geometry = corridor['geometry'][levels]
            outside = ((local[:, 0] < geometry[:, 3]) | (local[:, 0] > geometry[:, 4])
                       | (local[:, 1] < geometry[:, 5]) | (local[:, 1] > geometry[:, 6]))
            state['outside_platform_supported_steps'].scatter_add_(0, levels,
                (supported & outside).long())
        state['last_contact_seconds'].copy_(contact)
    state['tick'] = env.common_step_counter


def before_reset(env, ids):
    if not enabled(env):
        return
    state = _state(env)
    selected = torch.arange(env.num_envs, device=env.device)[ids]
    ended = ((env.episode_length_buf[selected] > 0)
             & (state['last_end_tick'][selected] != env.common_step_counter))
    levels = env.beamdojo_levels[selected]
    state['episodes'].scatter_add_(0, levels, ended.long())
    state['qualified_episodes'].scatter_add_(0, levels,
        (ended & env.beamdojo_qualified_success[selected]).long())
    failed = getattr(env, 'beamdojo_course_failed', torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    manager = getattr(env, 'termination_manager', None)
    if manager is not None and hasattr(manager, 'terminated'):
        terminated = manager.terminated
        failed = failed | (terminated if isinstance(terminated, torch.Tensor) else terminated.torch)
    state['failed_episodes'].scatter_add_(0, levels, (ended & failed[selected]).long())
    state['last_end_tick'][selected] = torch.where(ended,
        torch.full_like(selected, env.common_step_counter), state['last_end_tick'][selected])
    state['last_contact_seconds'][selected] = 0
    assign_levels(env, ids)


def diagnostics(env):
    if not enabled(env):
        return {}
    state = _state(env)
    histogram = torch.bincount(env.beamdojo_levels, minlength=len(WEIGHTS)).cpu().tolist()
    values = torch.stack([state[name] for name in COUNTERS]).cpu().tolist()
    result = {}
    for level in range(len(WEIGHTS)):
        result[f'CorridorCoverage/level_{level}_fraction'] = histogram[level] / env.num_envs
        for index, name in enumerate(COUNTERS):
            result[f'CorridorCoverage/level_{level}_{name}'] = values[index][level]
    return result


def snapshot(env):
    if not enabled(env):
        return None
    state = _state(env)
    return dict(spec=spec(), num_envs=env.num_envs, counts=assignment_counts(env.num_envs),
                assignment=state['assignment'].cpu().tolist(), terrain_types=state['terrain_types'],
                counters={name: state[name].cpu().tolist() for name in COUNTERS})


def restore(env, saved):
    """Restore cumulative exposure before a normal reset; never restore poses."""
    if not enabled(env):
        return
    state = _state(env)
    if saved is not None:
        expected = snapshot(env)
        for key in ('spec', 'num_envs', 'counts', 'assignment', 'terrain_types'):
            if saved.get(key) != expected[key]:
                raise ValueError('Incompatible mixed corridor snapshot: ' + key)
        counters = saved.get('counters', {})
        for name in COUNTERS:
            values = counters.get(name)
            if (not isinstance(values, list) or len(values) != len(WEIGHTS)
                    or any(type(v) is not int or not 0 <= v < 2**63 for v in values)):
                raise ValueError('Invalid corridor coverage counter: ' + name)
        for name in COUNTERS:
            state[name].copy_(torch.tensor(counters[name], dtype=torch.long, device=env.device))
    state['tick'] = None
    state['last_end_tick'].fill_(-1)
    state['last_contact_seconds'].zero_()
    assign_levels(env)
