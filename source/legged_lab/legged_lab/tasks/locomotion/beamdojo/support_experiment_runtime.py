"""Opt-in fixed-level sampling shared by both support-risk experiment arms.

No RNG draws, command changes, geometry refreshes or adaptive promotion. The
caller refreshes geometry after assignments and invokes before_reset before the
native reset clears episode lengths. Evaluation must leave the flag disabled.
"""
from copy import deepcopy
import torch

LEVELS = (0, 7, 8, 9, 10)
LEVEL_COUNT = 11


def enabled(env):
    return bool(getattr(env.cfg, 'beamdojo_support_fixed_levels', False))


def _ids(env, ids):
    all_ids = torch.arange(env.num_envs, device=env.device)
    return all_ids if ids is None else all_ids[ids] if isinstance(ids, slice) else torch.as_tensor(ids, device=env.device, dtype=torch.long)


def _state(env):
    if not hasattr(env, '_beamdojo_support_experiment_state'):
        env._beamdojo_support_experiment_state = {
            'level_episode_count': torch.zeros(LEVEL_COUNT, dtype=torch.long, device=env.device),
            'level_env_steps': torch.zeros(LEVEL_COUNT, dtype=torch.long, device=env.device),
            'tick': None,
            'last_end_tick': torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)}
    return env._beamdojo_support_experiment_state


def assign_levels(env, ids=None):
    if not enabled(env):
        return
    selected = _ids(env, ids)
    mapping = torch.tensor(LEVELS, device=env.device, dtype=torch.long)
    env.beamdojo_levels[selected] = mapping[selected % len(LEVELS)]
    env.beamdojo_success_streak[selected] = 0


def before_reset(env, ids):
    if not enabled(env):
        return
    state = _state(env)
    selected = _ids(env, ids)
    tick = env.common_step_counter
    ended = ((env.episode_length_buf[selected] > 0)
             & (state['last_end_tick'][selected] != tick))
    # Count actual ending level before assignment; no speculative exposure.
    state['level_episode_count'].scatter_add_(0, env.beamdojo_levels[selected], ended.long())
    state['last_end_tick'][selected] = torch.where(ended,
        torch.full_like(selected, tick), state['last_end_tick'][selected])
    assign_levels(env, selected)


def update(env):
    if not enabled(env):
        return
    state = _state(env)
    tick = env.common_step_counter
    if state['tick'] == tick:
        return
    # Native reset observation/reward inspection at zero length is not a
    # completed physics transition. On real steps every active env contributes.
    active = (env.episode_length_buf > 0).long()
    state['level_env_steps'].scatter_add_(0, env.beamdojo_levels, active)
    state['tick'] = tick


def diagnostics(env):
    if not enabled(env):
        return {}
    state = _state(env)
    histogram = torch.bincount(env.beamdojo_levels, minlength=LEVEL_COUNT)
    result = {}
    for level in range(LEVEL_COUNT):
        result[f'SupportExperiment/level_{level}_fraction'] = histogram[level].float()/env.num_envs
        result[f'SupportExperiment/level_{level}_episodes'] = state['level_episode_count'][level].float()
        result[f'SupportExperiment/level_{level}_env_steps'] = state['level_env_steps'][level].float()
    return result


def snapshot(env):
    if not enabled(env):
        return None
    state = _state(env)
    counts = [0] * LEVEL_COUNT
    for index in range(env.num_envs):
        counts[LEVELS[index % len(LEVELS)]] += 1
    return {'version': 1, 'num_envs': env.num_envs,
        'assignment': 'levels[env_id % 5]', 'fixed_levels': list(LEVELS),
        'fixed_level_counts': counts,
        'level_episode_count': state['level_episode_count'].cpu().tolist(),
        'level_env_steps': state['level_env_steps'].cpu().tolist(),
        'experiment_spec': deepcopy(getattr(env.cfg, 'beamdojo_support_experiment', None))}


def restore(env, saved=None):
    """Initialize or inherit cumulative coverage before an ordinary full reset.

    Does not assign levels or copy commands/RNG. Absolute simulator tick guards
    are reset because checkpoint step counters belong to the old environment.
    """
    if not enabled(env):
        return
    values = {name: [0]*LEVEL_COUNT for name in ('level_episode_count','level_env_steps')}
    if saved is not None:
        expected = [0]*LEVEL_COUNT
        for index in range(env.num_envs):
            expected[LEVELS[index % 5]] += 1
        if (saved.get('version') != 1 or saved.get('num_envs') != env.num_envs
                or saved.get('fixed_levels') != list(LEVELS)
                or saved.get('fixed_level_counts') != expected):
            raise ValueError('Incompatible fixed-level experiment snapshot')
        for name in values:
            data = saved.get(name)
            if (not isinstance(data, list) or len(data) != LEVEL_COUNT
                    or any(type(v) is not int or not 0 <= v < 2**63 for v in data)):
                raise ValueError('Invalid support experiment counter: '+name)
            values[name] = data
    state = _state(env)
    for name, value in values.items():
        state[name].copy_(torch.tensor(value, device=env.device, dtype=torch.long))
    state['tick'] = None
    state['last_end_tick'].fill_(-1)
