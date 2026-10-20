"""Opt-in position-action bounds derived from the live articulation limits."""
import torch


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def normalized_bounds(term):
    """Return bounds in this term's resolved action order, including offsets.

    Sim6 JointPositionAction uses _joint_ids, _scale and _offset in precisely
    this order. Offsets may be per-environment default joint positions.
    """
    if not any(cls.__name__ == 'JointPositionAction' for cls in type(term).__mro__):
        raise TypeError('Action bounds require absolute JointPositionAction terms')
    limits = _tensor(term._asset.data.soft_joint_pos_limits)[:, term._joint_ids]
    scale = torch.as_tensor(term._scale, device=limits.device, dtype=limits.dtype)
    offset = torch.as_tensor(term._offset, device=limits.device, dtype=limits.dtype)
    if not torch.isfinite(scale).all() or (scale == 0).any():
        raise ValueError('Joint action scale must be finite and nonzero')
    low, high = limits[..., 0], limits[..., 1]
    # Native term clipping is applied in position units after the affine map.
    # Intersect it too, so recorded actions describe the executed targets.
    if getattr(term.cfg, 'clip', None) is not None:
        clip = _tensor(term._clip)
        low = torch.maximum(low, clip[..., 0])
        high = torch.minimum(high, clip[..., 1])
    if (not torch.isfinite(limits).all() or not torch.isfinite(offset).all()
            or (low > high).any()):
        raise ValueError('Invalid or disjoint joint position limits/offsets')
    a, b = (low - offset) / scale, (high - offset) / scale
    return torch.minimum(a, b), torch.maximum(a, b)


def apply(env, action):
    """Return executed normalized actions; never mutate PPO's raw action tensor."""
    if not getattr(env.cfg, 'beamdojo_bound_joint_actions', False):
        return action
    pieces, summary = [], []
    start = 0
    first = not hasattr(env, 'beamdojo_action_bounds_summary')
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        low, high = normalized_bounds(term)
        end = start + term.action_dim
        pieces.append(torch.clamp(action[:, start:end], min=low, max=high))
        if first:
            summary.append({'term': name, 'joint_names': list(term._joint_names),
                            'normalized_lower_min': low.amin(dim=0).cpu().tolist(),
                            'normalized_lower_max': low.amax(dim=0).cpu().tolist(),
                            'normalized_upper_min': high.amin(dim=0).cpu().tolist(),
                            'normalized_upper_max': high.amax(dim=0).cpu().tolist()})
        start = end
    if start != action.shape[-1]:
        raise ValueError('Action dimension differs from resolved position action terms')
    executed = torch.cat(pieces, dim=-1)
    if first:
        env.beamdojo_action_bounds_summary = {
            'source': 'live soft_joint_pos_limits and resolved action affine mapping',
            'action_dimension': start, 'terms': summary}
    env._beamdojo_action_bounds_log = {
        'ActionBounds/raw_abs_max': action.detach().abs().max(),
        'ActionBounds/executed_abs_max': executed.detach().abs().max(),
        'ActionBounds/clip_fraction': (executed != action).float().mean(),
    }
    return executed


def diagnostics(env):
    if not getattr(env.cfg, 'beamdojo_bound_joint_actions', False):
        return {}
    histories = getattr(getattr(env, 'observation_manager', None),
                        '_group_obs_term_history_buffer', {})
    history = histories.get('policy', {}).get('last_action')
    actual = env.action_manager.action if history is None else history.buffer
    return {**getattr(env, '_beamdojo_action_bounds_log', {}),
            'ActionBounds/history_actual_abs_max': actual.detach().abs().max()}
