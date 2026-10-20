"""Optional causal per-foot contact-event cost; does not replace foothold reward.

Call update before rewards even in a weight-zero control arm. Call reset_state
AFTER native reset. No actor inputs or future failure labels enter this module.
Rates are positive costs: reward weight should be negative. Contact interruptions
end events, so this term is not a substitute for instantaneous unsafe-point cost.
"""
import math
import torch


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def ensure_state(env):
    if not hasattr(env, '_beamdojo_contact_event_risk'):
        shape = (env.num_envs, 2)
        zeros = lambda: torch.zeros(shape, device=env.device)
        env._beamdojo_contact_event_risk = {
            'exposure_seconds': zeros(), 'contact_seconds': zeros(),
            'unsafe_fraction': zeros(), 'severity': zeros(), 'cost_per_foot': zeros(),
            'contact': torch.zeros(shape, device=env.device, dtype=torch.bool),
            'event_count': zeros(), 'ended_event_count': zeros(),
            'peak_exposure_seconds': zeros(), 'cost_integral': zeros(),
            'tick': torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)}
    return env._beamdojo_contact_event_risk


def sample_contact(env, force_threshold=5.):
    """Same 35 points and conservative min3 XY map as strict foothold reward.

    This is a geometric risk proxy, not measured pressure/true support area.
    Forces are current net world-Z force, not contact sensor history maxima.
    """
    from .geometry import g1_sole_points, transform_sole_points
    from .upstream_terrain import surface_height
    robot = env.scene['robot']
    sensor = env.scene.sensors['contact_forces']
    if not hasattr(env, '_beamdojo_contact_risk_query'):
        names = ['left_ankle_roll_link', 'right_ankle_roll_link']
        env._beamdojo_contact_risk_query = (
            robot.find_bodies(names, preserve_order=True)[0],
            sensor.find_bodies(names, preserve_order=True)[0],
            g1_sole_points(device=env.device))
    feet, contacts, points = env._beamdojo_contact_risk_query
    data = robot.data
    soles = transform_sole_points(_tensor(data.body_pos_w)[:, feet],
        _tensor(data.body_quat_w)[:, feet][..., [3, 0, 1, 2]], points)
    _, safe = surface_height(env, soles, exact=False)
    unsafe = (~safe).float().mean(-1)
    contact = _tensor(sensor.data.net_forces_w)[:, contacts, 2] > force_threshold
    return unsafe, contact


def update(env, unsafe_start=.3, unsafe_full=.8, duration_scale=.1, force_threshold=5.):
    """Update once/control step; cost = severity * (1-exp(-exposure/tau)).

    severity is smoothstep of unsafe fraction between start/full. Exposure is
    integral(severity dt) within the current uninterrupted measured contact.
    Each foot costs [0,1], summed rate [0,2]; no command/motion gating. Safe
    stance costs zero, unsafe stationary stance keeps accumulating cost.
    Parameters are exploratory hyperparameters, not validated safety limits.
    """
    parameters = (unsafe_start, unsafe_full, duration_scale, force_threshold)
    if (not all(math.isfinite(x) for x in parameters)
            or not 0 <= unsafe_start < unsafe_full <= 1
            or duration_scale <= 0 or force_threshold < 0
            or not math.isfinite(env.step_dt) or env.step_dt <= 0):
        raise ValueError('Invalid contact-event risk parameters/control dt')
    state = ensure_state(env)
    # Do not resample simulator state on repeated reward/telemetry calls.
    tick = env.common_step_counter
    fresh = state['tick'] != tick
    sample_key = (tick, force_threshold)
    if getattr(env, '_beamdojo_contact_risk_sample_key', None) != sample_key:
        env._beamdojo_contact_risk_sample = sample_contact(env, force_threshold)
        env._beamdojo_contact_risk_sample_key = sample_key
    unsafe, contact = env._beamdojo_contact_risk_sample
    x = ((unsafe-unsafe_start)/(unsafe_full-unsafe_start)).clamp(0., 1.)
    severity = x.square() * (3.-2.*x) * contact
    dt = env.step_dt
    mask = fresh[:, None]
    exposure = torch.where(contact, state['exposure_seconds']+severity*dt, 0.)
    duration = torch.where(contact, state['contact_seconds']+dt, 0.)
    cost = severity * (-torch.expm1(-exposure/duration_scale))
    starts = contact & ~state['contact']
    ends = ~contact & state['contact']
    values = {'exposure_seconds': exposure, 'contact_seconds': duration,
        'unsafe_fraction': unsafe, 'severity': severity, 'cost_per_foot': cost,
        'contact': contact, 'event_count': state['event_count']+starts,
        'ended_event_count': state['ended_event_count']+ends,
        'peak_exposure_seconds': torch.maximum(state['peak_exposure_seconds'], exposure),
        'cost_integral': state['cost_integral']+cost*dt}
    for name, value in values.items():
        state[name].copy_(torch.where(mask, value, state[name]))
    state['tick'].fill_(tick)
    return state


def event_contact_risk(env, unsafe_start=.3, unsafe_full=.8, duration_scale=.1, force_threshold=5.):
    """Positive bounded [0,2] rate; e.g. weight=-.5 -> max -1 reward/sec."""
    return update(env, unsafe_start, unsafe_full, duration_scale, force_threshold)['cost_per_foot'].sum(-1)


def reset_state(env, env_ids=None):
    state = ensure_state(env)
    ids = slice(None) if env_ids is None else env_ids
    for name, value in state.items():
        value[ids] = env.common_step_counter if name == 'tick' else 0


def diagnostics(env):
    """Read-only scalar tensors; caller controls pre-reset sampling timing."""
    state = ensure_state(env)
    return {'ContactRisk/cost_rate': state['cost_per_foot'].sum(-1).mean(),
        'ContactRisk/severity': state['severity'].mean(),
        'ContactRisk/contact_fraction': state['contact'].float().mean(),
        'ContactRisk/exposure_seconds_max': state['exposure_seconds'].amax(),
        'ContactRisk/episode_peak_exposure_mean': state['peak_exposure_seconds'].amax(-1).mean(),
        'ContactRisk/events_per_env': state['event_count'].sum(-1).mean(),
        'ContactRisk/ended_events_per_env': state['ended_event_count'].sum(-1).mean(),
        'ContactRisk/episode_cost_integral': state['cost_integral'].sum(-1).mean()}
