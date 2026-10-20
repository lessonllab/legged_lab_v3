"""Finite terminal transition shared by live and archived mixed tasks."""
import math
import torch

REVISION = 'full_support_then_zero_v2'


def parking_command_impl(env, cruise, root_s, full_ring, ensure_state, goal_start):
    state = ensure_state(env)
    # Keep advancing until full support and the existing stable-goal margin match.
    # Never approach the parking threshold asymptotically under tiny commands.
    stop_s = goal_start(env) + .20
    state['parked'] |= full_ring & (root_s >= stop_s)
    target = goal_start(env) + .25
    speed = (cruise.clamp(.5, 1.) * ((target-root_s)/.20).clamp(0, 1)).clamp_min(.25)
    speed = torch.where(state['parked'], 0., speed)
    return torch.stack((speed, torch.zeros_like(speed), torch.zeros_like(speed)), -1)


def install(runtime):
    """Override only terminal commands; archived geometry and weights stay intact."""
    def corrected(env, cruise, root_s, full_ring):
        return parking_command_impl(env,cruise,root_s,full_ring,runtime.ensure_state,runtime.goal_start)
    runtime.parking_command = corrected
    runtime.update_course = lambda env, measurements=None: update_course_impl(env,measurements,runtime)


def update_course_impl(env, measurements, runtime):
    ensure_state=runtime.ensure_state;read_measurements=runtime.read_measurements
    _geometry=runtime._geometry;tensor=runtime.tensor;base=runtime.base;goal_start=runtime.goal_start
    state = ensure_state(env)
    tick = int(env.common_step_counter)
    active = state['last_tick'] != tick
    if not bool(active.any()):
        return state
    m = read_measurements(env) if measurements is None else measurements
    root, corners, full = _geometry(env, m)
    contact = m['contact_fz'] > 5
    finite = torch.isfinite(root).all(-1) & torch.isfinite(corners).all((-1, -2, -3))
    for key in ('lin', 'ang', 'gravity', 'contact_fz'):
        finite &= torch.isfinite(m[key]).all(-1)
    tilt = torch.acos((-m['gravity'][:, 2]).clamp(-1, 1))
    terms = getattr(env.cfg, 'terminations', None)
    minimum_height = getattr(getattr(terms, 'base_height', None), 'params', {}).get('minimum_height', .2)
    maximum_tilt = getattr(getattr(terms, 'bad_orientation', None), 'params', {}).get('limit_angle', .8)
    central = torch.where(env.square_family == 0, .7, .7 * math.sqrt(2))
    departed = state['departed'] | (root[:, 0] > central)
    # Physical map containment is checked BEFORE route rotation.
    boundary = (m['root'][:, :2].abs() >= 3.9).any(-1)
    boundary |= departed & (root[:, 1].abs() > .42)
    failure = (root[:, 2] < minimum_height) | (tilt > maximum_tilt) | ~finite | boundary
    failure |= env.beamdojo_course_failed
    expired = tensor(env.episode_length_buf) >= env.max_episode_length
    command = tensor(env.command_manager.get_command('base_velocity'))
    zero = torch.isfinite(command).all(-1) & (command[:, :2].norm(dim=-1) < 1e-4) & (command[:, 2].abs() < 1e-4)
    stable = full & contact.all(-1) & (m['lin'].norm(dim=-1) < .15) & (m['ang'].norm(dim=-1) < .3) & (tilt < .35)
    stable &= state['parked'] & zero & ~failure & ~expired
    duration = torch.where(stable, env.beamdojo_course_stand_time + env.step_dt, 0.)
    success = (duration >= base.params(env)['stand_seconds'] - 1e-6) & ~failure & ~expired
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
    state['departed'][active] = departed[active]
    state['last_xy'][active] = root[active, :2]
    state['initialized'][active] = True
    state['last_tick'][active] = tick
    env._upstream_started[active] = True
    return state
