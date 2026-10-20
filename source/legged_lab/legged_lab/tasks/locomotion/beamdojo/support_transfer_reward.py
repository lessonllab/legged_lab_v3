"""Training-only, confirmed-contact support-transfer experiment.

The score is a constant-height, constant-pressure-center LIPM proxy, not a
whole-body feasibility certificate. Geometrically usable sole samples approximate an
available pressure region; the simulator does not measure their pressure.
Actor inputs, controllers, and every original reward remain unchanged.
"""
import torch


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def rotation_xyzw(quat):
    """Body COM-frame to world rotation; Isaac Sim 6 quaternions are xyzw."""
    x, y, z, w = (quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*quat.shape[:-1], 3, 3)


def centroidal_state(masses, positions, velocities, angular_velocities, inertias, com_quaternions):
    """Mass-weighted COM and total world angular momentum about COM.

    Inertias are PhysX body-COM-frame tensors in user body order, not rotated
    link-frame tensors. Include both body spin and orbital angular momentum.
    """
    mass = masses.sum(-1)
    com = (masses[..., None] * positions).sum(-2) / mass[:, None]
    velocity = (masses[..., None] * velocities).sum(-2) / mass[:, None]
    rotation = rotation_xyzw(com_quaternions)
    omega_local = (rotation.transpose(-1, -2) @ angular_velocities[..., None]).squeeze(-1)
    local_spin = (inertias @ omega_local[..., None]).squeeze(-1)
    spin = (rotation @ local_spin[..., None]).squeeze(-1)
    orbital = torch.cross(positions-com[:, None], masses[..., None]*(velocities-velocity[:, None]), dim=-1)
    return com, velocity, (spin+orbital).sum(-2), mass


def contact_events(state, contact, *, release_steps=3, load_steps=2):
    """One event after confirmed release then confirmed re-load, per foot.

    Sustained stance and 1–2 frame sensor dropouts cannot repeatedly score.
    State is reset explicitly after the native simulator resets an episode.
    """
    state['off'].copy_(torch.where(contact, 0, state['off']+1))
    state['armed'] |= state['off'] >= release_steps
    state['on'].copy_(torch.where(contact, state['on']+1, 0))
    event = state['armed'] & (state['on'] >= load_steps)
    state['armed'] &= ~event
    return event


def reset_state(env, env_ids):
    if not hasattr(env, '_support_transfer_state'):
        return
    state = env._support_transfer_state
    # Preserve cumulative diagnostics, but never transfer event eligibility.
    state['episodes_without_event'] += ((state['episode_events'][env_ids] == 0) & state['episode_seen'][env_ids]).sum()
    state['episodes_observed'] += state['episode_seen'][env_ids].sum()
    for name in ('off', 'on', 'armed', 'episode_events', 'episode_seen'):
        state[name][env_ids] = 0
    # A manual reset need not advance common_step_counter. Preserve other
    # environments' final-transition cache, clear only the reset episodes.
    if hasattr(env, '_support_transfer_rate'):
        env._support_transfer_rate[env_ids] = 0
    for value in getattr(env, '_support_transfer_last', {}).values():
        value[env_ids] = 0


def _state(env):
    if not hasattr(env, '_support_transfer_state'):
        shape = (env.num_envs, 2)
        state = {name: torch.zeros(shape, device=env.device, dtype=torch.long) for name in ('off', 'on')}
        state['armed'] = torch.zeros(shape, device=env.device, dtype=torch.bool)
        state['episode_events'] = torch.zeros(env.num_envs, device=env.device)
        state['episode_seen'] = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for name in ('events', 'cost_sum', 'absolute_sum', 'held_sum', 'invalid', 'point_sum',
                     'old_available', 'nonzero_increment', 'held_high',
                     'momentum_sum', 'speed_sum', 'episodes_without_event', 'episodes_observed'):
            state[name] = torch.zeros((), device=env.device)
        env._support_transfer_state = state
    return env._support_transfer_state


@torch.no_grad()
def support_transfer_cost(env):
    from .geometry import g1_sole_points, transform_sole_points
    from .upstream_terrain import surface_height
    from . import corridor_runtime
    from .support_transfer_lipm import support_transfer_lipm_cost as solve
    if getattr(env.cfg, 'beamdojo_support_transfer_arm', None) != 'transfer':
        raise ValueError('Support-transfer reward requires explicit experiment contract')
    if getattr(env, '_support_transfer_tick', None) == env.common_step_counter:
        return env._support_transfer_rate
    robot = env.scene['robot']; sensor = env.scene.sensors['contact_forces']; data = robot.data
    if not hasattr(env, '_support_transfer_ids'):
        names = ['left_ankle_roll_link', 'right_ankle_roll_link']
        env._support_transfer_ids = (robot.find_bodies(names, preserve_order=True)[0],
                                    sensor.find_bodies(names, preserve_order=True)[0],
                                    g1_sole_points(device=env.device))
        # User-order quantities after startup randomization. This frozen profile
        # changes neither mass nor inertia during episodes.
        env._support_transfer_masses = _tensor(data.body_mass).clone().to(env.device)
        env._support_transfer_inertias = _tensor(data.body_inertia).clone().to(env.device).reshape(env.num_envs, -1, 3, 3)
        if not bool(torch.isfinite(env._support_transfer_masses).all() & (env._support_transfer_masses >= 0).all() & (env._support_transfer_masses.sum(-1) > 0).all()):
            raise ValueError('Invalid simulator body masses')
        if not bool(torch.isfinite(env._support_transfer_inertias).all()):
            raise ValueError('Invalid simulator body inertias')
    feet, contacts, sole_points = env._support_transfer_ids
    contact = _tensor(sensor.data.net_forces_w)[:, contacts, 2] > 5.
    state = _state(env)
    state['episode_seen'].fill_(True)
    event = contact_events(state, contact)
    command = _tensor(env.command_manager.get_command('base_velocity'))[:, :2]
    event &= (command.norm(dim=-1) >= .2)[:, None]
    env_ids, new_feet = event.nonzero(as_tuple=True)
    rate = torch.zeros(env.num_envs, device=env.device)
    env._support_transfer_last = dict(event=event, cost=rate.clone(), absolute=rate.clone(), held=rate.clone())
    if env_ids.numel():
        soles = transform_sole_points(_tensor(data.body_pos_w)[:, feet],
            _tensor(data.body_quat_w)[:, feet][..., [3, 0, 1, 2]], sole_points)
        heights, valid = surface_height(env, soles, exact=True)
        # Never let a suspended foot or a point over a deep pit provide force.
        available = valid & ((soles[..., 2]-heights).abs() <= .035) & contact[..., None]
        points = soles.clone(); points[..., 2] = heights
        com, velocity, momentum, mass = centroidal_state(
            env._support_transfer_masses[env_ids], _tensor(data.body_com_pos_w)[env_ids],
            _tensor(data.body_com_lin_vel_w)[env_ids], _tensor(data.body_ang_vel_w)[env_ids],
            env._support_transfer_inertias[env_ids], _tensor(data.body_com_quat_w)[env_ids])
        yaw = corridor_runtime.update(env)['reference_yaw'][env_ids]
        cmd = command[env_ids]
        world_command = torch.stack((yaw.cos()*cmd[:, 0]-yaw.sin()*cmd[:, 1],
                                     yaw.sin()*cmd[:, 0]+yaw.cos()*cmd[:, 1]), -1)
        inputs = (points[env_ids], available[env_ids], com, velocity, momentum, mass, world_command, new_feet)
        result = solve(*inputs, horizon=.3, length_scale=.5,
                       angular_momentum_band_ratio=(.04, .04, .08), velocity_error_scale=.5,
                       chunk_size=32)
        finite = result['finite']
        increment = torch.where(finite, result['increment'].clamp(0., 1.), 1.)
        # RewardManager multiplies this rate by dt: max one unit per event.
        # Both feet confirming at once are explicitly two events.
        rate.index_add_(0, env_ids, increment / env.step_dt)
        for name, value in (('cost', increment), ('absolute', result['new_cost']), ('held', result['held_cost'])):
            env._support_transfer_last[name].index_add_(0, env_ids, value)
        state['episode_events'].index_add_(0, env_ids, torch.ones_like(increment))
        state['events'] += increment.numel(); state['cost_sum'] += increment.sum()
        state['absolute_sum'] += result['new_cost'].sum(); state['held_sum'] += result['held_cost'].sum()
        state['invalid'] += (~finite).sum(); state['point_sum'] += available[env_ids].sum()
        old_available = available[env_ids, 1-new_feet].any(-1)
        state['old_available'] += old_available.sum()
        state['nonzero_increment'] += (increment > 1e-5).sum()
        state['held_high'] += (result['held_cost'] > .5).sum()
        state['momentum_sum'] += torch.nan_to_num(momentum.norm(dim=-1), nan=0., posinf=0., neginf=0.).sum()
        state['speed_sum'] += torch.nan_to_num(velocity[:, :2].norm(dim=-1), nan=0., posinf=0., neginf=0.).sum()
        if not getattr(env, '_support_transfer_native_probe_printed', False):
            print('[SupportTransfer native probe]', dict(events=int(increment.numel()),
                mean_mass_kg=float(mass.mean()), mean_available_points=float(available[env_ids].float().sum((-1, -2)).mean()),
                mean_absolute_cost=float(result['new_cost'].mean()), mean_held_cost=float(result['held_cost'].mean()),
                mean_increment=float(increment.mean()), finite_fraction=float(finite.float().mean()),
                old_support_available_fraction=float(old_available.float().mean()),
                nonzero_increment_fraction=float((increment > 1e-5).float().mean()),
                angular_momentum_includes_body_spin=True, actor_observations_added=False), flush=True)
            env._support_transfer_native_probe_printed = True
    denominator = state['events'].clamp_min(1.)
    env._support_transfer_log = {
        'SupportTransfer/events_per_env': state['events']/env.num_envs,
        'SupportTransfer/increment_per_event': state['cost_sum']/denominator,
        'SupportTransfer/absolute_per_event': state['absolute_sum']/denominator,
        'SupportTransfer/held_per_event': state['held_sum']/denominator,
        'SupportTransfer/old_support_available_fraction': state['old_available']/denominator,
        'SupportTransfer/nonzero_increment_fraction': state['nonzero_increment']/denominator,
        'SupportTransfer/held_high_fraction': state['held_high']/denominator,
        'SupportTransfer/invalid_fraction': state['invalid']/denominator,
        'SupportTransfer/available_points_per_event': state['point_sum']/denominator,
        'SupportTransfer/angular_momentum_kg_m2_s': state['momentum_sum']/denominator,
        'SupportTransfer/com_speed_m_s': state['speed_sum']/denominator,
        'SupportTransfer/episodes_without_event_fraction': state['episodes_without_event']/state['episodes_observed'].clamp_min(1),
    }
    env._support_transfer_tick = env.common_step_counter
    env._support_transfer_rate = rate
    return rate


@torch.no_grad()
def upper_excess_cost(env):
    """Simple pose-band control arm; leaves shoulder pitch and elbow free."""
    if getattr(env.cfg, 'beamdojo_support_transfer_arm', None) != 'upper':
        raise ValueError('Upper control requires explicit experiment contract')
    robot = env.scene['robot']
    if not hasattr(env, '_support_upper_ids'):
        selected = [(i, .2 if 'shoulder_roll' in name else .25 if 'shoulder_yaw' in name else .3)
                    for i, name in enumerate(robot.joint_names)
                    if 'shoulder_roll' in name or 'shoulder_yaw' in name or 'wrist' in name]
        if len(selected) != 10:
            raise ValueError('Expected G1 4 shoulder-roll/yaw and 6 wrist joints')
        env._support_upper_ids = ([i for i, _ in selected], torch.tensor([b for _, b in selected], device=env.device))
    ids, allowance = env._support_upper_ids
    deviation = (_tensor(robot.data.joint_pos)[:, ids]-_tensor(robot.data.default_joint_pos)[:, ids]).abs()
    cost = (deviation-allowance).clamp_min(0).square().sum(-1)
    env._support_transfer_log = {'SupportTransfer/upper_excess_pose_cost': cost.mean()}
    return cost
