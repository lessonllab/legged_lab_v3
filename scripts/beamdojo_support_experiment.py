"""Explicit, CPU-readable contract for the paired support-risk experiment."""
from copy import deepcopy

EVENT_PARAMETERS = dict(unsafe_start=.3, unsafe_full=.8, duration_scale=.1, force_threshold=5.)
TAIL_PARAMETERS = dict(identity_margin_halfwidth=.5, tail_width_halfwidth=.5)
EVENT_FUNCTION = 'legged_lab.tasks.locomotion.beamdojo.contact_event_risk.event_contact_risk'


def experiment_spec(arm):
    if arm not in ('baseline', 'event'):
        raise ValueError('Unknown support experiment arm')
    return {
        'version': 1, 'arm': arm, 'training_domain': 'physical',
        'fixed_levels': [0, 7, 8, 9, 10], 'assignment': 'level = fixed_levels[env_id % 5]',
        'terrain_progression': False, 'evaluation_level_override': True,
        'event_reward_weight': 0. if arm == 'baseline' else -17.5,
        'event_parameters': EVENT_PARAMETERS.copy(),
        'event_reward_channel': 'sparse together with existing foothold; dense motion unchanged',
        'event_formula': 'sum_foot severity * (1-exp(-integral_contact(severity*dt)/duration_scale)); severity=smoothstep((min3_unsafe_fraction-unsafe_start)/(unsafe_full-unsafe_start))',
        'event_scope': 'Fz>force_threshold; exposure resets at loss of contact and episode reset; no future labels',
        'mean_tail_parameters': TAIL_PARAMETERS.copy(),
        'task_and_command_rewards': 'unchanged; no movement gate on event cost',
        'classification': 'exploratory mechanism test, not original adaptive BeamDojo replication',
    }


def spec_for_contract(parent):
    spec = parent.get('support_experiment')
    if spec is not None and spec != experiment_spec(spec.get('arm')):
        raise ValueError('Unknown or changed support experiment contract')
    return deepcopy(spec)


def apply_config(cfg, arm):
    if (cfg.beamdojo_stage != 2 or getattr(cfg, 'beamdojo_virtual_course', False)
            or not getattr(cfg, 'beamdojo_virtual_handoff', False)
            or not getattr(cfg, 'beamdojo_bound_joint_actions', False)
            or cfg.beamdojo_course_schedule != 'smooth_top_v1'):
        raise ValueError('Support experiment requires physical bounded v16 task')
    from isaaclab.managers import RewardTermCfg
    from legged_lab.tasks.locomotion.beamdojo.contact_event_risk import event_contact_risk
    spec = experiment_spec(arm)
    cfg.beamdojo_support_experiment = spec
    cfg.beamdojo_support_fixed_levels = bool(cfg.beamdojo_curriculum)
    cfg.beamdojo_contact_event_risk = True
    cfg.recovery_easy_replay_fraction = 0.
    cfg.recovery_challenge_fraction = 0.
    cfg.rewards.contact_event_risk = RewardTermCfg(
        func=event_contact_risk, weight=spec['event_reward_weight'], params=EVENT_PARAMETERS.copy())


def contract_amendment(source, arm):
    if source.get('version') != 16 or source.get('stage') != 2:
        raise ValueError('Support experiment amendment requires v16 Stage 2')
    if 'support_experiment' in source:
        raise ValueError('Support experiment contract is already amended')
    result = deepcopy(source)
    spec = experiment_spec(arm)
    result['support_experiment'] = spec
    result['action_mean_tail_bounds'] = {
        'version': 1, 'kind': 'identity_interior_rational_gaussian_mean',
        'bounds_source': 'execution_action_bounds', **TAIL_PARAMETERS,
        'raw_mlp_penalty': True, 'execution_clamp_unchanged': True,
        'likelihood': 'Gaussian with transformed mean; sampling/logprob/KL/entropy share this distribution',
        'guard': 'raw MLP checked finite; Gaussian samples retain abs100 guard; no finite-sample guarantee',
    }
    result['curriculum'].update(initial_training_levels=[0, 7, 8, 9, 10],
                                terrain_owner='fixed support experiment strata',
                                consecutive_passes_to_promote=None, demote_on_failure=False)
    result['recovery']['easy_replay_fraction'] = 0.
    for key in ('reward_weights', 'reward_functions', 'reward_parameters'):
        result[key].pop('contact_event_risk', None)
    if spec['event_reward_weight']:
        result['reward_weights']['contact_event_risk'] = spec['event_reward_weight']
        result['reward_functions']['contact_event_risk'] = EVENT_FUNCTION
        result['reward_parameters']['contact_event_risk'] = EVENT_PARAMETERS.copy()
    result['adaptations'].append('Support mechanism experiment: fixed equal physical level strata and common identity-interior mean-tail safeguard; only event arm adds contact-event risk to sparse critic')
    return result


def migration_contract_matches(parent, actual, key, arm):
    try:
        expected = contract_amendment(parent, arm)
    except (KeyError, ValueError, TypeError):
        return False
    return expected.get(key) == actual.get(key) and (key in expected) == (key in actual)


def verify_mean_tail(algorithm, env):
    import torch
    from legged_lab.tasks.locomotion.beamdojo.action_bounds import normalized_bounds
    low, high = [], []
    for name in env.action_manager.active_terms:
        left, right = normalized_bounds(env.action_manager.get_term(name))
        low.append(left.amax(dim=0)); high.append(right.amin(dim=0))
    expected = {'version': 1, 'kind': 'identity_interior_rational_gaussian_mean',
                'lower': torch.cat(low).cpu().tolist(), 'upper': torch.cat(high).cpu().tolist(),
                **TAIL_PARAMETERS, 'raw_mlp_penalty': True, 'execution_clamp_unchanged': True}
    if algorithm.action_mean_tail_bounds_report() != expected:
        raise ValueError('Saved mean-tail distribution disagrees with support experiment contract')
    return expected


def save_initial_snapshot(env, observations, algorithm, path):
    """Only initial state is paired; stochastic training rollouts subsequently diverge."""
    import torch
    def tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if hasattr(value, 'torch'):
            return value.torch
        try:
            return torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError):
            import warp
            return warp.to_torch(value)
    def cpu(value):
        return tensor(value).detach().cpu().clone()
    robot = env.scene['robot']
    data = robot.data
    with torch.inference_mode():
        first_action = algorithm.actor(observations)
    view = robot.root_view
    snapshot = {
        'observations': {k: cpu(v) for k, v in observations.items()},
        'first_action': cpu(first_action),
        'root_pos_local': cpu(tensor(data.root_pos_w) - tensor(env.scene.env_origins)),
        'root_quat_w': cpu(data.root_quat_w), 'joint_pos': cpu(data.joint_pos),
        'joint_vel': cpu(data.joint_vel), 'root_lin_vel_w': cpu(data.root_lin_vel_w),
        'root_ang_vel_w': cpu(data.root_ang_vel_w), 'levels': cpu(env.beamdojo_levels),
        'command': cpu(env.command_manager.get_command('base_velocity')),
        'body_masses': cpu(view.get_masses()), 'materials': cpu(view.get_material_properties()),
        'stiffness': cpu(data.joint_stiffness), 'damping': cpu(data.joint_damping),
    }
    torch.save(snapshot, path)
    return str(path.resolve())
