"""Explicit frozen-level two-column training migration from baseline A81802.

Contract helpers are CPU-readable without importing Isaac Sim. Native imports
are deferred until build_corridor_config, after the application is launched.
"""
from copy import deepcopy
import hashlib
import json
import math

SOURCE_CONTRACT_SHA256 = 'c321f9f022f7c90c15e5a4a44d61f71bdea555513e9a5f5956a26061981426dc'
RUNTIME = 'legged_lab.tasks.locomotion.beamdojo.corridor_runtime'
GPU_PAIR_CAPACITIES = {
    'gpu_found_lost_pairs_capacity': 2**23,
    'gpu_found_lost_aggregate_pairs_capacity': 2**27,
    'gpu_total_aggregate_pairs_capacity': 2**23,
}


BOUNDARY_V2 = {
    'version': 2,
    'longitudinal_inset_m': .75,
    'x_bounds': 'per-level supported_x_bounds_m inset by 0.75 m at both ends',
    'y_administrative_boundary': False,
    'y_rule': 'leaving the actual lateral lane outside the start platform fails directional qualification; never a successful administrative exit',
    'trigger': 'root local X <= safe_min or >= safe_max',
    'termination': 'administrative timeout at safe boundary; existing terminal support and directional qualification retained',
    'collision_geometry_changed': False,
}


BODY_ALIGNMENT_V2 = {
    'version': 2,
    'tracking_function': RUNTIME + '.track_lin_vel_xy_dual_frame_exp',
    'formula': 'exp(-max(episode_reference_yaw_xy_error_squared,current_body_yaw_xy_error_squared)/std_squared)',
    'reward_weight': 1., 'std': .5,
    'same_frame_behavior': 'identical to original episode-reference tracking reward',
    'actor_inputs_changed': False,
    'migration': 'full actor, critics, optimizer and iteration resume; only tracking reward function amended',
}


MIXED_LEVEL_WEIGHTS = (12, 9, 9, 9, 9, 3, 3, 3, 1, 1, 1)
MIXED_LEVEL_SAMPLER = {
    'version': 1,
    'weights': list(MIXED_LEVEL_WEIGHTS),
    'assignment': 'largest remainder quotas stratified by terrain type; weighted deficit interleaving within each type',
    'episode_resampling': False,
}

MIXED_TERRAIN_OWNER = 'canonical mixed levels 0..10 with persistent per-environment assignment'
MIXED_ADVANCEMENT = 'fixed mixture across chunks; no within-chunk curriculum or external single-level advancement'
MIXED_RESUME = 'full model/optimizer resume with canonical fixed mixed-level assignment'


def mixed_levels_for_contract(parent):
    section = parent.get('corridor_training', {})
    declaration = section.get('mixed_level_sampler')
    if declaration is None:
        if (parent.get('curriculum', {}).get('terrain_owner') == MIXED_TERRAIN_OWNER
                or section.get('level_advancement') == MIXED_ADVANCEMENT
                or section.get('subsequent_chunks') == MIXED_RESUME):
            raise ValueError('Mixed sampling metadata was removed from its training contract')
        return False
    if declaration != MIXED_LEVEL_SAMPLER:
        raise ValueError('Unsupported or altered mixed-level sampler')
    if (parent.get('curriculum', {}).get('terrain_owner') != MIXED_TERRAIN_OWNER
            or parent.get('curriculum', {}).get('initial_training_levels') != list(range(11))
            or parent.get('physical_alignment', {}).get('physical_initial_levels') != list(range(11))
            or section.get('level_advancement') != MIXED_ADVANCEMENT
            or section.get('subsequent_chunks') != MIXED_RESUME):
        raise ValueError('Mixed sampler disagrees with curriculum declaration')
    return True


def body_alignment_for_contract(parent):
    section = parent.get('corridor_training')
    if section is None:
        return True
    declaration = section.get('body_alignment')
    if declaration is None:
        return False
    if declaration != BODY_ALIGNMENT_V2:
        raise ValueError('Unsupported or altered body-alignment declaration')
    return True


def boundary_version_for_contract(parent):
    """Missing boundary metadata means the original map boundary, never v2."""
    section = parent.get('corridor_training')
    if section is None:
        return 2
    boundary = section.get('boundary')
    if boundary is None:
        return 1
    if boundary != BOUNDARY_V2:
        raise ValueError('Unsupported or altered corridor boundary declaration')
    return 2


def contract_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_source_contract(parent):
    """Return the exact original A contract, including through corridor resumes."""
    if 'corridor_training' in parent:
        section = parent['corridor_training']
        if section.get('version') != 1:
            raise ValueError('Unsupported corridor training profile')
        source = section.get('original_source_contract')
    else:
        source = parent
    if not isinstance(source, dict) or contract_sha256(source) != SOURCE_CONTRACT_SHA256:
        raise ValueError('Corridor migration requires the exact baseline A81802 contract')
    boundary_version_for_contract(parent)
    body_alignment_for_contract(parent)
    mixed_levels_for_contract(parent)
    from beamdojo_exploration_contract import validate
    validate(parent.get('action_exploration_scale'))
    return deepcopy(source)


def _level(level):
    if type(level) is not int or not 0 <= level <= 10:
        raise ValueError('Corridor level must be an integer in 0..10')
    return level


def _expected_rewards(source, body_alignment=False):
    expected = {key: deepcopy(source[key]) for key in
                ('reward_weights', 'reward_functions', 'reward_parameters')}
    expected['reward_weights'].update(track_lin_vel_xy=1., course_heading=-.5,
                                     course_lateral=-.5, corridor_progress=.5)
    expected['reward_functions'].update(track_lin_vel_xy=RUNTIME+'.track_lin_vel_xy_yaw_frame_exp',
        course_heading=RUNTIME+'.heading_error_l2', course_lateral=RUNTIME+'.lateral_error_l2',
        corridor_progress=RUNTIME+'.signed_progress')
    expected['reward_parameters'].update(track_lin_vel_xy={'std': .5, 'command_name': 'base_velocity'},
        course_heading={'scale': .5}, course_lateral={'scale': .3},
        corridor_progress={'normalization_speed': 1.})
    if body_alignment:
        expected['reward_functions']['track_lin_vel_xy'] = BODY_ALIGNMENT_V2['tracking_function']
    return expected


def _semantic_parameter(value):
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, dict):
        return {k: _semantic_parameter(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_semantic_parameter(v) for v in value]
    fields = ('name', 'joint_names', 'body_names', 'preserve_order')
    if all(hasattr(value, key) for key in fields):
        return {key: _semantic_parameter(getattr(value, key)) for key in fields}
    raise TypeError(f'Unsupported reward parameter: {type(value)}')


def validate_runtime_rewards(cfg, source_contract):
    """Inspect real config; legacy common.contract deletes restored heading terms."""
    source = validate_source_contract(source_contract)
    active = {name: term for name, term in vars(cfg.rewards).items()
              if term is not None and hasattr(term, 'weight') and term.weight != 0}
    actual = {
        'reward_weights': {k: float(v.weight) for k, v in active.items()},
        'reward_functions': {k: f'{v.func.__module__}.{v.func.__name__}' for k, v in active.items()},
        'reward_parameters': {k: _semantic_parameter(v.params) for k, v in active.items()},
    }
    for section, expected in _expected_rewards(source, getattr(cfg, 'beamdojo_corridor_body_alignment', False)).items():
        if actual[section] != expected:
            raise ValueError(f'Actual corridor config differs from declared {section}')
    return actual


def make_corridor_contract(base_contract, source_contract, level, *, cfg, evaluation_fixed_level=False):
    """Amend the actual runtime contract; preserve every unrelated field."""
    source = validate_source_contract(source_contract)
    level = _level(level)
    parent_mixed = mixed_levels_for_contract(source_contract)
    runtime_mixed = getattr(cfg, 'beamdojo_corridor_mixed_levels', False)
    if type(runtime_mixed) is not bool:
        raise ValueError('Runtime mixed sampler enable flag must be boolean')
    if evaluation_fixed_level:
        if runtime_mixed:
            raise ValueError('Fixed-level evaluation must disable runtime mixed sampling')
        contract_mixed = parent_mixed
    else:
        if parent_mixed and not runtime_mixed:
            raise ValueError('Cannot silently remove the checkpoint mixed sampler')
        contract_mixed = runtime_mixed
    if runtime_mixed and tuple(getattr(cfg, 'beamdojo_corridor_mixed_weights', ())) != MIXED_LEVEL_WEIGHTS:
        raise ValueError('Runtime mixed weights differ from canonical sampler')
    expected_rewards = validate_runtime_rewards(cfg, source)
    if any(getattr(cfg.sim.physics, name) != value for name, value in GPU_PAIR_CAPACITIES.items()):
        raise ValueError('Corridor GPU pair capacities differ from validated profile')
    result = deepcopy(base_contract)
    # These are actor/control compatibility guarantees, not merely metadata.
    for key in ('version', 'stage', 'task', 'robot', 'action_dimension', 'joint_names',
                'asset_sha256', 'observation_source', 'observation_dimensions', 'map',
                'gait_reference', 'foot_safety_projection', 'policy_hz', 'simulation_hz',
                'ppo', 'observations', 'g1_control_reference', 'execution_action_bounds'):
        if result.get(key) != source.get(key):
            raise ValueError(f'Corridor migration changed protected actor/control field: {key}')
    result.pop('support_experiment', None)
    result.pop('training_budget', None)
    result['action_mean_tail_bounds'] = deepcopy(source['action_mean_tail_bounds'])
    if source_contract.get('action_exploration_scale') is not None:
        result['action_exploration_scale']=deepcopy(source_contract['action_exploration_scale'])
    for section, expected in expected_rewards.items():
        common_expected = {k: v for k, v in expected.items() if k not in ('course_heading', 'course_lateral')}
        if result[section] != common_expected:
            raise ValueError(f'Actual base contract differs from declared {section}')
    for section in ('reward_weights', 'reward_functions', 'reward_parameters'):
        for name, expected in source[section].items():
            if name != 'track_lin_vel_xy' and result[section].get(name) != expected:
                raise ValueError(f'Corridor migration changed unrelated reward: {section}.{name}')
    for key in ('reward_weights', 'reward_functions', 'reward_parameters'):
        result[key].pop('contact_event_risk', None)
    result['reward_weights'].update(track_lin_vel_xy=1., course_heading=-.5,
                                   course_lateral=-.5, corridor_progress=.5)
    result['reward_functions'].update(track_lin_vel_xy=RUNTIME+'.track_lin_vel_xy_yaw_frame_exp',
        course_heading=RUNTIME+'.heading_error_l2', course_lateral=RUNTIME+'.lateral_error_l2',
        corridor_progress=RUNTIME+'.signed_progress')
    result['reward_parameters'].update(track_lin_vel_xy={'std': .5, 'command_name': 'base_velocity'},
        course_heading={'scale': .5}, course_lateral={'scale': .3},
        corridor_progress={'normalization_speed': 1.})
    # RewardManager and dual-critic semantics remain unchanged: all new terms
    # join dense rewards, while the existing foothold term remains sparse.
    result['commands'].update(initial_ranges={'lin_vel_x': [.5, 1.], 'lin_vel_y': [0., 0.],
        'ang_vel_z': [0., 0.]}, limit_ranges={'lin_vel_x': [.5, 1.], 'lin_vel_y': [0., 0.],
        'ang_vel_z': [0., 0.]}, standing_fraction=0., heading_command=False,
        resampling_s=20., curriculum='disabled; forward speed uniformly sampled in [0.5,1.0] m/s')
    result['curriculum'].update(enabled=False, initial_level=level,
        initial_training_levels=[level], terrain_owner='externally selected two-column level',
        consecutive_passes_to_promote=None, demote_on_failure=False)
    result['course_tracking']['motion_reward_shaping'] = (
        'episode-reference velocity tracking; heading and lateral penalties restored; signed progress velocity reward')
    result['course_tracking']['reward_only_truth'] = ['episode reset world yaw', 'episode reset world XY', 'world XY velocity']
    result['course_tracking']['straight_command_fraction'] = 1.
    result['course_tracking']['command_sampling_override'] = 'all commands forward with zero lateral/yaw rate and no standing'
    result['virtual_handoff'].update(motion_rewards='official rewards with episode-reference velocity tracking, restored heading/lateral and signed progress',
        removed_rewards=[], recovery_scope='disabled; external fixed-level chunk selection')
    result['alignment_scope'] = 'A81802 actor/control preserved; explicit real two-column direction-tracking rewards and reset/command distribution'
    result['terrain_family'] = 'two_original_columns_shared_real_collision_and_ideal_heightmap'
    result['map']['surface_source'] = 'two retained original grid columns plus central platform; matching real collision heightfield and three-neighbour-minimum heightmap'
    result['physical_alignment'].update(spawn='central platform, selected pair midpoint Y +/-0.05 m; yaw 0 or pi',
        physical_initial_levels=[level],
        spawn_yaw_range_rad=None, min_signed_progress_m=3.)
    qualification = ('At timeout or native boundary: original terminal physical support qualification AND '
        'episode-reference forward displacement >=3 m AND never leave real lane envelope outside start platform '
        'AND heading RMS <=15 degrees AND heading peak <=30 degrees; no fall')
    result['physical_alignment']['success'] = qualification
    result['course_integrity']['stage2_qualification'] = qualification
    result['curriculum']['qualification'] = qualification
    result['recovery'].update(easy_replay_fraction=0., training_only_curriculum=False,
        training_yaw_half_range_rad=None, evaluation_yaw_range_rad=None,
        pass_rule='disabled recovery promotions; corridor qualification used')
    result['corridor_training'] = {
        'version': 1, 'level': level, 'original_source_contract_sha256': SOURCE_CONTRACT_SHA256,
        'gpu_pair_capacities': dict(GPU_PAIR_CAPACITIES),
        'original_source_contract': source,
        'geometry': 'two adjacent original Y columns nearest map center; original stone size/pitch/phase and central 1m platform; other raised support removed',
        'geometry_source_layout_stage': 1, 'geometry_schedule': 'smooth_top_v1',
        'physical_and_perceived_geometry_identical': True,
        'reset_yaw_choices_rad': [0., math.pi], 'reset_yaw_probabilities': [.5, .5],
        'reset_y_relative_to_pair_midpoint_m': [-.05, .05],
        'direction_reference': 'immutable episode reset yaw; command speed resampling never rotates reference',
        'lateral_reference': 'immutable episode reset XY; real lane bounds separately gate success',
        'progress_reward': 'min(world XY velocity projected onto reset forward direction / 1 m/s, 1); negative values unclipped to debit fast retreat; active for commanded speed >=0.1m/s',
        'qualification_thresholds': {'net_forward_displacement_min_m': 3.,
            'heading_rms_max_rad': math.pi / 12, 'heading_peak_max_rad': math.pi / 6,
            'lane_departures_allowed': 0, 'falls_allowed': 0,
            'terminal_physical_support_required': True, 'completion': '20s timeout or native boundary'},
        'reward_channels': {'foothold': 'sparse', 'other_rewards': 'dense'},
        'actor_inputs_changed': False, 'privileged_reward_truth_added_to_actor': False,
        'initial_migration': 'actor and normalization transferred; fresh critics and optimizer; saved action distribution and mean-tail safeguards restored',
        'subsequent_chunks': 'full model/optimizer resume; only externally selected fixed level may change',
        'level_advancement': 'external evaluator decision between 1000-update chunks; no within-chunk curriculum',
        'initial_run_plan': {'num_envs': 4096, 'total_updates': 3000,
            'chunk_updates': 1000, 'initial_level': 0, 'maximum_level': 10,
            'evaluation_yaw_degrees': [0, 180], 'episodes_per_heading': 64,
            'promote_one_level': 'directed passage rate >=0.80 separately in both headings',
            'demote_one_level': 'either heading directed passage rate <0.50; floor at level 0',
            'otherwise': 'retain current level'},
    }
    if contract_mixed:
        result['corridor_training']['mixed_level_sampler'] = deepcopy(MIXED_LEVEL_SAMPLER)
        result['curriculum']['terrain_owner'] = MIXED_TERRAIN_OWNER
        result['curriculum']['initial_training_levels'] = list(range(11))
        result['physical_alignment']['physical_initial_levels'] = list(range(11))
        result['corridor_training']['level_advancement'] = MIXED_ADVANCEMENT
        result['corridor_training']['subsequent_chunks'] = MIXED_RESUME
    if getattr(cfg, 'beamdojo_corridor_body_alignment', False):
        result['corridor_training']['body_alignment'] = deepcopy(BODY_ALIGNMENT_V2)
        result['reward_functions']['track_lin_vel_xy'] = BODY_ALIGNMENT_V2['tracking_function']
    boundary_version = getattr(cfg, 'beamdojo_corridor_boundary_version', 1)
    if boundary_version not in (1, 2):
        raise ValueError('Unsupported runtime corridor boundary version')
    if boundary_version == 2:
        result['corridor_training']['boundary'] = deepcopy(BOUNDARY_V2)
        # X uses support-derived bounds; Y has no administrative boundary.
        # Lateral departure remains a directional-qualification failure gate.
        result['course_integrity']['stage2_boundary_abs_xy_m'] = [None, None]
    elif source_contract.get('corridor_training') is not None and (
            'gpu_pair_capacities' not in source_contract['corridor_training']):
        # Preserve old prepared contracts verbatim on ordinary evaluation.
        # Resource metadata may be filled only in an explicit v2 migration.
        result['corridor_training'].pop('gpu_pair_capacities', None)
    result.setdefault('adaptations', []).append('Explicit two-column directed-locomotion training; original actor inputs and numerical distribution safeguards retained')
    return result


def validate_corridor_resume(parent, actual, *, allow_boundary_migration=False,
                             allow_body_alignment_migration=False, allow_mixed_migration=False):
    """Reject drift except the selected fixed level and recorded update budget."""
    validate_source_contract(parent)
    validate_source_contract(actual)
    if 'corridor_training' not in parent:
        raise ValueError('Full corridor resume requires a corridor checkpoint')
    def normalized(value):
        value = deepcopy(value)
        value.pop('training_budget', None)
        value['corridor_training'].pop('level', None)
        for key in ('initial_level', 'initial_training_levels'):
            value['curriculum'].pop(key, None)
        value['physical_alignment'].pop('physical_initial_levels', None)
        return value
    old, new = normalized(parent), normalized(actual)
    old_version = boundary_version_for_contract(parent)
    new_version = boundary_version_for_contract(actual)
    if allow_boundary_migration and old_version == 1 and new_version == 2:
        # Only the canonical boundary declaration and its old X field change.
        if new['course_integrity'].get('stage2_boundary_abs_xy_m') != [None, None]:
            raise ValueError('Incorrect safe-boundary contract')
        if old['course_integrity'].get('stage2_boundary_abs_xy_m') != [15.9, 15.9]:
            raise ValueError('Source did not use the expected legacy boundary')
        new['corridor_training'].pop('boundary')
        new['course_integrity']['stage2_boundary_abs_xy_m'] = [15.9, 15.9]
        if 'gpu_pair_capacities' not in old['corridor_training']:
            if new['corridor_training'].get('gpu_pair_capacities') != GPU_PAIR_CAPACITIES:
                raise ValueError('Only the exact GPU resource amendment is permitted')
            new['corridor_training'].pop('gpu_pair_capacities')
    if (allow_body_alignment_migration and not body_alignment_for_contract(parent)
            and body_alignment_for_contract(actual)):
        old_function = RUNTIME + '.track_lin_vel_xy_yaw_frame_exp'
        if old['reward_functions'].get('track_lin_vel_xy') != old_function:
            raise ValueError('Body alignment requires original corridor tracking function')
        if new['reward_functions'].get('track_lin_vel_xy') != BODY_ALIGNMENT_V2['tracking_function']:
            raise ValueError('Body alignment requires exact dual-frame tracking function')
        new['reward_functions']['track_lin_vel_xy'] = old_function
        new['corridor_training'].pop('body_alignment')
    if (allow_mixed_migration and not mixed_levels_for_contract(parent)
            and mixed_levels_for_contract(actual)):
        if old['curriculum'].get('terrain_owner') != 'externally selected two-column level':
            raise ValueError('Mixed migration requires the original fixed-level corridor')
        new['corridor_training'].pop('mixed_level_sampler')
        new['curriculum']['terrain_owner'] = old['curriculum']['terrain_owner']
        for key in ('level_advancement', 'subsequent_chunks'):
            new['corridor_training'][key] = old['corridor_training'][key]
    if old != new:
        raise ValueError('Corridor resume contract mismatch outside selected level')
    return True


def build_corridor_config(source_contract, num_envs, seed, device, level, *, boundary_version=None,
                          body_alignment=None, mixed_levels=None):
    """Build simulator config, preserving the original observation/control stack."""
    source = validate_source_contract(source_contract)
    level = _level(level)
    if boundary_version is None:
        boundary_version = boundary_version_for_contract(source_contract)
    if boundary_version not in (1, 2):
        raise ValueError('Boundary version must be 1 or 2')
    if body_alignment is None:
        body_alignment = body_alignment_for_contract(source_contract)
    if type(body_alignment) is not bool:
        raise ValueError('Body alignment must be a boolean')
    if mixed_levels is None:
        mixed_levels = mixed_levels_for_contract(source_contract)
    if type(mixed_levels) is not bool:
        raise ValueError('Mixed levels must be a boolean')
    from beamdojo_common import build_config, scanner_profile_for_contract
    from beamdojo_corridor_diagnostic import install_two_column_corridor, corridor_metadata
    from isaaclab.managers import RewardTermCfg
    from legged_lab.tasks.locomotion.beamdojo import corridor_runtime, corridor_mixed_runtime
    if mixed_levels and corridor_mixed_runtime.spec() != MIXED_LEVEL_SAMPLER:
        raise ValueError('Mixed runtime sampler differs from canonical contract')
    cfg, agent = build_config(stage=2, num_stones=source['num_stones'], num_envs=num_envs,
        seed=seed, device=device, perception='ideal_heightmap', paper_aligned=True,
        initial_level=level, curriculum=False, unitree_motion_aligned=True,
        unitree_scanner_profile=scanner_profile_for_contract(source), course_integrity=True,
        course_tracking=True, physical_aligned=True, continuous_physical=True,
        recovery=True, recovery_max_level=10, recovery_yaw_half_range=math.pi,
        recovery_neutral_unattempted=source['recovery']['neutral_unattempted'],
        strict_foothold=True, virtual_handoff=True, bound_joint_actions=True,
        course_schedule='smooth_top_v1', recovery_speed_gate=False)
    install_two_column_corridor()
    # A fixed level puts all 4096 isolated actors on the same terrain tile.
    # Broadphase candidates need more capacity than the previous mixed levels;
    # PhysX otherwise drops interactions even though training keeps running.
    for name, value in GPU_PAIR_CAPACITIES.items():
        setattr(cfg.sim.physics, name, value)
    cfg.beamdojo_corridor_training = True
    cfg.beamdojo_corridor_boundary_version = boundary_version
    cfg.beamdojo_corridor_body_alignment = body_alignment
    cfg.beamdojo_corridor_mixed_levels = mixed_levels
    cfg.beamdojo_corridor_mixed_weights = MIXED_LEVEL_WEIGHTS
    cfg.beamdojo_corridor_geometry_by_level = [corridor_metadata(i, seed=seed) for i in range(11)]
    cfg.beamdojo_corridor_source_contract = source
    cfg.beamdojo_support_fixed_levels = False
    cfg.beamdojo_contact_event_risk = False
    cfg.recovery_easy_replay_fraction = 0.
    cfg.recovery_challenge_fraction = 0.
    cfg.curriculum = None
    cfg.tracking_straight_fraction = 1.
    command = cfg.commands.base_velocity
    command.heading_command = False
    command.rel_standing_envs = 0.
    command.resampling_time_range = (20., 20.)
    for ranges in (command.ranges, command.limit_ranges):
        ranges.lin_vel_x = (.5, 1.)
        ranges.lin_vel_y = (0., 0.)
        ranges.ang_vel_z = (0., 0.)
    tracking_function = (corridor_runtime.track_lin_vel_xy_dual_frame_exp if body_alignment
                         else corridor_runtime.track_lin_vel_xy_yaw_frame_exp)
    cfg.rewards.track_lin_vel_xy = RewardTermCfg(func=tracking_function,
        weight=1., params={'std': .5, 'command_name': 'base_velocity'})
    cfg.rewards.course_heading = RewardTermCfg(func=corridor_runtime.heading_error_l2,
        weight=-.5, params={'scale': .5})
    cfg.rewards.course_lateral = RewardTermCfg(func=corridor_runtime.lateral_error_l2,
        weight=-.5, params={'scale': .3})
    cfg.rewards.corridor_progress = RewardTermCfg(func=corridor_runtime.signed_progress,
        weight=.5, params={'normalization_speed': 1.})
    corridor_runtime.configure_reset(cfg)
    return cfg, agent
