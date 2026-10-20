"""Native Sim6 BeamDojo adaptation; imports require AppLauncher first."""
import importlib.metadata
import hashlib
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TASK = "G1-BeamDojo-29dof-v0"
REFERENCE_COMMIT = "9a47b7dfe85fdcc737c3c8963d80d05997376575"
UNITREE_REFERENCE_COMMIT = "4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
ROBOT_USD = PROJECT / "source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"


def scanner_profile_for_contract(parent):
    """Never reinterpret an existing model's spatial input based on new defaults."""
    version = parent.get('version')
    profiles = {5: 'legacy_upstream', 6: 'unitree_g1', 7: 'beamdojo_paper', 8: 'beamdojo_paper', 9: 'beamdojo_paper', 10: 'beamdojo_paper', 11: 'beamdojo_paper', 12: 'beamdojo_paper', 13: 'beamdojo_paper', 14: 'beamdojo_paper', 15: 'beamdojo_paper', 16: 'beamdojo_paper', 17: 'beamdojo_paper'}
    if version not in profiles:
        raise ValueError(f'Unsupported Unitree motion checkpoint version: {version}')
    return profiles[version]


def velocity_range_for_contract(parent):
    limits = parent.get('commands', {}).get('limit_ranges', {}).get('lin_vel_x', [0., 1.])
    return float(limits[0]), float(limits[1])


def continuous_contract_matches(parent, evaluated, key, requested=False):
    """Allow only the explicit v10 -> v11 passage/reset and spawn revision."""
    transfer = (requested and parent.get('version') == 10 and evaluated.get('version') == 11
                and parent.get('stage') == evaluated.get('stage') == 2)
    if transfer and key == 'version':
        return True
    allowed = {
        'physical_alignment': {'version', 'continuous_passage', 'reset_on_distance_milestone',
            'spawn_yaw_range_rad', 'min_signed_progress_m', 'min_net_displacement_m',
            'qualification_timing', 'success'},
        'course_integrity': {'stage2_qualification'},
    }
    if transfer and key in allowed:
        old, new = parent.get(key), evaluated.get(key)
        return (isinstance(old, dict) and isinstance(new, dict)
                and {k:v for k,v in old.items() if k not in allowed[key]}
                == {k:v for k,v in new.items() if k not in allowed[key]})
    return parent.get(key) == evaluated.get(key)


def recovery_contract_matches(parent, evaluated, key, requested=False):
    """Allow exactly the declared v10/v11 -> v12 task-recovery amendment."""
    transfer = (requested and parent.get('version') in (10,11) and evaluated.get('version') == 12
                and parent.get('stage') == evaluated.get('stage') == 2)
    if not transfer:
        return parent.get(key) == evaluated.get(key)
    if key == 'version':
        return True
    if key in ('reward_weights','reward_functions','reward_parameters'):
        old,new = parent.get(key),evaluated.get(key)
        if not isinstance(old,dict) or not isinstance(new,dict) or set(new)-set(old) != {'command_stall'}:
            return False
        if key == 'reward_weights':
            return new['command_stall'] == -1. and {k:v for k,v in new.items() if k!='command_stall'} == old
        unchanged = set(old)-{'feet_clearance','gait'}
        if any(new.get(k)!=old[k] for k in unchanged):
            return False
        if key == 'reward_parameters':
            return new['feet_clearance']==old['feet_clearance'] and new['gait']==old['gait'] and new['command_stall']=={}
        prefix='legged_lab.tasks.locomotion.beamdojo.recovery_rewards.'
        return (new['feet_clearance']==prefix+'feet_clearance_moving' and new['gait']==prefix+'gait_moving'
                and new['command_stall']==prefix+'command_stall')
    if key in ('physical_alignment','course_integrity'):
        continuous = dict(evaluated)
        continuous['version']=11
        return continuous_contract_matches(parent,continuous,key,parent.get('version')==10)
    return parent.get(key)==evaluated.get(key)


def recovery_advancement_matches(old, new):
    """Explicitly allow cap/yaw expansion and the neutral unattempted-task fix."""
    changes = {'training_max_level','training_yaw_half_range_rad','neutral_unattempted'}
    if not isinstance(old,dict) or not isinstance(new,dict):
        return False
    if ({k:v for k,v in old.items() if k not in changes}
            != {k:v for k,v in new.items() if k not in changes}):
        return False
    old_cap,new_cap=old.get('training_max_level'),new.get('training_max_level')
    old_yaw,new_yaw=old.get('training_yaw_half_range_rad'),new.get('training_yaw_half_range_rad')
    old_neutral,new_neutral=old.get('neutral_unattempted',False),new.get('neutral_unattempted',False)
    return (type(old_cap) is int and type(new_cap) is int and 0<=old_cap<=new_cap<=8
            and type(old_yaw) in (int,float) and type(new_yaw) in (int,float)
            and math.isfinite(old_yaw) and math.isfinite(new_yaw)
            and 0.<=old_yaw<=new_yaw<=math.pi
            and type(old_neutral) is bool and type(new_neutral) is bool
            and (not old_neutral or new_neutral))


def strict_foothold_contract_matches(parent, evaluated, key, requested=False):
    """Only the explicit v12 -> v13 zero-deadband physical-foot amendment."""
    transfer = (requested and parent.get('version') == 12 and evaluated.get('version') == 13
                and parent.get('stage') == evaluated.get('stage') == 2)
    if not transfer:
        return parent.get(key) == evaluated.get(key)
    if key == 'version':
        return True
    if key == 'recovery':
        return recovery_advancement_matches(parent.get(key), evaluated.get(key))
    if key == 'physical_alignment':
        old, new = parent.get(key), evaluated.get(key)
        if not isinstance(old, dict) or not isinstance(new, dict):
            return False
        expected = {**old, 'reward_only_unsupported_deadband': 0.,
                    'strict_foothold': True, 'training_contact_error_fraction_max': .05,
                    'strict_reward_rule': 'negative count of contacting sole samples over virtual gaps; no overhang allowance and no sole-height reward test'}
        return new == expected
    return parent.get(key) == evaluated.get(key)


def virtual_handoff_contract_amendment(source, max_level=2, yaw=math.pi, neutral=False):
    """Exact declared v9 amendment, shared by generation and migration validation."""
    from copy import deepcopy
    if (type(max_level) is not int or not 0 <= max_level <= 8
            or type(yaw) not in (int, float) or not math.isfinite(yaw)
            or not 0 <= yaw <= math.pi or type(neutral) is not bool):
        raise ValueError('Invalid virtual handoff curriculum bounds')
    result = deepcopy(source)
    result.update(version=14, stage=2,
        terrain_family='stones_everywhere_shared_virtual_physical',
        alignment_scope='Restored pinned Unitree motion rewards/control/PPO; source IMU inputs and command distribution retained; virtual-query strict physical foothold',
        restored_unitree_motion=True,
        virtual_handoff={'version': 1, 'source_version': 9, 'source_stage': 1,
            'source_command_curriculum': 'preserve checkpoint ranges',
            'query': 'three-neighbour minimum', 'motion_rewards': 'official 19 plus foothold',
            'removed_rewards': ['course_heading', 'course_lateral'],
            'recovery_scope': 'terrain level management only'})
    for name in ('reward_weights', 'reward_functions', 'reward_parameters'):
        for removed in ('course_heading', 'course_lateral'):
            result[name].pop(removed, None)
    tracking = result['course_tracking']
    tracking.pop('heading_reward', None)
    tracking.pop('lateral_reward', None)
    tracking['reward_only_truth'] = []
    tracking['motion_reward_shaping'] = 'removed course_heading and course_lateral; IMU observation and command sampling retained'
    size = result['course_integrity']['stage1_map_size_m']
    success = 'At normal timeout or administrative boundary: actual supported travel >=3 m, XY net displacement >=3 m in any direction, >=1 valid foot-second, current real top contact and no fall; milestone does not terminate'
    result['physical_alignment'] = {
        'version': 2, 'map_size_m': [size, size], 'layout_source_stage': 1,
        'spawn': 'original central platform', 'physical_initial_levels': [0, min(2, max_level)],
        'min_supported_travel_m': 3., 'min_signed_progress_m': None,
        'min_net_displacement_m': 3., 'min_valid_contact_seconds': 1.,
        'partial_sole_support_allowed': True, 'reward_only_unsupported_deadband': 0.,
        'success': success, 'heightmap_surface': 'three-neighbour minimum, matching source virtual query; collision mesh retains exact raster',
        'no_absolute_pose_in_actor': True, 'continuous_passage': True,
        'reset_on_distance_milestone': False, 'spawn_yaw_range_rad': [-yaw, yaw],
        'qualification_timing': 'normal episode timeout or administrative map boundary, with current real top contact and no fall',
        'strict_foothold': True, 'training_contact_error_fraction_max': .05,
        'strict_reward_rule': 'negative count of contacting sole samples over virtual gaps; no overhang allowance and no sole-height reward test'}
    result['course_integrity'].update(stage2_course_length_m=size, stage2_course_width_m=size,
        stage2_boundary_abs_xy_m=[size / 2 - .1] * 2,
        stage2_command_task_override='central spawn for either command direction; progress goal without end-platform latch',
        route_extension='same 32x32 m raster for both stages', stage2_qualification=success)
    result['curriculum'].update(initial_training_levels=[0, min(2, max_level)],
        max_level=max_level, demote_on_failure=True,
        qualification='actual physical passage AND commanded motion; easy replay cannot promote unattempted hard levels; contacting sole-point error <=5% as on virtual terrain')
    result['recovery'] = {'version': 1, 'training_max_level': max_level,
        'training_yaw_half_range_rad': yaw, 'evaluation_yaw_range_rad': [-math.pi, math.pi],
        'easy_replay_fraction': .2, 'moving_command_threshold_m_s': .2,
        'min_moving_seconds': 1., 'min_directed_progress_ratio': .6,
        'max_xy_velocity_rmse_m_s': .4, 'min_single_support_fraction': .1,
        'lift_gait_gate': 'none; original official formulas', 'stall_weight': 0.,
        'training_only_curriculum': True, 'restored_unitree_motion': True}
    if neutral:
        result['recovery']['neutral_unattempted'] = True
    result['adaptations'] = [entry for entry in result['adaptations']
        if entry != 'User-authorized IMU-relative heading feedback and simulation-only heading/cross-track reward shaping']
    result['adaptations'].append('Explicit virtual checkpoint handoff: official motion rewards restored; source three-neighbour perception and strict foothold; physical colliders and terrain curriculum only')
    return result


def virtual_handoff_contract_matches(parent, evaluated, key, requested=False):
    """Only explicitly migrate Stage 1 v9 to the exact declared Stage 2 v14 task."""
    if not (requested and parent.get('version') == 9 and parent.get('stage') == 1
            and evaluated.get('version') == 14 and evaluated.get('stage') == 2):
        return parent.get(key) == evaluated.get(key)
    recovery = evaluated.get('recovery', {})
    try:
        expected = virtual_handoff_contract_amendment(parent,
            recovery.get('training_max_level'), recovery.get('training_yaw_half_range_rad'),
            recovery.get('neutral_unattempted', False))
    except (ValueError, KeyError, TypeError, AttributeError):
        return False
    return expected.get(key) == evaluated.get(key) and (key in expected) == (key in evaluated)


def action_bounds_contract_amendment(source):
    """Declare only execution-time bounds; sampled PPO actions remain unchanged."""
    from copy import deepcopy
    if source.get('version') != 14 or source.get('stage') != 2:
        raise ValueError('Action bounds require a Stage 2 v14 contract')
    result = deepcopy(source)
    result['version'] = 15
    result['execution_action_bounds'] = {
        'derived_from': 'asset soft_joint_pos_limits', 'scale': .25,
        'offset': 'default_joint_pos', 'apply_before': 'ActionManager',
        'last_action': 'executed bounded actions', 'ppo_action': 'unbounded sampled actions',
        'raw_guard': 'before execution clamp'}
    result['g1_control_reference']['execution_action_range'] = (
        '(asset soft_joint_pos_limits - default_joint_pos) / 0.25; per joint execution clamp before ActionManager')
    return result


def action_bounds_contract_matches(parent, evaluated, key, requested=False):
    """Only explicit Stage 2 v14 -> v15 may add the declared execution bounds."""
    if not (requested and parent.get('version') == 14 and evaluated.get('version') == 15
            and parent.get('stage') == evaluated.get('stage') == 2):
        return parent.get(key) == evaluated.get(key) and (key in parent) == (key in evaluated)
    try:
        expected = action_bounds_contract_amendment(parent)
    except (KeyError, TypeError, ValueError):
        return False
    return expected.get(key) == evaluated.get(key) and (key in expected) == (key in evaluated)


def course_schedule_spec(profile='legacy_9'):
    """Load the pure schedule without importing Isaac Sim for CLI/budget checks."""
    from importlib.util import spec_from_file_location, module_from_spec
    path = PROJECT / 'source/legged_lab/legged_lab/tasks/locomotion/beamdojo/course_schedule.py'
    spec = spec_from_file_location('_beamdojo_course_schedule_contract', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.profile_spec(profile)


def schedule_profile_for_contract(parent):
    value = parent.get('course_schedule')
    if parent.get('version') not in (16, 17):
        if value is not None:
            raise ValueError('Legacy checkpoint must not silently change its course schedule')
        return 'legacy_9'
    if value != course_schedule_spec('smooth_top_v1'):
        raise ValueError('v16 requires the exact smooth-top terrain declaration')
    return 'smooth_top_v1'


def smooth_course_contract_amendment(source, max_level=10):
    """Explicit v15 -> v16: add two geometric levels and remove speed pass gates."""
    from copy import deepcopy
    if source.get('version') != 15 or source.get('stage') != 2:
        raise ValueError('Smooth course migration requires a physical v15 checkpoint')
    previous = source['recovery']['training_max_level']
    mapped_cap = 10 if previous == 8 else previous
    if type(max_level) is not int or not mapped_cap <= max_level <= 10:
        raise ValueError('Smooth course cap must preserve or expand the mapped source cap')
    result = deepcopy(source)
    result['version'] = 16
    result['course_schedule'] = course_schedule_spec('smooth_top_v1')
    result['recovery'].update(training_max_level=max_level, speed_gate=False,
        max_xy_velocity_rmse_m_s=None, min_directed_progress_ratio=None,
        min_directed_distance_m=3.,
        pass_rule='physical passage, directed travel >=3 m, moving command >=1 s, single support >=10%, finite data; speed deviation diagnostic only')
    result['curriculum'].update(max_level=max_level,
        qualification='actual physical passage AND directed travel >=3 m AND contacting sole-point error <=5%; speed tracking is diagnostic only')
    result['curriculum']['schedule_migration'] = 'legacy levels 0..7 unchanged; legacy 8 maps to 10; inserted 8 and 9 have 38 cm and 34 cm stones, 20 cm gaps'
    result['adaptations'].append('User-authorized smooth top-level curriculum and speed scoring without a speed-deviation pass gate; original rewards/control/PPO retained')
    return result


def smooth_course_contract_matches(parent, evaluated, key, requested=False):
    if not (requested and parent.get('version') == 15 and evaluated.get('version') == 16
            and parent.get('stage') == evaluated.get('stage') == 2):
        return parent.get(key) == evaluated.get(key) and (key in parent) == (key in evaluated)
    try:
        expected = smooth_course_contract_amendment(parent, evaluated['recovery']['training_max_level'])
    except (KeyError, TypeError, ValueError):
        return False
    return expected.get(key) == evaluated.get(key) and (key in expected) == (key in evaluated)


def virtual_course_contract_amendment(source):
    """Declare virtual collision and the explicit bounded-distribution safeguard."""
    from copy import deepcopy
    if source.get('version') != 16 or source.get('stage') != 2:
        raise ValueError('Virtual course migration requires a Stage 2 v16 checkpoint')
    result = deepcopy(source)
    result.update(version=17, training_domain='virtual', collision='plane',
                  terrain_family='stones_everywhere_virtual_flat_collision')
    result['virtual_course'] = {
        'version': 1, 'source_version': 16, 'training_domain': 'virtual',
        'collision': 'plane', 'internal_stage_schema': 2,
        'layout': 'smooth_top_v1; same 32 m virtual raster and 0-10 levels',
        'reward_query': 'unchanged three-neighbour minimum and contacting 35-point unsafe count',
        'valid_support': 'measured contact near plane top AND at least one sole sample on the exact virtual raster top; strict episode contact error uses three-neighbour minimum',
        'physical_evidence': False,
        'curriculum_transfer': 'preserve actual/target levels, cap, command ranges and RNG; clear success streaks once',
        'learning_state_transfer': 'preserve actor mean, both critics and iteration; project only out-of-range std dimensions and clear only their optimizer moments'}
    result['action_distribution_bounds'] = {
        'version': 1, 'bounds_source': 'execution_action_bounds',
        'standard_deviation_min': 1e-6,
        'standard_deviation_max': 'per-axis halfwidth (upper-lower)/2',
        'projection_timing': 'after load and each optimizer step',
        'projected_adam_state': 'zero exp_avg/exp_avg_sq (and max_exp_avg_sq if present) only for projected std dimensions; preserve step and all other optimizer state',
        'mean_hard_clip': False, 'mean_out_of_bounds_loss_coefficient': .01,
        'mean_out_of_bounds_loss': 'mean(((relu(lower-mean)+relu(mean-upper))/halfwidth)^2)',
        'raw_action_guard_abs': 100.}
    alignment = result['physical_alignment']
    alignment['collision_surface'] = 'plane'
    alignment['support_mode'] = 'measured flat-plane contact filtered by virtual raster'
    alignment['heightmap_surface'] = 'three-neighbour minimum virtual query; flat physical collision'
    alignment['success'] = 'At normal timeout or administrative boundary: virtual-mask-supported travel >=3 m, XY net displacement >=3 m, >=1 valid foot-second, current measured contact on a virtual top and no fall; not evidence of physical pillar traversal'
    alignment['qualification_timing'] = 'normal episode timeout or administrative map boundary, with current measured contact on a virtual top and no fall'
    result['course_integrity']['stage2_qualification'] = alignment['success']
    result['curriculum']['qualification'] = 'virtual-mask-supported passage AND directed travel >=3 m AND contacting sole-point error <=5%; speed tracking is diagnostic only'
    result['recovery']['pass_rule'] = 'virtual-mask-supported passage, directed travel >=3 m, moving command >=1 s, single support >=10%, finite data; speed deviation diagnostic only'
    result['adaptations'].append('Explicit physical-to-virtual continuation on flat collision: preserve learned mean/critics and terrain levels, bound action std with selective optimizer-moment clearing and auxiliary mean-overflow loss, clear cross-domain success streaks; no claim of physical pillar mastery')
    return result


def virtual_course_contract_matches(parent, evaluated, key, requested=False):
    """Only explicit v16 -> v17 may replace real pillars with flat collision."""
    if not (requested and parent.get('version') == 16 and evaluated.get('version') == 17
            and parent.get('stage') == evaluated.get('stage') == 2):
        return parent.get(key) == evaluated.get(key) and (key in parent) == (key in evaluated)
    try:
        expected = virtual_course_contract_amendment(parent)
    except (KeyError, TypeError, ValueError):
        return False
    return expected.get(key) == evaluated.get(key) and (key in expected) == (key in evaluated)


def virtual_sampler_contract_spec():
    return {'version': 1, 'easy_fraction': .2, 'challenge_fraction': .2,
            'challenge_levels': [7, 8, 9, 10], 'target_fraction': .6,
            'challenge_progression': 'no target/streak promotion or demotion'}


def virtual_frontier_sampling_for_contract(parent):
    """Reject unknown sampling semantics instead of silently inheriting them."""
    if 'virtual_curriculum_sampling' not in parent:
        return False
    if (parent.get('version') != 17 or parent.get('stage') != 2
            or parent['virtual_curriculum_sampling'] != virtual_sampler_contract_spec()):
        raise ValueError('Invalid virtual frontier sampler contract')
    return True


def virtual_sampler_contract_matches(parent, evaluated, key, requested=False):
    transfer = (requested and parent.get('version') == evaluated.get('version') == 17
                and parent.get('stage') == evaluated.get('stage') == 2
                and 'virtual_curriculum_sampling' not in parent)
    if transfer and key == 'virtual_curriculum_sampling':
        return evaluated.get(key) == virtual_sampler_contract_spec()
    return parent.get(key) == evaluated.get(key) and (key in parent) == (key in evaluated)


def build_config(stage=1, num_stones=2, num_envs=2048, seed=42, device="cuda:0",
                 perception="ideal_heightmap", scan_rpy=None, *, paper_aligned=False,
                 initial_level=0, curriculum=True, repo_aligned=False, upstream_aligned=False,
                 unitree_motion_aligned=False, unitree_scanner_profile='beamdojo_paper',
                 course_integrity=False, course_tracking=False, physical_aligned=False,
                 continuous_physical=False, recovery=False, recovery_max_level=2,
                 recovery_yaw_half_range=.2617993877991494,
                 recovery_neutral_unattempted=False, strict_foothold=False, virtual_handoff=False,
                 bound_joint_actions=False, course_schedule='legacy_9', recovery_speed_gate=True,
                 virtual_course=False, virtual_frontier_sampling=False, support_experiment=None):
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from legged_lab.tasks.locomotion.beamdojo.env_cfg import build_env_cfg
    from legged_lab.tasks.locomotion.unitree_official.agents import OfficialPPORunnerCfg
    if not importlib.metadata.version("isaacsim").startswith("6."):
        raise RuntimeError("BeamDojo port requires the existing Isaac Sim 6 runtime")
    if perception not in ("ideal_heightmap", "ray_lidar"):
        raise ValueError("Unsupported perception source")
    paper_aligned = paper_aligned or repo_aligned or upstream_aligned
    cfg = build_env_cfg(stage, num_stones, num_envs, device, seed,
                        paper_aligned=paper_aligned, repo_aligned=repo_aligned, initial_level=initial_level,
                        curriculum=curriculum, upstream_aligned=upstream_aligned,
                        unitree_motion_aligned=unitree_motion_aligned)
    cfg.scene.robot.spawn.usd_path = str(ROBOT_USD)
    schedule = course_schedule_spec(course_schedule)
    if not 0 <= initial_level <= schedule['max_level']:
        raise ValueError('Initial difficulty exceeds this course schedule')
    if course_schedule != 'legacy_9' and not (virtual_handoff and bound_joint_actions and recovery and stage == 2):
        raise ValueError('Smooth course requires execution-bounded physical virtual-handoff recovery')
    if type(recovery_speed_gate) is not bool or (course_schedule == 'legacy_9' and not recovery_speed_gate):
        raise ValueError('Speed-pass amendment requires the explicit smooth-course profile')
    # Configure the inherited physical task before rebuilding its new rows;
    # the smooth schedule is intentionally invalid for the old Stage-2 layout.
    cfg.beamdojo_course_schedule = 'legacy_9'
    cfg.beamdojo_recovery_speed_gate = recovery_speed_gate
    if unitree_scanner_profile not in ('beamdojo_paper', 'unitree_g1', 'legacy_upstream'):
        raise ValueError('Unknown Unitree scanner profile')
    cfg.beamdojo_scanner_profile = unitree_scanner_profile
    if unitree_motion_aligned and unitree_scanner_profile == 'beamdojo_paper':
        cfg.commands.base_velocity.limit_ranges.lin_vel_x = (-1.5, 1.5)
    if course_integrity:
        if unitree_scanner_profile != 'beamdojo_paper':
            raise ValueError('Course-integrity migration requires the BeamDojo 15x15 map')
        cfg.configure_course_integrity()
    if course_tracking:
        cfg.configure_course_tracking()
    if physical_aligned:
        cfg.configure_physical_alignment()
    if continuous_physical:
        cfg.configure_continuous_physical()
    if recovery:
        cfg.configure_recovery()
        cfg.recovery_max_level=recovery_max_level
        cfg.recovery_yaw_half_range=recovery_yaw_half_range
        cfg.recovery_neutral_unattempted=recovery_neutral_unattempted
    if strict_foothold:
        if not (recovery and physical_aligned and stage == 2 and unitree_motion_aligned):
            raise ValueError('Strict foothold requires the continuous Unitree-motion physical recovery task')
        cfg.beamdojo_strict_foothold = True
        cfg.beamdojo_allowed_unsupported_fraction = 0.
    if virtual_handoff:
        cfg.configure_virtual_handoff()
    if bound_joint_actions:
        cfg.configure_action_bounds()
    if course_schedule != 'legacy_9':
        cfg.beamdojo_course_schedule = course_schedule
        cfg.configure_course()
    if virtual_course:
        cfg.configure_virtual_course()
    if virtual_frontier_sampling:
        if not virtual_course:
            raise ValueError('Frontier sampling requires the v17 virtual course')
        cfg.recovery_challenge_fraction = .2
        cfg.recovery_challenge_levels = (7, 8, 9, 10)
    if support_experiment is not None:
        from beamdojo_support_experiment import apply_config
        apply_config(cfg, support_experiment)
    cfg.observation_source = perception
    if perception == "ray_lidar":
        if upstream_aligned or unitree_motion_aligned:
            raise ValueError("Upstream 225-height input has no raw-LiDAR migration contract yet")
        if num_envs > 64:
            raise ValueError("Raw-point LiDAR cache is currently limited to 64 coverage environments")
        from legged_lab.tasks.locomotion.beamdojo.lidar import make_lidar_cfg, lidar_observation
        cfg.scene.beamdojo_lidar = make_lidar_cfg(
            scan_rpy, stage=stage, prim_path="{ENV_REGEX_NS}/Robot/torso_link")
        cfg.observations.terrain.ideal_heightmap.func = lidar_observation
    agent = OfficialPPORunnerCfg()
    agent.seed, agent.device = seed, device
    agent.experiment_name = "g1_beamdojo_fullbody_sim6"
    agent.algorithm.class_name = "legged_lab.rsl_rl.beamdojo:DualCriticPPO"
    agent.obs_groups = {"actor": ["policy", "terrain"], "critic": ["critic", "terrain"]}
    if course_tracking:
        agent.obs_groups['actor'].append('course_tracking')
        agent.obs_groups['critic'].append('course_tracking')
    agent.actor.obs_normalization = not unitree_motion_aligned
    agent.critic.obs_normalization = not unitree_motion_aligned
    agent.actor.class_name = "legged_lab.rsl_rl.beamdojo:BeamDojoMLPModel"
    agent.critic.class_name = "legged_lab.rsl_rl.beamdojo:BeamDojoMLPModel"
    if paper_aligned and not upstream_aligned and not unitree_motion_aligned:
        # Table X gives 100 control steps and [512, 216, 128], not 256.
        agent.num_steps_per_env = 100
        agent.actor.hidden_dims = [512, 216, 128]
        agent.critic.hidden_dims = [512, 216, 128]
        # Minibatch count/std/LR are not specified in the paper. Smaller
        # minibatches keep 4096 full-body environments within the 16 GB GPU.
        agent.algorithm.num_mini_batches = 8
        agent.actor.distribution_cfg.init_std = .5
    if upstream_aligned:
        from beamdojo_upstream_profile import configure_agent
        configure_agent(agent)
        agent.actor.class_name = "legged_lab.rsl_rl.beamdojo.normalization:UpstreamMLPModel"
        agent.critic.class_name = "legged_lab.rsl_rl.beamdojo.normalization:UpstreamMLPModel"
    agent.resume = False
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version("rsl-rl-lib"))
    return cfg, agent


def reward_parameter_contract(value):
    """Stable semantic parameters; resolved runtime IDs are intentionally omitted."""
    from isaaclab.managers import SceneEntityCfg
    if isinstance(value, SceneEntityCfg):
        return {key: reward_parameter_contract(getattr(value, key))
                for key in ('name', 'joint_names', 'body_names', 'preserve_order')}
    if isinstance(value, dict):
        return {k: reward_parameter_contract(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [reward_parameter_contract(v) for v in value]
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    raise TypeError(f'Unsupported reward contract parameter {type(value)}')


def contract(cfg, observations, actions, scan_rpy=None, joint_names=None):
    """Checkpoint compatibility and truthful perception provenance."""
    result = {
        "version": 1, "task": TASK, "robot": "local G1 29DOF",
        "action_dimension": actions, "stage": cfg.beamdojo_stage,
        "joint_names": joint_names, "asset_sha256": hashlib.sha256(ROBOT_USD.read_bytes()).hexdigest(),
        "num_stones": cfg.beamdojo_num_stones, "observation_source": cfg.observation_source,
        "observation_dimensions": {k: list(v.shape[1:]) for k, v in observations.items()},
        "map": {"size": 15, "resolution_m": .1, "update_hz": 10},
        "gait_reference": {"type": "continuous_command_aware", "same_foot_stride_m": .57,
                           "cycle_period_limits_s": [1., 1.9]},
        "foot_safety_projection": {"sample_count": 35, "x_bounds_m": [-.055, .145],
                                   "y_bounds_m": [-.035, .035], "local_z_m": -.035,
                                   "type": "full_collision_aabb_safety_proxy"},
        "policy_hz": 50, "simulation_hz": 200,
        "ideal_map_noise": {"enabled": cfg.beamdojo_map_noise, "height_bias_m": .01,
                            "point_noise_m": .01, "gradient_endpoint_m": .02,
                            "yaw_episode_rad": .02, "repeat_probability": .05,
                            "extension_probability": .05, "paper_exact_parameters": False},
        "scan_rpy_assumption": scan_rpy,
        "scan_calibrated": False, "physical_lidar_verified": False,
        "reference_repository_commit": REFERENCE_COMMIT,
        "adaptations": ["29 learned actions", "reward-shaped arms; no AMP",
                        "fixed staggered course", "validity and map age observations",
                        "stable finish hold", "native Isaac Sim 6 / RSL 5.4.1"],
    }
    if getattr(cfg, "beamdojo_paper_aligned", False):
        result.update({
            "version": 2, "training_profile": "beamdojo_paper_g1_fullbody",
            "gait_reference": {"type": "emergent_no_phase_or_forced_foot_height"},
            "terrain_family": "stones_everywhere" if cfg.beamdojo_stage == 1 else "stepping_stones",
            "curriculum": {"enabled": cfg.beamdojo_curriculum,
                           "initial_level": cfg.beamdojo_initial_level,
                           "max_level": 8, "consecutive_passes_to_promote": 3,
                           "demote_on_failure": False, "stone_counts_stage2": [2, 4, 6]},
            "ideal_map_noise": {"enabled": cfg.beamdojo_map_noise, "height_bias_m": .03,
                                "point_noise_m": .03, "gradient_endpoint_m": .03,
                                "yaw_episode_rad": .2, "repeat_probability": .2,
                                "extension_probability": .6, "paper_exact_parameters": True},
            "ppo": {"num_steps_per_env": 100, "hidden_dims": [512, 216, 128],
                    "num_epochs": 5, "num_mini_batches": 8, "initial_std": .5,
                    "initial_learning_rate": .001, "learning_rate_schedule": "adaptive",
                    "advantage_weights": [1., .25],
                    "unspecified_in_paper": ["num_mini_batches", "initial_std", "learning_rate", "schedule"]},
            "adaptations": ["29 learned actions and reward-shaped natural arms; no AMP",
                            "G1 local USD / official PD / base height .78 m",
                            "2 -> 4 -> 6 stone-count curriculum in Stage 2",
                            "validity and map age observations", "stable finish hold",
                            "numerically bounded feet-distance and joint-power terms",
                            "native Isaac Sim 6 / RSL 5.4.1; physics 200 Hz",
                            "user extension of total training budget to 30000 updates"],
        })
    if getattr(cfg, "beamdojo_repo_aligned", False):
        from legged_lab.tasks.locomotion.beamdojo import repo_mdp
        result.update({
            "version": 3,
            "training_profile": "beamdojo_repository_rewards_g1_fullbody",
            "alignment_scope": "23 active repository rewards plus phase observations; G1 port, not an exact entire-pipeline reproduction",
            "g1_control_reference": {"repository": "https://github.com/unitreerobotics/unitree_rl_lab",
                                     "commit": UNITREE_REFERENCE_COMMIT,
                                     "verified_identical_fields": ["init_state", "actuators", "joint_sdk_names"],
                                     "action_scale": .25, "default_joint_offset": True,
                                     "policy_hz": 50, "physics_hz": 200},
            "reward_anatomy": {"base_height_target_m": repo_mdp.BASE_HEIGHT_TARGET,
                               "sole_offset_m": repo_mdp.SOLE_OFFSET,
                               "foot_distance_bounds_m": [repo_mdp.MIN_FEET_DISTANCE, repo_mdp.MAX_FEET_DISTANCE],
                               "knee_distance_bounds_m": [repo_mdp.MIN_KNEE_DISTANCE, repo_mdp.MAX_KNEE_DISTANCE],
                               "max_contact_force_N": repo_mdp.MAX_CONTACT_FORCE,
                               "support_contact": "world vertical force > 5 N",
                               "leg_reference_names": list(repo_mdp.LEG_NAMES)},
            "gait_reference": {"type": "repository_periodic_joint_and_contact_reference",
                               "cycle_time_s": .64, "swing_joint_amplitude_rad": .17,
                               "target_foot_height_m": .06, "double_support_sin_threshold": .1,
                               "g1_swing_pitch_signs": [-1, 1, -1],
                               "reference_actions_added_to_actor": False,
                               "zero_command": "double support; no foot lift or air-time bonus"},
            "reward_weights": {name: float(term.weight) for name, term in vars(cfg.rewards).items()
                               if term is not None and hasattr(term, 'weight') and term.weight != 0},
            "reward_functions": {name: f'{term.func.__module__}.{term.func.__name__}'
                                 for name, term in vars(cfg.rewards).items()
                                 if term is not None and hasattr(term, 'weight') and term.weight != 0},
            "adaptations": ["G1 29 learned actions; 12 leg joints follow a reward reference",
                            "G1 joint-axis signs and valid default pose; .78 m base target and rotated .035 m sole offset",
                            "separate natural-arm and arm-excess rewards; no AMP and no reference action injection",
                            "zero-command no-lift gating requested by user",
                            "partial-reset history isolation and current-step reference timing fixes",
                            "paper-profile terrain, commands, PPO, 5-frame histories and terminations retained",
                            "validity and map age observations; stable finish hold",
                            "native Isaac Sim 6 / RSL 5.4.1; policy 50 Hz / physics 200 Hz",
                            "4096 environments / 30000-update total budget"],
        })
    if getattr(cfg, "beamdojo_upstream_aligned", False):
        from beamdojo_upstream_profile import PPO, PROFILE
        result.update({
            "version": 4,
            "training_profile": PROFILE,
            "alignment_scope": "Pinned humanoid-gym task/reward/PPO parameters with Unitree G1 mechanics and official upper-body posture terms",
            "map": {"size":15, "resolution_m":.05, "dimension":225, "update_hz":50,
                    "normalization":"clamp(root_z - 0.5 - height, -1, 1) * 5"},
            "ppo": PPO.copy(),
            "observations": {"actor_history":15, "critic_history":3,
                             "clip":18., "extra_validity_or_age":False},
            "commands": {"lin_vel_x":[-.3,.6], "lin_vel_y":[-.3,.3],
                         "heading":[-3.14,3.14], "heading_gain":.5,
                         "effective_yaw_clip":[-1.,1.], "resampling_s":8.},
            "episode_length_s":96.,
            "g1_failure_conditions": {
                "revision": "2026-10-03_official_g1_fall_detection",
                "reference": "unitree_rl_lab/g1/29dof/velocity_env_cfg.py",
                "minimum_root_height_m": .2,
                "maximum_body_tilt_rad": .8,
                "pelvis_contact_force_N": 1.,
                "time_limit_is_failure": False,
            },
            "mass_randomization_reference": {"repository":"unitreerobotics/unitree_rl_lab",
                                             "body":"torso_link", "added_kg":[-1.,3.]},
            "terrain_port_details": ["Source 2 cm support rasters verified for all 9 levels in both stages",
                                     "Native Sim6 boxes instead of IsaacGym heightfield triangulation",
                                     "Same-level terrain columns share deterministic source layout",
                                     "Queries clamp at each terrain tile boundary"],
            "memory_implementation":"deterministic MLP activation recomputation; unchanged full source minibatches and optimizer steps",
            "curriculum": {"enabled":cfg.beamdojo_curriculum,
                           "initial_training_levels":[0,4], "max_level":8,
                           "consecutive_passes_to_promote":3,
                           "demote_on_failure":False, "overflow":"random level 0..8"},
            "terrain_family":"stones_everywhere_stage1" if cfg.beamdojo_stage==1 else "stepping_stones",
            "upper_body_reference": {"repository":"unitreerobotics/unitree_rl_lab",
                                     "commit":UNITREE_REFERENCE_COMMIT,
                                     "joint_deviation_arms":-.1, "joint_deviation_waists":-1.},
            "adaptations": ["G1 29 learned actions, official PD/limits/default posture/action scale",
                            "G1 named leg axes, .78 m base target, measured foot geometry",
                            "Unitree official arm and waist posture rewards; no custom swing rewards",
                            "Unitree official G1 height/tilt failure terms restored alongside pelvis contact",
                            "User-requested zero-command no-lift gating",
                            "Native Isaac Sim 6 / RSL 5.4.1, official 50 Hz control / 200 Hz physics",
                            "Numerical distribution safeguards and native timeout bootstrap convention",
                            "User-authorized 4096 environments and total 30000-update budget"],
        })
        # The new input has none of the old map's validity/age or noise contract.
        result.pop('ideal_map_noise', None)
    if getattr(cfg, 'beamdojo_unitree_motion_aligned', False):
        from beamdojo_unitree_motion_profile import PROFILE, PPO
        from legged_lab.tasks.locomotion.unitree_official.env_cfg import CommandsCfg
        cmd = CommandsCfg().base_velocity
        if cfg.beamdojo_scanner_profile == 'beamdojo_paper':
            cmd.limit_ranges.lin_vel_x = (-1.5, 1.5)
        active_rewards = {name: term for name, term in vars(cfg.rewards).items()
                          if term is not None and hasattr(term, 'weight') and term.weight != 0}
        result.update({
            'version': 5, 'training_profile': PROFILE,
            'alignment_scope': 'Complete pinned Unitree G1 motion rewards/control/commands/command curriculum and PPO, plus BeamDojo terrain/map/foothold and dual critic',
            'map': {'size': 15, 'resolution_m': .05, 'dimension': 225, 'update_hz': 50,
                    'normalization': 'clamp(root_z - 0.5 - height, -1, 1) * 5'},
            'ppo': PPO.copy(),
            'observations': {'actor_history': 5, 'critic_history': 5,
                             'proprioception': 'unaltered Unitree policy/critic terms, scaling and noise',
                             'extra_validity_or_age': False},
            'gait_reference': {'type': 'official_contact_gait', 'period_s': .8,
                               'phase_offset': [0., .5], 'contact_threshold': .55,
                               'periodic_joint_reference': False, 'amp': False},
            'commands': {
                'initial_ranges': {key: list(getattr(cmd.ranges, key)) for key in ('lin_vel_x','lin_vel_y','ang_vel_z')},
                'limit_ranges': {key: list(getattr(cmd.limit_ranges, key)) for key in ('lin_vel_x','lin_vel_y','ang_vel_z')},
                'resampling_s': 10., 'heading_command': False, 'standing_fraction': .02,
                'curriculum': 'official lin_vel_cmd_levels: expand x/y by .1 at >80% normalized linear tracking reward; yaw curriculum disabled'},
            'episode_length_s': 20.,
            'g1_failure_conditions': {'minimum_root_height_m': .2, 'maximum_body_tilt_rad': .8,
                                      'time_limit_is_failure': False},
            'g1_control_reference': {'repository': 'https://github.com/unitreerobotics/unitree_rl_lab',
                'commit': UNITREE_REFERENCE_COMMIT, 'verified_identical_fields': ['init_state','actuators','joint_sdk_names'],
                'action_scale': .25, 'default_joint_offset': True, 'policy_hz': 50, 'physics_hz': 200,
                'extra_action_delay_or_noise': False},
            'reward_anatomy': {'base_height_target_m': .78, 'sole_offset_m': .035,
                               'support_contact': 'world vertical force > 5 N'},
            'reward_weights': {name: float(term.weight) for name, term in active_rewards.items()},
            'reward_functions': {name: f'{term.func.__module__}.{term.func.__name__}' for name, term in active_rewards.items()},
            'reward_parameters': {name: reward_parameter_contract(term.params) for name, term in active_rewards.items()},
            'terrain_family': 'stones_everywhere_stage1' if cfg.beamdojo_stage == 1 else 'stepping_stones',
            'curriculum': {'terrain_owner': 'BeamDojo', 'enabled': cfg.beamdojo_curriculum,
                           'initial_training_levels': [0,4], 'max_level': 8,
                           'consecutive_passes_to_promote': 3, 'demote_on_failure': False},
            'adaptations': ['Native Isaac Sim 6 / RSL 5.4.1 API port',
                            'Current-frame 225-height map appended to official proprioception',
                            'BeamDojo virtual/physical terrain and three-success terrain progression',
                            'One sparse foothold penalty; dense/sparse normalized advantages 1.0/0.25',
                            'Stage 2 reset placed on start platform at x=-3.5 rather than center void',
                            '4096 environments; stage budgets recorded by the training pipeline'],
        })
        result.pop('ideal_map_noise', None)
        if cfg.beamdojo_scanner_profile in ('unitree_g1', 'beamdojo_paper'):
            from legged_lab.tasks.locomotion.beamdojo.upstream_perception import scanner_spec
            spec = scanner_spec(cfg)
            result['version'] = 6
            result['map'] = {
                'profile': spec['profile'], 'shape_xy': spec['shape_xy'],
                'size_xy_m': spec['size_xy_m'], 'resolution_m': spec['resolution_m'],
                'dimension': spec['count'], 'ordering': spec['ordering'],
                'frame': spec['frame'], 'alignment': 'yaw', 'offset_xyz_m': [0., 0., 20.],
                'ray_direction': [0., 0., -1.], 'update_hz': 50,
                'normalization': 'clamp(root_z - 0.5 - height, -1, 1) * 5',
                'surface_source': 'BeamDojo virtual/physical heightfield, three-neighbour minimum',
                'scope': 'Unitree scanner geometry; BeamDojo values, not physical LiDAR',
            }
            result['adaptations'][1] = 'Current-frame official Unitree 187-point scanner footprint with BeamDojo elevation values'
            if spec['profile'] == 'beamdojo_paper':
                result['version'] = 7
                result['map']['offset_xyz_m'] = [0., 0., 0.]
                result['map']['scope'] = 'BeamDojo paper spatial footprint; source normalization and 50 Hz training sampling retained, not physical LiDAR'
                result['map']['spatial_reference'] = 'https://arxiv.org/html/2502.10363v3#S4.SS4.SSS1'
                result['adaptations'][1] = 'Base-centered BeamDojo paper 15x15 map, 0.1 m spacing, 225 heights'
                result['adaptations'].append('User-requested longitudinal command curriculum limits [-1.5, 1.5] m/s; initial ranges and other limits unchanged')
    if getattr(cfg, 'beamdojo_course_integrity', False):
        from legged_lab.tasks.locomotion.beamdojo import upstream_terrain as terrain
        result['version'] = 8
        result['course_integrity'] = {
            'version': 1,
            'stage1_map_size_m': cfg.beamdojo_stage1_map_size,
            'stage2_course_length_m': cfg.beamdojo_stage2_course_length,
            'stage2_course_width_m': terrain.SIZE,
            'out_of_bounds': 'unsupported void, height origin_z - 1 m',
            'boundary_truncation': 'administrative timeout with bootstrap; hard failures take priority',
            'stage1_boundary_abs_xy_m': cfg.beamdojo_stage1_map_size / 2 - .75,
            'stage2_boundary_abs_xy_m': [cfg.beamdojo_stage2_course_length / 2 - .1, terrain.SIZE / 2 - .1],
            'minimum_supported_travel_m': terrain.COURSE_MIN_DISTANCE,
            'maximum_contact_sample_error': terrain.COURSE_MAX_ERROR,
            'minimum_contact_foot_seconds': terrain.COURSE_MIN_CONTACT_SECONDS,
            'finish_hold_s': terrain.COURSE_STAND_SECONDS,
            'finish_linear_speed_max_m_s': terrain.COURSE_MAX_LINEAR_SPEED,
            'finish_angular_speed_max_rad_s': terrain.COURSE_MAX_ANGULAR_SPEED,
            'stage2_command_task_override': 'official sampling and curriculum; direction-matched start platform; zero command after entry into target platform interior',
            'stage1_qualification': 'At episode timeout or boundary reset; >=3 m supported travel AND >=3 m net displacement; contact error <=5%; no fall',
            'route_extension': '32x32 m virtual map and 16 m physical route; unchanged 15x15 local actor map',
        }
        result['curriculum']['qualification'] = 'supported traversal and contact accuracy; stage2 actual end-platform stable finish'
        result['adaptations'].append('Explicit course-integrity revision: no off-map safe support or radius-only success')
    if getattr(cfg, 'beamdojo_course_tracking', False):
        result['version'] = 9
        result['course_tracking'] = {
            'version': 1, 'actor_group': 'course_tracking', 'dimension': 3,
            'actor_features': ['sin(relative_imu_yaw)', 'cos(relative_imu_yaw)-1', 'straight_command_active'],
            'actor_inputs': ['body gyro', 'projected gravity', 'velocity commands', 'control dt'],
            'absolute_pose_in_actor': False,
            'gyro_yaw_rate': '-(omega_y*gravity_y + omega_z*gravity_z)/(gravity_y^2+gravity_z^2); denominator >=0.25',
            'gyro_noise_std_rad_s': cfg.tracking_gyro_noise_std,
            'gyro_residual_bias_bound_rad_s': cfg.tracking_gyro_bias,
            'straight_mask': '|vx|>=0.1, |vy|<=0.0001, |yaw_rate_command|<=0.0001',
            'straight_command_fraction': cfg.tracking_straight_fraction,
            'command_sampling_override': 'Official ranges/standing/curriculum retained; 40% of moving resamples set lateral and yaw commands to zero',
            'reference_reset': 'episode reset or entry into a new straight segment; straight resampling/speed/sign changes retain reference',
            'reward_only_truth': ['segment-relative yaw', 'segment-relative cross-track distance'],
            'heading_reward': {'weight': -.5, 'scale_rad': .5, 'normalized_square_cap': 4.},
            'lateral_reward': {'weight': -.5, 'scale_m': .3, 'normalized_square_cap': 4.},
            'deployment_scope': 'Short-term gyro heading hold; requires equivalent segment/reset logic and filtered bias-calibrated IMU; no exact positional line recovery without odometry',
        }
        result['adaptations'].append('User-authorized IMU-relative heading feedback and simulation-only heading/cross-track reward shaping')
    if getattr(cfg, 'beamdojo_virtual_handoff', False):
        # Capture the virtual contract before physical/recovery metadata overwrites it.
        smooth = getattr(cfg, 'beamdojo_course_schedule', 'legacy_9') == 'smooth_top_v1'
        legacy_cap = (8 if cfg.recovery_max_level == 10 else min(7, cfg.recovery_max_level)) if smooth else cfg.recovery_max_level
        result = virtual_handoff_contract_amendment(result, legacy_cap,
            cfg.recovery_yaw_half_range, cfg.recovery_neutral_unattempted)
        if getattr(cfg, 'beamdojo_bound_joint_actions', False):
            result = action_bounds_contract_amendment(result)
        if smooth:
            if getattr(cfg, 'beamdojo_recovery_speed_gate', True):
                raise ValueError('v16 explicitly disables speed-deviation pass gates')
            result = smooth_course_contract_amendment(result, cfg.recovery_max_level)
        if getattr(cfg, 'beamdojo_virtual_course', False):
            result = virtual_course_contract_amendment(result)
            if getattr(cfg, 'recovery_challenge_fraction', 0.):
                if (cfg.recovery_challenge_fraction != .2
                        or tuple(cfg.recovery_challenge_levels) != (7, 8, 9, 10)
                        or cfg.recovery_easy_replay_fraction != .2):
                    raise ValueError('Virtual sampler differs from the declared 20/20/60 scheme')
                result['virtual_curriculum_sampling'] = virtual_sampler_contract_spec()
        if getattr(cfg, 'beamdojo_support_experiment', None) is not None:
            from beamdojo_support_experiment import contract_amendment
            result = contract_amendment(result, cfg.beamdojo_support_experiment['arm'])
        return result
    if getattr(cfg, 'beamdojo_physical_aligned', False):
        result['version'] = 10
        result['map']['surface_source'] = 'Exact 2 cm raster cells shared by virtual map and physical collision mesh'
        result['terrain_family'] = 'stones_everywhere_shared_virtual_physical'
        result['physical_alignment'] = {
            'version': 1, 'map_size_m': [cfg.beamdojo_stage1_map_size] * 2,
            'layout_source_stage': 1, 'spawn': 'original central platform',
            'physical_initial_levels': [0, 2], 'min_supported_travel_m': 3.,
            'min_signed_progress_m': 3., 'min_valid_contact_seconds': 1.,
            'partial_sole_support_allowed': True,
            'reward_only_unsupported_deadband': cfg.beamdojo_allowed_unsupported_fraction,
            'success': 'actual top-contact-supported traversal and signed net progress >=3 m, no fall; partial soles accepted',
            'heightmap_surface': 'single exact raster cell, matching colliders',
            'no_absolute_pose_in_actor': True,
        }
        result['course_integrity'] = {**result['course_integrity'],
            'stage2_course_length_m': cfg.beamdojo_stage1_map_size,
            'stage2_course_width_m': cfg.beamdojo_stage1_map_size,
            'stage2_boundary_abs_xy_m': [cfg.beamdojo_stage1_map_size / 2 - .1] * 2,
            'stage2_command_task_override': 'central spawn for either command direction; progress goal without end-platform latch',
            'route_extension': 'same 32x32 m raster for both stages',
            'stage2_qualification': result['physical_alignment']['success']}
        result['curriculum']['qualification'] = result['physical_alignment']['success']
        result['curriculum']['initial_training_levels'] = [0, 2]
        result['adaptations'].append('User-authorized early physical handoff, same virtual geometry, partial-sole stable traversal')
    if getattr(cfg, 'beamdojo_physical_continuous', False):
        result['version'] = 11
        result['physical_alignment'].update({
            'version': 2, 'continuous_passage': True,
            'reset_on_distance_milestone': False,
            'spawn_yaw_range_rad': [-3.141592653589793, 3.141592653589793],
            'min_signed_progress_m': None, 'min_net_displacement_m': 3.,
            'qualification_timing': 'normal episode timeout or administrative map boundary, with current real top contact and no fall',
            'success': 'At normal timeout or administrative boundary: actual supported travel >=3 m, XY net displacement >=3 m in any direction, >=1 valid foot-second, current real top contact and no fall; milestone does not terminate',
        })
        result['course_integrity']['stage2_qualification'] = result['physical_alignment']['success']
        result['curriculum']['qualification'] = result['physical_alignment']['success']
        result['adaptations'].append('User-authorized continuous physical passage and random initial yaw; body-frame commands retained')
    if getattr(cfg,'beamdojo_recovery',False):
        result['version']=12
        result['alignment_scope']='Pinned Unitree G1 control/commands/PPO/numerical reward weights, with opt-in clearance/gait movement gates, stall penalty and qualified recovery curriculum'
        result['recovery']={
            'version':1,'training_max_level':cfg.recovery_max_level,
            'training_yaw_half_range_rad':cfg.recovery_yaw_half_range,
            'evaluation_yaw_range_rad':[-3.141592653589793,3.141592653589793],
            'easy_replay_fraction':cfg.recovery_easy_replay_fraction,
            'moving_command_threshold_m_s':.2,'min_moving_seconds':1.,
            'min_directed_progress_ratio':.6,'max_xy_velocity_rmse_m_s':.4,
            'min_single_support_fraction':.1,
            'lift_gait_gate':'smoothstep of command-aligned speed ratio; full credit at ratio >=0.5',
            'stall_weight':-1.,'training_only_curriculum':True,
        }
        result['curriculum'].update({'initial_training_levels':[0,min(2,cfg.recovery_max_level)],
            'max_level':cfg.recovery_max_level,'demote_on_failure':True,
            'qualification':'actual physical passage AND commanded motion; easy replay cannot promote unattempted hard levels'})
        if getattr(cfg,'recovery_neutral_unattempted',False):
            result['recovery']['neutral_unattempted']=True
            result['adaptations'].append('Neutral curriculum outcome without sufficient translation opportunity (1 s and 3 m commanded travel); actual failure still demotes')
        result['adaptations'].append('Explicit recovery of continuous task: reset old difficulty, retain actor/critics/Adam and train limited yaw before full-yaw deployment evaluation')
    if getattr(cfg, 'beamdojo_strict_foothold', False):
        result['version'] = 13
        result['physical_alignment'].update({
            'reward_only_unsupported_deadband': 0., 'strict_foothold': True,
            'training_contact_error_fraction_max': .05,
            'strict_reward_rule': 'negative count of contacting sole samples over virtual gaps; no overhang allowance and no sole-height reward test'})
        result['curriculum']['qualification'] += '; contacting sole-point error <=5% as on virtual terrain'
        result['adaptations'].append('User-authorized real-pillar foothold penalty identical to virtual point counting, zero reward overhang allowance; virtual 5% episode contact-error qualification')
    return result
