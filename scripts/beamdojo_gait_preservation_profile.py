"""Preserve all official motion terms; named yaw/parking additions only."""
from copy import deepcopy
import sys

ADDITIONS = ('preserve_heading', 'preserve_yaw_rate', 'parking_stability', 'parking_pose')


def active_variant(completed):
    return 0 if completed < 1000 else (3 if completed < 2000 else 7)


def build(parent, num_envs, seed, device, phase='generalize', initial_level=8, flat_fraction=.5):
    from isaaclab.managers import RewardTermCfg
    import torch
    from beamdojo_square_extended_profile import build as previous_build
    from legged_lab.tasks.locomotion.beamdojo import course_tracking
    import beamdojo_gait_preservation_rewards as rewards
    import beamdojo_square_mixed_runtime as runtime
    cfg, agent = previous_build(parent, num_envs, seed, device, 'generalize', initial_level, flat_fraction)
    # Rebind references captured by the geometry/environment build, before any
    # native manager is constructed. Historical profiles remain unchanged.
    import beamdojo_gait_preservation_runtime as corrected
    names = ('ensure_state', 'goal_start', '_geometry', 'parking_command', 'update_course',
             'time_out', 'before_reset', 'course_metrics', 'snapshot', 'restore')
    replacements = {getattr(runtime, name): getattr(corrected, name) for name in names}
    corrected.install(runtime)
    for obj in (cfg.terminations,):
        for value in vars(obj).values():
            if hasattr(value, 'func') and value.func in replacements:
                value.func = replacements[value.func]
    for module in list(sys.modules.values()):
        if module is None or '.beamdojo' not in getattr(module, '__name__', ''):
            continue
        for key, value in list(vars(module).items()):
            try:
                replacement = replacements.get(value)
            except TypeError:
                continue
            if replacement is not None:
                setattr(module, key, replacement)
    cfg.rewards.preserve_heading = RewardTermCfg(func=course_tracking.heading_error_l2,
        weight=-.20, params={'scale': .35})
    cfg.rewards.preserve_yaw_rate = RewardTermCfg(func=rewards.straight_yaw_rate_l2,
        weight=-.10, params={'scale': .35})
    cfg.rewards.parking_stability = RewardTermCfg(func=rewards.parking_stability, weight=.40)
    cfg.rewards.parking_pose = RewardTermCfg(func=rewards.parking_pose_l2, weight=-.05)
    cfg.square_mixed_active_variant = 0
    cfg.gait_preservation_regular_replay = .5
    native_initialize = runtime.initialize_geometry
    def initialize(env):
        strength = env.cfg.square_mixed_active_variant
        try:
            # The archived builder's first initializer samples randint(1,k+1).
            # Its temporary sample is overwritten by the regular-first sampler.
            env.cfg.square_mixed_active_variant = max(1, strength)
            native_initialize(env)
        finally:
            env.cfg.square_mixed_active_variant = strength
        if getattr(env, '_gait_preservation_sampler', False):
            return
        env._gait_preservation_sampler = True
        def choose(ids):
            family = env.square_family[ids]
            strength = int(env.cfg.square_mixed_active_variant)
            if strength == 0:
                variants = torch.zeros_like(family)
            else:
                variants = torch.randint(1, strength+1, family.shape, device=env.device)
                replay = torch.rand(len(ids), device=env.device) < env.cfg.gait_preservation_regular_replay
                variants[replay] = 0
            env.square_variant[ids] = variants
            columns = family*32 + variants*4 + torch.randint(0, 4, family.shape, device=env.device)
            flat = env.flat_rehearsal_mask[ids]
            columns[flat] = 64 + ids[flat] % 16
            env.square_variant[ids[flat]] = 0
            env.beamdojo_levels[ids[flat]] = 0
            runtime.tensor(env.scene.terrain.terrain_types)[ids] = columns
        env.square_choose_variants = choose
        choose(torch.arange(env.num_envs, device=env.device))
        env.square_refresh_geometry(env, None)
    runtime.initialize_geometry = initialize
    from legged_lab.tasks.locomotion.beamdojo import upstream_perception
    native_query = upstream_perception.surface_height
    def guarded_query(env, points, **kwargs):
        # Observation-manager shape discovery can call the query directly,
        # bypassing the upstream ensure-state entry point.
        initialize(env)
        return native_query(env, points, **kwargs)
    for module in list(sys.modules.values()):
        if module is None or '.beamdojo' not in getattr(module, '__name__', ''):
            continue
        for key, value in list(vars(module).items()):
            if value is native_initialize:
                setattr(module, key, initialize)
            elif value is native_query:
                setattr(module, key, guarded_query)
    cfg.gait_preservation = dict(version=1, teacher_iteration=11700,
        frozen_teacher_scope='straight forward vx0.3..1; exclude stop/backward/turn/lateral',
        retention_coefficient=.10, retention_action_scale=.8,
        retention_arm_weight=1.5, regular_replay_fraction=.5,
        block_schedule={'0..999': 0, '1000..1999': 3, '2000+': 7},
        extra_actor_observations=False, motion_rewards='official unchanged plus named yaw/parking additions')
    agent.algorithm.learning_rate = .0001
    agent.algorithm.schedule = 'fixed'
    return cfg, agent


def make_contract(base, cfg, parent):
    from beamdojo_square_extended_profile import make_contract as previous_contract
    from beamdojo_common import reward_parameter_contract
    from beamdojo_gait_preservation_runtime import REVISION
    saved = {name: getattr(cfg.rewards, name) for name in ADDITIONS}
    # Run the existing strict validator over every official term. Explicitly
    # append only these declared additions after that check succeeds.
    try:
        for name in ADDITIONS:
            setattr(cfg.rewards, name, None)
        result = previous_contract(deepcopy(base), cfg, parent)
    finally:
        for name, term in saved.items():
            setattr(cfg.rewards, name, term)
    for name, term in saved.items():
        result['reward_weights'][name] = float(term.weight)
        result['reward_functions'][name] = f'{term.func.__module__}.{term.func.__name__}'
        result['reward_parameters'][name] = reward_parameter_contract(term.params)
    result['gait_preservation'] = deepcopy(cfg.gait_preservation)
    result['gait_preservation']['reward_additions'] = list(ADDITIONS)
    result['square_mixed']['replay_fraction'] = cfg.gait_preservation_regular_replay
    result['square_course']['terminal'] = dict(revision=REVISION, depth_from_ring_entry_m=1.,
        speed='speed-dependent approach, then control-time smoothstep ending at exact zero',
        min_braking_s=.8, max_deceleration_mps2=.8,
        full_support_required='safe entry latch and final success; not each braking tick')
    result['course_tracking']['motion_reward_shaping'] = 'named pure-straight heading/yaw additions; IMU relative heading inputs unchanged'
    result['square_course']['motion_rewards'] = 'official weights/functions/parameters unchanged; four named heading/parking additions'
    result['square_course']['initialization'] = '11700 actor/normalization/dualcritic warm start; new optimizer and soft frozen teacher'
    return result
