"""Paired frozen-policy comparison on a common archived v1 Isaac Sim scene.

The evaluated checkpoint's own normalization is restored. All observations,
control, terrain, initial state and terminal logic come from --reference. No
training updates, no live training/play mutations. Flat course masking is
restored in this evaluation process to avoid stone-only corridor termination.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from datetime import datetime


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--environment', type=Path, help='Use this saved scene for both frozen actors; default reference.')
    parser.add_argument('--variants', type=int, nargs='+', default=[0, 3])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--geometry-seed', type=int,
                        help='Explicit held-out atlas seed; all dimensions/variant definitions stay the same.')
    parser.add_argument('--candidate-no-residual', action='store_true',
                        help='Ablate the learned support-transition residual on the frozen candidate for diagnosis.')
    parser.add_argument('--candidate-residual-group', choices=['all','no_arms','leg_pitch_roll'],default='all',
                        help='Frozen evaluation intervention: mask selected residual joints before native mean tails.')
    parser.add_argument('--arm-audit', action='store_true',
                        help='Record native link dynamics for arm coordination/momentum diagnostics.')
    parser.add_argument('--labels', nargs=2, help='Explicit unique output labels, including same-checkpoint interventions.')
    parser.add_argument('--candidate-arm-scale', type=float, default=1.,
                        help='Evaluation-only shoulder pitch/roll action scale about saved per-case means.')
    parser.add_argument('--arm-center-trace', type=Path,
                        help='Natural rollout trace defining each case mean shoulder action, .5..8s.')
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device='cpu', visualizer=['none'])
    args = parser.parse_args()
    args.reference = args.reference.resolve(strict=True)
    args.candidate = args.candidate.resolve(strict=True)
    args.environment = (args.environment or args.reference).resolve(strict=True)
    labels = args.labels or [args.reference.stem.removeprefix('model_'), args.candidate.stem.removeprefix('model_')]
    if labels[0] == labels[1]:
        parser.error('Checkpoint iteration labels must differ')
    if not 0. <= args.candidate_arm_scale <= 1.:
        parser.error('Arm intervention scale must be in [0,1]')
    if args.candidate_arm_scale != 1. and args.arm_center_trace is None:
        parser.error('Arm intervention requires a natural reference trace')
    args.output.mkdir(parents=True, exist_ok=True)
    def write(name, value):
        path = args.output/name
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        tmp.replace(path)
    def status(**value):
        write('status.json', dict(time=datetime.now().astimezone().isoformat(), **value))
    status(state='initializing')
    app = AppLauncher(args)
    wrapped = None
    code = 0
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from tensordict import TensorDict
        from types import MethodType
        from isaaclab.app.sim_launcher import launch_simulation
        from isaaclab.utils.seed import configure_seed
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from beamdojo_common import TASK, contract
        from play_beamdojo_square_mixed import load_snapshot_modules, validate_contract, contracts_equal

        reference_contract = json.loads((args.reference.parent/'contract.json').read_text())
        parent = json.loads((args.environment.parent/'contract.json').read_text())
        candidate = json.loads((args.candidate.parent/'contract.json').read_text())
        meta = validate_contract(parent)
        shared_keys = ('action_dimension', 'joint_names', 'asset_sha256',
                       'observation_dimensions', 'map', 'policy_hz', 'simulation_hz',
                       'g1_control_reference', 'execution_action_bounds', 'observations')
        for key in shared_keys:
            if not (contracts_equal(parent.get(key), candidate.get(key))
                    and contracts_equal(parent.get(key), reference_contract.get(key))):
                raise ValueError('Policies incompatible in shared field '+key)
        for path, recorded in ((args.reference, reference_contract), (args.candidate, candidate)):
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if not contracts_equal(saved['infos']['beamdojo_contract'], recorded):
                raise ValueError('Embedded contract mismatch '+str(path))
            del saved
        load_snapshot_modules(args.environment)
        import beamdojo_square_mixed_runtime as rt
        if parent.get('gait_preservation'):
            from beamdojo_gait_preservation_profile import build, make_contract
            from beamdojo_gait_preservation_runtime import REVISION
        elif meta['version'] == 2:
            from beamdojo_square_extended_profile import build, make_contract
            from beamdojo_square_extended_runtime import REVISION
        else:
            from beamdojo_square_mixed_profile import build, make_contract
            from beamdojo_square_terminal import install, update_course_impl, REVISION
            from beamdojo_flat_rehearsal import apply_flat_course_state, flat_mask
            install(rt)
            def update_with_flat(env, measurements=None):
                state = update_course_impl(env, measurements, rt)
                if bool(flat_mask(env).any()):
                    apply_flat_course_state(env, state,
                        rt.read_measurements(env) if measurements is None else measurements, rt.tensor)
                return state
            rt.update_course = update_with_flat
        commands = [('forward_0.5', [.5, 0., 0.]), ('forward_1.0', [1., 0., 0.]),
                    ('backward', [-.5, 0., 0.]), ('left', [0., .3, 0.]),
                    ('right', [0., -.3, 0.]), ('turn_left', [0., 0., .2]),
                    ('turn_right', [0., 0., -.2]), ('stand', [0., 0., 0.])]
        cases = []
        for mode, command in commands:
            for replica in range(4):
                cases.append(dict(kind='flat', mode=mode, command=command,
                    family=0, route=0, level=0, variant=0, replica=replica, column=64+replica))
        flat_count = len(cases)
        for family in (0, 1):
            for route in range(4):
                for level in (8, 10):
                    for variant in args.variants:
                        cols = [i for i, (_, f, v) in enumerate(meta['specs']) if f == family and v == variant]
                        if len(cols) < 2:
                            raise ValueError(f'Scene lacks two replicas for variant {variant}')
                        for replica in range(2):
                            cases.append(dict(kind='stone', mode='course', command=[.75, 0., 0.],
                                family=family, route=route, level=level, variant=variant,
                                replica=replica, column=cols[replica]))
        n = len(cases)
        geometry_seed=meta['seed'] if args.geometry_seed is None else args.geometry_seed
        cfg, agent = build(parent, n, geometry_seed, args.device, meta['phase'], flat_fraction=flat_count/n)
        cfg.seed = args.seed
        cfg.episode_length_s = 20.
        cfg.observations.policy.enable_corruption = False
        cfg.beamdojo_map_noise = False
        cfg.tracking_gyro_bias = 0.
        cfg.tracking_gyro_noise_std = 0.
        cfg.events.push_robot = None
        # Identical nominal reset velocities for both actors. Startup randomized
        # material and masses are shared by keeping one physical environment.
        if hasattr(cfg.events, 'reset_robot_joints'):
            cfg.events.reset_robot_joints.params['velocity_range'] = (0., 0.)
        agent.algorithm.num_mini_batches = 1
        configure_seed(args.seed)
        with launch_simulation(cfg, args):
            wrapped = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=agent.clip_actions)
            env = wrapped.unwrapped
            obs = wrapped.get_observations()
            device = env.device
            actual = make_contract(contract(cfg, obs, wrapped.num_actions, None,
                list(env.scene['robot'].joint_names)), cfg, parent)
            for key in shared_keys + ('reward_weights', 'reward_functions', 'reward_parameters'):
                if key in actual and not contracts_equal(actual[key], parent[key]):
                    raise ValueError('Shared scene contract mismatch '+key)
            if args.geometry_seed is None and actual['square_mixed']['support_raster_sha256'] != meta['support_raster_sha256']:
                raise ValueError('Reference atlas mismatch')
            fields = {key: torch.tensor([c[key] for c in cases], device=device, dtype=torch.long)
                      for key in ('family', 'route', 'level', 'variant', 'column')}
            ids = torch.arange(n, device=device)
            flat = ids < flat_count
            fixed = torch.tensor([c['command'] for c in cases], device=device)
            def assign(reset_ids):
                for key, attr in [('family', 'square_family'), ('route', 'square_route'),
                                  ('variant', 'square_variant'), ('level', 'beamdojo_levels')]:
                    getattr(env, attr)[reset_ids] = fields[key][reset_ids]
                rt.tensor(env.scene.terrain.terrain_types)[reset_ids] = fields['column'][reset_ids]
                rt.refresh_angles(env)
                env.square_refresh_geometry(env, reset_ids)
            env.square_choose_variants = assign
            assign(ids)
            term = env.command_manager.get_term('base_velocity')
            native_command = term._update_command
            def fixed_update(self):
                self._square_cruise[:] = .75
                native_command()
                self.vel_command_b[flat] = fixed[flat]
            term._update_command = MethodType(fixed_update, term)
            robot = env.scene['robot']
            if args.arm_audit:
                # PhysX tensors documents get_inertias as COM inertia expressed
                # in the rigid-body-prim frame, NOT the world frame. Rotate by
                # link quaternion below, rather than trusting the generic data
                # interface's world-frame description.
                from isaaclab.utils.math import matrix_from_quat
                body_mass = rt.tensor(robot.data.body_mass).clone()
                body_inertia = rt.tensor(robot.data.body_inertia).reshape(n, -1, 3, 3).clone()
                np.savez_compressed(args.output/'body_properties.npz',
                    body_names=np.array(robot.body_names), mass=body_mass.cpu().numpy(),
                    inertia_link=body_inertia.cpu().numpy())
            knee_ids = robot.find_joints(['.*knee_joint'], preserve_order=True)[0]
            arm_ids = robot.find_joints(['.*shoulder_pitch_joint', '.*shoulder_roll_joint'], preserve_order=True)[0]
            arm_center = None
            if args.candidate_arm_scale != 1.:
                source_label = args.arm_center_trace.stem.removeprefix('traces_')
                source_cases = json.loads((args.arm_center_trace.parent/f'episodes_{source_label}.json').read_text())
                identity = ('kind', 'mode', 'command', 'family', 'route', 'level', 'variant', 'replica', 'column')
                if len(source_cases) != n or any(any(a[k] != b[k] for k in identity)
                                                   for a, b in zip(source_cases, cases)):
                    raise ValueError('Intervention center trace has different evaluation cases')
                with np.load(args.arm_center_trace, allow_pickle=False) as natural:
                    if list(natural['joint_names']) != list(robot.joint_names) or natural['active'].shape[1] != n:
                        raise ValueError('Intervention center trace joint/case dimensions differ')
                    arm_center = []
                    for cid in range(n):
                        mask = natural['active'][:, cid] & (natural['time'][:, cid] >= .5) & (natural['time'][:, cid] <= 8.)
                        if not mask.any():
                            raise ValueError('No natural action samples for case '+str(cid))
                        arm_center.append(natural['processed_action'][mask, cid][:, arm_ids].mean(0))
                    arm_center = torch.tensor(np.stack(arm_center), device=device)
            traces = {}
            results = {}
            initial_reference = None
            initial_robot = None
            for label, checkpoint in zip(labels, (args.reference, args.candidate)):
                actor_contract=reference_contract if label==labels[0] else candidate
                from beamdojo_support_transition_loader import configure_actor
                settings=configure_actor(agent.to_dict(),actor_contract,checkpoint)
                runner = OnPolicyRunner(wrapped, settings, log_dir=None, device=agent.device)
                runner.load(str(checkpoint), load_cfg={'actor': True}, map_location=agent.device)
                policy = runner.get_inference_policy(device=agent.device)
                if label==labels[1] and args.candidate_no_residual:
                    if not hasattr(policy,'diagnostic_disable_residual'):
                        raise ValueError('Residual ablation requires a support-transition actor')
                    policy.diagnostic_disable_residual=True
                intervention_hook=None
                if label==labels[1] and args.candidate_residual_group!='all':
                    from beamdojo_support_transition_diagnostics import intervene
                    intervention_hook=intervene(policy,robot.joint_names,args.candidate_residual_group)
                actor_before = {k: v.detach().cpu().clone() for k, v in runner.alg.actor.state_dict().items()}
                configure_seed(args.seed)
                # A fixed reset is independent of the preceding model's outcome.
                env.termination_manager.compute = getattr(env, '_paired_original_compute', env.termination_manager.compute)
                obs = wrapped.reset()[0]
                term._update_command()
                env._upstream_heightmap_cache = None
                env._upstream_critic_frame_cache = None
                env.observation_manager.reset(ids)
                obs = TensorDict(env.observation_manager.compute(), batch_size=[n])
                policy.reset(torch.ones(n, device=device, dtype=torch.bool))
                initial = torch.cat((rt.tensor(robot.data.root_pos_w)-rt.tensor(env.scene.env_origins),
                    rt.tensor(robot.data.root_quat_w), rt.tensor(robot.data.root_lin_vel_w),
                    rt.tensor(robot.data.root_ang_vel_w), rt.tensor(robot.data.joint_pos),
                    rt.tensor(robot.data.joint_vel)), -1).detach().cpu().clone()
                if initial_reference is None:
                    initial_reference = initial
                    initial_robot = (rt.tensor(robot.data.joint_pos).cpu().numpy().copy(),
                                     rt.tensor(robot.data.root_quat_w).cpu().numpy().copy())
                elif not torch.allclose(initial, initial_reference, atol=1e-6, rtol=0):
                    raise ValueError('Paired initial state mismatch '+str(float((initial-initial_reference).abs().max())))
                finished = torch.zeros(n, device=device, dtype=torch.bool)
                reached = finished.clone()
                rows = {}
                frames = []
                prediction_prefix = {}
                native_compute = env.termination_manager.compute
                env._paired_original_compute = native_compute
                root_q = rt.tensor(robot.data.root_quat_w)
                initial_yaw = torch.atan2(2*(root_q[:,3]*root_q[:,2]+root_q[:,0]*root_q[:,1]),
                    1-2*(root_q[:,1]**2+root_q[:,2]**2))
                last_yaw = initial_yaw.clone()
                unwrapped_yaw = torch.zeros(n, device=device)
                def capture():
                    done = rt.tensor(native_compute())
                    active = ~finished
                    m = rt.read_measurements(env)
                    route_root, _, full = rt._geometry(env, m)
                    reached.logical_or_(active & ~flat & full & (m['contact_fz']>5).all(-1) & ~env.beamdojo_course_failed)
                    q = rt.tensor(robot.data.root_quat_w)
                    yaw = torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2))
                    delta = yaw-last_yaw
                    unwrapped_yaw.add_(torch.atan2(delta.sin(), delta.cos()))
                    last_yaw.copy_(yaw)
                    elapsed = rt.tensor(env.episode_length_buf).float()*env.step_dt
                    expected_yaw = fixed[:,2]*elapsed
                    foot_centers = m['corners'].mean(-2)
                    joint_pos = rt.tensor(robot.data.joint_pos)
                    frame = dict(active=active.cpu().numpy(), time=elapsed.cpu().numpy(),
                        root=route_root.cpu().numpy(), yaw_deg=(unwrapped_yaw*180/math.pi).cpu().numpy(),
                        heading_error_deg=((unwrapped_yaw-expected_yaw)*180/math.pi).cpu().numpy(),
                        body_velocity=rt.tensor(robot.data.root_lin_vel_b).cpu().numpy(),
                        body_omega=m['ang'].cpu().numpy(), gravity=m['gravity'].cpu().numpy(),
                        contact=(m['contact_fz']>5).cpu().numpy(), foot_centers=foot_centers.cpu().numpy(),
                        contact_fz=m['contact_fz'].cpu().numpy(), full_foot_support=m['supported'].cpu().numpy(),
                        joint_pos=joint_pos.cpu().numpy(), joint_vel=rt.tensor(robot.data.joint_vel).cpu().numpy(),
                        root_quat=q.cpu().numpy(), root_local=m['root'].cpu().numpy(),
                        command=rt.tensor(env.command_manager.get_command('base_velocity')).cpu().numpy())
                    if hasattr(policy,'last_transition_delta'):
                        frame['transition_delta_rad']=(policy.last_transition_delta*.25).detach().cpu().numpy()
                        frame['transition_active']=policy.last_transition_gate.detach().cpu().numpy()
                        frame['transition_region_cell']=policy.last_transition_region.argmax(-1).cpu().numpy()
                        frame['transition_handoff']=policy.last_transition_handoff.cpu().numpy()
                        frame.update(prediction_prefix)
                    if args.arm_audit:
                        pos = rt.tensor(robot.data.body_com_pos_w) - rt.tensor(env.scene.env_origins)[:, None, :]
                        vel = rt.tensor(robot.data.body_com_vel_w)
                        com = (pos * body_mass[..., None]).sum(1) / body_mass.sum(1, keepdim=True)
                        vcom = (vel[..., :3] * body_mass[..., None]).sum(1) / body_mass.sum(1, keepdim=True)
                        link_q = rt.tensor(robot.data.body_link_quat_w)
                        rotation = matrix_from_quat(link_q)
                        omega_link = torch.einsum('nbji,nbj->nbi', rotation, vel[..., 3:])
                        spin_link = torch.einsum('nbij,nbj->nbi', body_inertia, omega_link)
                        spin_world = torch.einsum('nbij,nbj->nbi', rotation, spin_link)
                        momentum = spin_world + torch.linalg.cross(
                            pos-com[:, None, :], body_mass[..., None]*(vel[..., :3]-vcom[:, None, :]))
                        frame.update(body_momentum=momentum.cpu().numpy(), com_local=com.cpu().numpy(),
                            link_local=(rt.tensor(robot.data.body_link_pos_w)-rt.tensor(env.scene.env_origins)[:,None,:]).cpu().numpy(),
                            applied_torque=rt.tensor(robot.data.applied_torque).cpu().numpy(),
                            processed_action=rt.tensor(env.action_manager.action).cpu().numpy())
                    # CPU native tensors may expose mutable simulator buffers.
                    # Freeze each frame before the next physics/reset update.
                    frames.append({key: value.copy() for key, value in frame.items()})
                    for i in (active & done).nonzero().flatten().cpu().tolist():
                        rows[i] = dict(**cases[i], case_id=i, reached=bool(reached[i]),
                            stable_park=bool(env.beamdojo_qualified_success[i]),
                            failed=bool(env.beamdojo_course_failed[i]),
                            timeout=bool(rt.tensor(env.termination_manager.time_outs)[i]),
                            duration_s=float(elapsed[i]), final_route_xy=route_root[i,:2].cpu().tolist())
                    finished.logical_or_(done & active)
                    return done
                env.termination_manager.compute = capture
                status(state='evaluating', model=label, cases=n, completed=0)
                with torch.inference_mode():
                    for step in range(env.max_episode_length+5):
                        if bool(finished.all()):
                            break
                        action = policy(obs.select(*agent.obs_groups['actor']))
                        if hasattr(policy,'last_transition_delta'):
                            prediction_prefix={
                                'transition_prefix_root':(rt.tensor(robot.data.root_pos_w)-rt.tensor(env.scene.env_origins)).cpu().numpy().copy(),
                                'transition_prefix_quat':rt.tensor(robot.data.root_quat_w).cpu().numpy().copy(),
                            }
                        if label == labels[1] and arm_center is not None:
                            action = action.clone()
                            action[:, arm_ids] = arm_center + args.candidate_arm_scale*(action[:, arm_ids]-arm_center)
                        obs, _, done, _ = wrapped.step(action)
                        policy.reset(done)
                        if step%100 == 0:
                            status(state='evaluating', model=label, cases=n,
                                   completed=int(finished.sum()), step=step, arrivals=int(reached.sum()))
                if len(rows) != n:
                    raise ValueError(f'Incomplete evaluation {len(rows)}/{n}')
                data = {key: np.stack([f[key] for f in frames]) for key in frames[0]}
                traces[label] = data
                np.savez_compressed(args.output/f'traces_{label}.npz', **data,
                    joint_names=np.array(robot.joint_names), initial_joint_pos=initial_robot[0],
                    initial_quat=initial_robot[1])
                for i, row in rows.items():
                    # Fixed first 0.5..3 seconds: avoids weighting a cautious,
                    # longer rollout more heavily than a quick/failed rollout.
                    mask = data['active'][:,i] & (data['time'][:,i]>=.5) & (data['time'][:,i]<=3.)
                    if not mask.any():
                        row['metrics_0p5_to_3s'] = None
                        continue
                    v = data['body_velocity'][mask,i]
                    cmd = data['command'][mask,i]
                    contacts = data['contact'][mask,i]
                    g = data['gravity'][mask,i]
                    jp = data['joint_pos'][mask,i]
                    row['metrics_0p5_to_3s'] = dict(
                        observed_seconds=float(mask.sum()*env.step_dt),
                        xy_velocity_error_mps=float(np.linalg.norm(v[:,:2]-cmd[:,:2],axis=-1).mean()),
                        mean_forward_velocity_mps=float(v[:,0].mean()),
                        yaw_rate_error_rads=float(np.abs(data['body_omega'][mask,i,2]-cmd[:,2]).mean()),
                        mean_abs_heading_error_deg=float(np.abs(data['heading_error_deg'][mask,i]).mean()),
                        final_heading_error_deg=float(data['heading_error_deg'][mask,i][-1]),
                        mean_abs_lateral_offset_m=float(np.abs(data['root'][mask,i,1]).mean()),
                        torso_tilt_rms_deg=float(np.sqrt(np.mean(np.arccos(np.clip(-g[:,2],-1,1))**2))*180/math.pi),
                        double_support_fraction=float(contacts.all(-1).mean()),
                        single_support_fraction=float((contacts.sum(-1)==1).mean()),
                        flight_fraction=float((contacts.sum(-1)==0).mean()),
                        knee_excursion_deg=float(np.ptp(jp[:,knee_ids],axis=0).mean()*180/math.pi),
                        arm_excursion_deg=float(np.ptp(jp[:,arm_ids],axis=0).mean()*180/math.pi),
                        joint_speed_rms_rads=float(np.sqrt(np.mean(data['joint_vel'][mask,i]**2))),
                        root_vertical_std_m=float(np.std(data['root'][mask,i,2])),
                    )
                    # Crossing a common 2m marker is useful for straight motion.
                    direction = -1 if row['mode']=='backward' else 1
                    progress = direction*data['root'][:,i,0]
                    crossed = np.flatnonzero(data['active'][:,i] & (progress>=2.))
                    row['heading_at_2m_deg'] = float(data['heading_error_deg'][crossed[0],i]) if len(crossed) else None
                for k, v in actor_before.items():
                    if not torch.equal(v, runner.alg.actor.state_dict()[k].cpu()):
                        raise ValueError('Actor changed '+k)
                ordered = [rows[i] for i in range(n)]
                results[label] = dict(checkpoint=str(checkpoint), episodes=ordered,
                    actor_unchanged=True, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                write(f'episodes_{label}.json', ordered)
                env.termination_manager.compute = native_compute
                if intervention_hook is not None:intervention_hook.remove()
                del policy, runner
            def summary(rows):
                valid = [r['metrics_0p5_to_3s'] for r in rows if r['metrics_0p5_to_3s'] is not None]
                return dict(cases=len(rows), metrics_valid_cases=len(valid), failed=sum(r['failed'] for r in rows),
                    arrivals=sum(r['reached'] for r in rows), stable_parks=sum(r['stable_park'] for r in rows),
                    metrics={k:float(np.mean([r[k] for r in valid])) for k in valid[0]} if valid else {})
            for label in results:
                rows = results[label]['episodes']
                results[label]['by_flat_mode'] = {mode:summary([r for r in rows if r['mode']==mode]) for mode,_ in commands}
                stones = [r for r in rows if r['kind']=='stone']
                results[label]['stone_overall'] = summary(stones)
                results[label]['stone_by_level'] = {str(level):summary([r for r in stones if r['level']==level]) for level in (8,10)}
            map_scope='saved maps; not held-out generalization' if args.geometry_seed is None else f'held-out atlas seed{geometry_seed}; same variant definitions, not a new terrain family'
            output = dict(protocol=f'Common scene profile from {args.environment}; paired fixed first episodes, {flat_count} flat +{n-flat_count} stones; variants {args.variants}. Metrics fixed0.5..3s before reset. Same physics domain randomization shared; no noise/push. {map_scope}.',
                policy_updates=0, identical_initial_physical_state=True, seed=args.seed,
                flat_masking_fix='evaluation-only restore flat physical-failure masking; no stone corridor on flat',
                terminal_revision=REVISION, common_environment_checkpoint=str(args.environment),
                geometry_seed=geometry_seed,actual_support_raster_sha256=actual['square_mixed']['support_raster_sha256'],
                actual_geometry_specs=actual['square_mixed']['specs'],
                candidate_residual_disabled=args.candidate_no_residual,
                candidate_residual_group=args.candidate_residual_group,
                saved_reward_objectives_differ=not contracts_equal(reference_contract['reward_weights'], candidate['reward_weights']),
                episode_limit_s=20., shared_contract_keys=list(shared_keys),
                candidate_shoulder_action_scale=args.candidate_arm_scale,
                arm_center_trace=str(args.arm_center_trace) if args.arm_center_trace else None,
                intervention_note='Action intervention changes closed-loop state, not a retrained/optimized policy; cannot identify optimal arm amplitude.',
                knee_joint_names=[robot.joint_names[i] for i in knee_ids],
                arm_joint_names=[robot.joint_names[i] for i in arm_ids], results=results)
            write('results.json', output)
            status(state='complete', cases_per_model=n, initial_state_paired=True)
            print('PAIRED COMPARISON COMPLETE', flush=True)
    except BaseException:
        code = 1
        status(state='failed', error=traceback.format_exc())
        raise
    finally:
        if wrapped is not None:
            wrapped.close()
        app.app.close(exit_code=code)


if __name__ == '__main__':
    main()
