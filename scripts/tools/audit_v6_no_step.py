"""Read-only checkpoint audit: reference encoding, discriminator and finite rollouts."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path

import gymnasium as gym
import torch
import yaml
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab.utils.math import quat_apply
from rsl_rl.runners import OnPolicyRunner
import legged_lab.tasks
from legged_lab.tasks.locomotion.amp.mdp.style_state import (
    tensor, encode_style, agent_style_state, mirror_style, joint_mirror_map,
)


def summary(x):
    x = x.detach().float().reshape(-1).cpu()
    assert torch.isfinite(x).all()
    return dict(mean=x.mean().item(), p05=x.quantile(.05).item(),
                p50=x.median().item(), p95=x.quantile(.95).item())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1000)
    add_launcher_args(p)
    args = p.parse_args()
    task = 'LeggedLab-Isaac-AMP-Depth-Target-G1-Play-v2'
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
    cfg.sim.physics = copy.deepcopy(cfg.sim.physics.default)
    cfg.scene.num_envs = 6
    cfg.seed = 42
    cfg.scene.terrain.max_init_terrain_level = 0
    gen = cfg.scene.terrain.terrain_generator
    gen.num_cols = 6
    gen.num_rows = 1
    gen.difficulty_range = (0., 0.)
    for sub in gen.sub_terrains.values():
        sub.proportion = 1.
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            obs, _ = env.reset()
            robot = env.scene['robot']
            names = yaml.safe_load(Path('scripts/tools/retarget/config/g1_29dof.yaml').read_text())['lab_dof_names']
            assert robot.joint_names == names
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            agent.device = env.device
            runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=env.device)
            runner.load(args.checkpoint, map_location=env.device)
            policy = runner.get_inference_policy(device=env.device)
            disc = runner.alg.amp_discriminator
            result = {'checkpoint': args.checkpoint, 'steps_per_mode': args.steps,
                      'joint_order_matches_retarget': True, 'terrain': list(gen.sub_terrains),
                      'scope': 'Six easiest terrain tiles; fixed forward 0.5 m/s commands; finite diagnostic, no training.'}

            def score(states):
                with torch.no_grad():
                    rew, d = disc.predict_style_reward(states.contiguous(), env.step_dt)
                return {'D': summary(d), 'style': summary(rew), 'raw_style': summary(rew / .25)}

            ref = env.animation_manager.get_term('animation')
            data = ref.motion_data_term
            # Kinematic round trip tests the real agent encoder and history order.
            expected = encode_style(ref.get_root_quat(), ref.get_dof_pos(), ref.get_dof_vel(),
                ref.get_root_vel_w(), ref.get_root_ang_vel_w(), tensor(robot.data.default_joint_pos)[:, None])
            errors = []
            for frame in range(10):
                q = ref.get_root_quat()[:, frame]
                robot.write_root_pose_to_sim(torch.cat((ref.get_root_pos_w()[:, frame] + env.scene.env_origins, q), -1))
                robot.write_joint_state_to_sim(ref.get_dof_pos()[:, frame], ref.get_dof_vel()[:, frame])
                w = ref.get_root_ang_vel_w()[:, frame]
                com_v = ref.get_root_vel_w()[:, frame] + torch.cross(w, quat_apply(q, tensor(robot.data.body_com_pos_b)[:, 0]), dim=-1)
                robot.write_root_velocity_to_sim(torch.cat((com_v, w), -1))
                env.sim.forward()
                env.scene.update(env.physics_dt)
                errors.append(float((agent_style_state(env) - expected[:, frame]).abs().max()))
                obs = env.observation_manager.compute(update_history=True)
            result['encoding_max_error'] = max(errors)
            result['history_max_error'] = float((obs['disc'] - expected).abs().max())
            assert max(errors) < 2e-4 and result['history_max_error'] < 2e-4
            result['references'] = {}
            indices, signs = joint_mirror_map(robot.joint_names, env.device)
            all_states = []
            for i, name in enumerate(data.motion_weights_dict):
                ids = torch.full((128,), i, device=env.device, dtype=torch.long)
                start = torch.linspace(0., max(0., float(data.motion_durations[i]) - .2), 128, device=env.device)
                times = start[:, None] + torch.arange(10, device=env.device)[None] * env.step_dt
                state = data.get_motion_state(ids.repeat_interleave(10), times.reshape(-1))
                z = encode_style(state['root_quat'], state['dof_pos'], state['dof_vel'],
                    state['root_vel_w'], state['root_ang_vel_w'], tensor(robot.data.default_joint_pos)[0]).reshape(128, 10, 67)
                reflected = mirror_style(z, indices, signs)
                paired = torch.cat((z, reflected))
                all_states.append(paired)
                speed = z[..., 61:63].norm(dim=-1).mean(-1)
                result['references'][name] = {'weight': float(data.motion_weights[i]),
                    'speed_m_s': summary(speed), 'slow_window_fraction': float((speed < .15).float().mean()),
                    'joint_frame_delta_rad': summary((z[:, 1:, 3:32] - z[:, :-1, 3:32]).abs()),
                    'score': score(paired)}
            all_states = torch.cat(all_states)
            result['reference_all'] = score(all_states)
            frozen = all_states[:, :1].expand(-1, 10, -1).clone()
            frozen[..., 32:] = 0.
            result['reference_frozen_pose_zero_velocity'] = score(frozen)
            result['reference_reverse_time'] = score(all_states.flip(1))
            result['reference_sampling'] = {'clips': len(data.motion_weights_dict),
                'random_fetch': ref.cfg.random_fetch, 'random_initialize': ref.cfg.random_initialize,
                'frame_spacing_s': env.step_dt,
                'weighted_slow_window_fraction': sum(v['weight'] * v['slow_window_fraction'] for v in result['references'].values())}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
            print('REFERENCE_DONE', json.dumps({k: v for k, v in result.items() if k != 'references'}), flush=True)

            result['rollouts'] = {}
            for mode in ('policy', 'zero_action', 'policy_sampled'):
                torch.manual_seed(42)
                obs, _ = env.reset()
                command = env.command_manager.get_term('base_velocity')
                # Override only this diagnostic instance; goals are ignored to isolate forward response.
                original_compute = command.compute
                def fixed_compute(dt):
                    original_compute(dt)
                    command.vel_command_b[:] = command.vel_command_b.new_tensor([.5, 0., 0.])
                    command.is_standing_env[:] = False
                command.compute = fixed_compute
                command.compute(0.)
                obs = wrapped.get_observations()
                states, rews, terms, velocities, knees, air, dones_all, fail_all = [], [], [], [], [], [], [], []
                knee_ids = [robot.joint_names.index(f'{s}_knee_joint') for s in ('left', 'right')]
                feet_ids = [robot.body_names.index(f'{s}_ankle_roll_link') for s in ('left', 'right')]
                positions = []
                with torch.inference_mode():
                    for step in range(args.steps):
                        if mode == 'zero_action':
                            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
                        elif mode == 'policy_sampled':
                            actions = policy(obs, stochastic_output=True)
                        else:
                            actions = policy(obs)
                        obs, reward, done, info = wrapped.step(actions)
                        policy.reset(done)
                        if step >= 10:
                            s = info.get('terminal_obs', {}).get('disc', obs['disc'])
                            states.append(s.detach().cpu())
                            rews.append(reward.detach().cpu())
                            terms.append(env.reward_manager._step_reward.detach().cpu().clone() * env.step_dt)
                            velocities.append(tensor(robot.data.root_link_lin_vel_w).detach().cpu().clone())
                            knees.append(tensor(robot.data.joint_pos)[:, knee_ids].detach().cpu().clone())
                            positions.append(tensor(robot.data.body_pos_w)[:, feet_ids].detach().cpu().clone())
                            contact = env.scene['contact_forces']
                            sensor_feet = [contact.body_names.index(f'{s}_ankle_roll_link') for s in ('left', 'right')]
                            air.append(tensor(contact.data.current_air_time)[:, sensor_feet].detach().cpu().clone())
                            dones_all.append(done.detach().cpu().clone())
                            fail_all.append(tensor(env.termination_manager.terminated).detach().cpu().clone())
                command.compute = original_compute
                s = torch.cat(states).to(env.device)
                term_means = torch.stack(terms).mean((0, 1))
                r = {'score': score(s), 'task_per_step': summary(torch.cat(rews)),
                     'reward_terms_per_step': dict(zip(env.reward_manager.active_terms, term_means.tolist())),
                     'world_vx_m_s': summary(torch.stack(velocities)[..., 0]),
                     'xy_speed_m_s': summary(torch.stack(velocities)[..., :2].norm(dim=-1)),
                     'knee_rad': summary(torch.stack(knees)),
                     'foot_air_fraction': float((torch.stack(air) > .04).float().mean()),
                     'finished_episodes': int(torch.stack(dones_all).sum()),
                     'failed_episodes': int(torch.stack(fail_all).sum())}
                result['rollouts'][mode] = r
                torch.save({'disc': torch.cat(states), 'knee': torch.stack(knees), 'foot_pos': torch.stack(positions),
                            'done': torch.stack(dones_all), 'reference': all_states.cpu()}, args.output.with_name(mode + '_states.pt'))
                args.output.write_text(json.dumps(result, indent=2))
                print('ROLLOUT_DONE', mode, json.dumps(r), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
