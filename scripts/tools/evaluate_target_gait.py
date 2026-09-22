"""Finite deterministic locomotion evaluation from standing, with contact cycles."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import gymnasium as gym
import torch
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import legged_lab.tasks
from legged_lab.tasks.locomotion.amp.mdp.style_state import tensor
from legged_lab.tasks.locomotion.amp.mdp.locomotion_progress import motion_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', default='LeggedLab-Isaac-AMP-Depth-Target-G1-Play-v3')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--speeds', type=float, nargs='+', default=[.5, 1.5])
    add_launcher_args(p)
    args = p.parse_args()
    cfg = load_cfg_from_registry(args.task, 'env_cfg_entry_point')
    agent = load_cfg_from_registry(args.task, 'rsl_rl_cfg_entry_point')
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
    cfg.sim.physics = copy.deepcopy(cfg.sim.physics.default)
    cfg.scene.num_envs = 6
    cfg.seed = 42
    gen = cfg.scene.terrain.terrain_generator
    gen.num_cols, gen.num_rows = 6, 1
    gen.difficulty_range = (0., 0.)
    cfg.scene.terrain.max_init_terrain_level = 0
    for sub in gen.sub_terrains.values():
        sub.proportion = 1.
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg).unwrapped
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=env.device)
            runner.load(args.checkpoint, map_location=env.device)
            policy = runner.get_inference_policy(device=env.device)
            robot = env.scene['robot']
            sensor = env.scene['contact_forces']
            feet = [sensor.body_names.index(f'{side}_ankle_roll_link') for side in ('left', 'right')]
            result = {'checkpoint': args.checkpoint, 'task': args.task, 'steps': args.steps,
                      'scope': 'Six easiest tiles, standing resets; requested forward speed capped by each terrain range; finite diagnostic, not a success-rate benchmark.', 'speeds': {}}
            for speed in args.speeds:
                torch.manual_seed(42)
                env.reset()
                command = env.command_manager.get_term('base_velocity')
                caps = torch.tensor([min(speed, cfg.commands.base_velocity.speed_ranges[name][1])
                                     for name in gen.sub_terrains], device=env.device)
                original = command.compute
                def fixed(dt):
                    original(dt)
                    command.vel_command_b.zero_()
                    command.vel_command_b[:, 0] = caps
                    command.is_standing_env[:] = False
                command.compute = fixed
                command.compute(0.)
                obs = wrapped.get_observations()
                velocity, air_history, gates, styles, heights, long_air, outside = [], [], [], [], [], [], []
                failures = torch.zeros(6, device=env.device)
                contact_cycles = torch.zeros(6, 2, device=env.device)
                was_air = torch.zeros(6, 2, device=env.device, dtype=torch.bool)
                with torch.inference_mode():
                    for i in range(args.steps):
                        obs, _, done, info = wrapped.step(policy(obs))
                        policy.reset(done)
                        air = tensor(sensor.data.current_air_time)[:, feet] > .04
                        contact_cycles += (air & ~was_air & ~done.bool()[:, None]).float()
                        was_air = air & ~done.bool()[:, None]
                        failures += tensor(env.termination_manager.terminated).float()
                        if i >= 100:
                            foot_ids = [robot.body_names.index(f'{side}_ankle_roll_link') for side in ('left', 'right')]
                            heights.append((tensor(robot.data.root_pos_w)[:, 2] -
                                            tensor(robot.data.body_pos_w)[:, foot_ids, 2].min(-1).values).clone())
                            velocity.append(motion_state(env)[1].clone())
                            air_history.append(air.clone())
                            long_air.append((tensor(sensor.data.current_air_time)[:, feet] > .10).clone())
                            local_xy = tensor(robot.data.root_pos_w)[:, :2] - tensor(env.scene.env_origins)[:, :2]
                            half_size = local_xy.new_tensor(gen.size) * .5
                            outside.append((local_xy.abs() > half_size).any(-1))
                            gate = obs.get('style_gate', torch.ones(6, 1, device=env.device)).reshape(-1)
                            gates.append(gate)
                            state = info.get('terminal_obs', {}).get('disc', obs['disc'])
                            style, _ = runner.alg.amp_discriminator.predict_style_reward(state.contiguous(), env.step_dt)
                            styles.append(style)
                command.compute = original
                v = torch.stack(velocity)
                air_mean = torch.stack(air_history).float().mean((0, 2))
                rows = []
                for j, name in enumerate(gen.sub_terrains):
                    rows.append({'terrain': name, 'command_vx': float(caps[j]), 'vx_mean': float(v[:, j, 0].mean()),
                                 'xy_speed_mean': float(v[:, j].norm(dim=-1).mean()),
                                 'foot_air_fraction': float(air_mean[j]),
                                 'left_liftoffs': int(contact_cycles[j, 0]), 'right_liftoffs': int(contact_cycles[j, 1]),
                                 'failures': int(failures[j]), 'gate_mean': float(torch.stack(gates)[:, j].mean())})
                    rows[-1]['pelvis_above_lower_ankle_m'] = float(torch.stack(heights)[:, j].mean())
                    rows[-1]['foot_air_over_100ms_fraction'] = float(torch.stack(long_air)[:, j].float().mean())
                    rows[-1]['outside_spawn_tile_fraction'] = float(torch.stack(outside)[:, j].float().mean())
                result['speeds'][str(speed)] = rows
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2))
                print('GAIT_RESULT', speed, json.dumps(rows), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
