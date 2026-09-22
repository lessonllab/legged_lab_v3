"""Read-only checkpoint rollout measuring individual course promotion blockers."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import gymnasium as gym
import torch
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
from isaaclab.managers import CurriculumTermCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import legged_lab.tasks
from legged_lab.tasks.locomotion.amp.mdp.scratch_curriculum import scratch_tracking_curriculum
from legged_lab.tasks.locomotion.amp.mdp.style_state import tensor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--modes', nargs='+', choices=['deterministic', 'sampled', 'capped'], default=['deterministic', 'sampled'])
    add_launcher_args(p)
    args = p.parse_args()
    task = 'LeggedLab-Isaac-AMP-Scratch-G1-v0'
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
    cfg.scene.num_envs = 64
    cfg.seed = 42
    cfg.sim.physics = cfg.sim.physics.default.copy()
    records, mode = [], 'initial'
    def audit(env, env_ids):
        cmd = env.command_manager.get_term('base_velocity')
        ids = cmd._ids(env_ids)
        ids = ids[(tensor(env.episode_length_buf)[ids] > 0) & ~cmd.skip_curriculum_once[ids]]
        if len(ids) and mode != 'initial':
            m = cmd.metrics
            values = torch.stack([
                tensor(env.scene.terrain.terrain_levels)[ids].float(),
                m['tracking_exp_vel_xy'][ids], m['moving_tracking_exp_vel_xy'][ids],
                m['target_progress_m'][ids], m['moving_opportunity_s'][ids],
                tensor(env.termination_manager.time_outs)[ids].float(),
                tensor(env.termination_manager.terminated)[ids].float(),
                cmd.success_streak[ids].float(),
                tensor(env.episode_length_buf)[ids].float() * env.step_dt,
            ], -1).cpu().tolist()
            records.extend(dict(mode=mode, level=v[0], xy=v[1], moving_xy=v[2], progress=v[3],
                                opportunity=v[4], timeout=bool(v[5]), failed=bool(v[6]),
                                prior_streak=v[7], duration=v[8]) for v in values)
        return scratch_tracking_curriculum(env, env_ids)
    cfg.curriculum.terrain_levels = CurriculumTermCfg(func=audit)
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=env.device)
            for label in args.modes:
                mode = 'initial'
                torch.manual_seed(42)
                runner.load(args.checkpoint, map_location=env.device)
                runner.alg.eval_mode()
                runner.alg.actor.distribution.std_range[1] = .5 if label == 'capped' else 1.5
                mode = label
                obs = wrapped.get_observations()
                with torch.inference_mode():
                    for _ in range(args.steps):
                        action = runner.alg.actor(obs, stochastic_output=label != 'deterministic')
                        obs, _, done, _ = wrapped.step(action)
                        runner.alg.actor.reset(done)
            summaries = {}
            for label in args.modes:
                for terrain in ('all', 'flat', 'rough'):
                    rows = [r for r in records if r['mode'] == label and
                            (terrain == 'all' or (r['level'] == 0) == (terrain == 'flat'))]
                    n = max(1, len(rows))
                    success = lambda r: r['xy'] > .6 and r['progress'] >= 1. and r['opportunity'] >= 2. and r['timeout'] and not r['failed']
                    summaries[label + '/' + terrain] = {
                        'completed_episodes': len(rows),
                        'mean_duration_s': sum(r['duration'] for r in rows) / n,
                        'xy_below_upgrade_fraction': sum(r['xy'] <= .6 for r in rows) / n,
                        'progress_below_1m_fraction': sum(r['progress'] < 1. for r in rows) / n,
                        'opportunity_below_2s_fraction': sum(r['opportunity'] < 2. for r in rows) / n,
                        'failed_fraction': sum(r['failed'] for r in rows) / n,
                        'single_episode_pass_fraction': sum(success(r) for r in rows) / n,
                        'three_streak_ready_fraction': sum(success(r) and r['prior_streak'] >= 2 for r in rows) / n,
                        'moving_xy': sum(r['moving_xy'] for r in rows) / n,
                    }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'checkpoint': args.checkpoint, 'steps_per_mode': args.steps,
                'scope': '64 environments; selected action modes, no policy updates; finite diagnostics, not convergence evidence.',
                'summary': summaries, 'episodes': records}, indent=2))
            print('AUDIT_RESULT', json.dumps(summaries), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
