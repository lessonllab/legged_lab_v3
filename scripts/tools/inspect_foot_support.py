"""Check actual sole rays on a 20 cm step, with controlled stance loading."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
import trimesh
import gymnasium as gym
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
from isaaclab.terrains import TerrainGeneratorCfg, SubTerrainBaseCfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.utils import load_cfg_from_registry
import legged_lab.tasks
from legged_lab.tasks.locomotion.amp.mdp.foot_support import support_costs


def step_terrain(difficulty, cfg):
    lower = trimesh.creation.box(extents=(4., 4., .1))
    lower.apply_translation((2., 2., -.25))
    upper = trimesh.creation.box(extents=(2., 4., .2))
    upper.apply_translation((1., 2., -.1))
    return [lower, upper], np.array([1., 2., 0.])


def tensor(x):
    return x if isinstance(x, torch.Tensor) else x.torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, default=Path('docs/analysis/foot_support_20260911'))
    add_launcher_args(parser)
    args = parser.parse_args()
    task = 'LeggedLab-Isaac-AMP-Depth-Instinct-Warmup-G1-Play-v0'
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    cfg.sim.physics = copy.deepcopy(cfg.sim.physics.default)
    cfg.scene.num_envs = 4
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=42, size=(4., 4.), num_rows=1, num_cols=4, border_width=0.,
        sub_terrains={'step': SubTerrainBaseCfg(function=step_terrain, proportion=1.)})
    cfg.scene.terrain.max_init_terrain_level = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset()
            robot = env.scene['robot']
            q = torch.zeros_like(tensor(robot.data.default_joint_pos))
            base = tensor(robot.data.default_root_state).clone()
            base[:, :3] += env.scene.env_origins
            base[:, 3:7] = torch.tensor([0., 0., 0., 1.], device=env.device)
            base[:, 7:] = 0.
            foot_ids = [robot.body_names.index(f'{side}_ankle_roll_link') for side in ('left', 'right')]
            def place(state):
                robot.write_root_state_to_sim(state)
                robot.write_joint_state_to_sim(q, torch.zeros_like(q))
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(env.physics_dt)
            results = {}
            for label, x, ground in [('safe_upper', .5, 0.), ('near_edge', .86, 0.),
                                     ('overhang', .99, 0.), ('safe_lower', 1.5, -.2)]:
                place(base)
                target = env.scene.env_origins.clone()
                target[:, 0] += x
                target[:, 2] += ground + .035
                state = base.clone()
                state[:, :3] += target - tensor(robot.data.body_pos_w)[:, foot_ids[0]]
                place(state)
                heights = []
                for i, side in enumerate(('left', 'right')):
                    sensor = env.scene[f'{side}_sole_scanner']
                    sensor.update(env.step_dt, force_recompute=True)
                    hits = tensor(sensor.data.ray_hits_w)
                    pos = tensor(robot.data.body_pos_w)[:, foot_ids[i]]
                    quat = tensor(robot.data.body_quat_w)[:, foot_ids[i]]
                    local = quat_apply_inverse(quat[:, None].expand(-1, 55, -1), hits - pos[:, None])
                    heights.append(local[..., 2].reshape(-1, 5, 11))
                h = torch.stack(heights, 1)
                # Isolate ray geometry from transient contact solver/reset state.
                force = torch.full((4, 2), 100., device=env.device)
                duration = torch.ones_like(force)
                deficit, edge, ratio = support_costs(h, force, duration)
                if label.startswith('safe'):
                    assert deficit.max() == 0 and edge.max() == 0, (label, h, deficit, edge)
                elif label == 'near_edge':
                    assert deficit.max() == 0 and edge.min() > 0, (label, deficit, edge)
                else:
                    assert deficit.min() > 0 and edge.min() > 0, (label, deficit, edge)
                swing_d, swing_e, _ = support_costs(h, force * 0., duration)
                assert swing_d.max() == 0 and swing_e.max() == 0
                assert torch.isfinite(deficit).all() and torch.isfinite(edge).all()
                results[label] = {'support_fraction': ratio.cpu().tolist(),
                                  'support_cost': deficit.cpu().tolist(), 'edge_cost': edge.cpu().tolist(),
                                  'swing_cost': 0.}
                np.save(args.output_dir / f'{label}_sole_heights.npy', h.cpu().numpy())
            (args.output_dir / 'step_test.json').write_text(json.dumps(results, indent=2))
            print('PASS', json.dumps(results), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
