"""Replay the exact reference clip list saved with a training run in Viser."""
import argparse
import copy
from pathlib import Path
import sys
import time
import gymnasium as gym
import torch
import yaml
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import load_cfg_from_registry
import legged_lab.tasks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rsl_rl'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--motion', help='One clip name from the saved training weights')
    parser.add_argument('--list_motions', action='store_true')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--max_steps', type=int, default=0, help='0 keeps the viewer running')
    add_launcher_args(parser)
    parser.set_defaults(device='cpu', visualizer=['none'])
    args = parser.parse_args()
    saved = yaml.load((args.run_dir / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    data = saved['motion_data']['motion_dataset']
    weights = {k: float(v) for k, v in data['motion_data_weights'].items()}
    if args.list_motions:
        print('\n'.join(weights))
        return
    if args.motion:
        if args.motion not in weights:
            parser.error('Motion not used by this training run; use --list_motions')
        weights = {args.motion: weights[args.motion]}
    task = 'LeggedLab-Isaac-Animation-G1-v0'
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    cfg.sim.physics = copy.deepcopy(getattr(cfg.sim.physics, 'default', cfg.sim.physics))
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.motion_data.motion_dataset.motion_data_dir = data['motion_data_dir']
    cfg.motion_data.motion_dataset.motion_data_weights = weights
    cfg.animation.animation.random_initialize = False
    viewer = None
    with launch_simulation(cfg, args):
        from physx_viser import PhysxViser
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset()
            env.sim.forward()
            env.scene.update(0.)
            viewer = PhysxViser(env, port=args.port, follow=True, asset_name='robot_anim')
            viewer.server.gui.add_markdown('动作集：' + str(data['motion_data_dir']) +
                                           '\n\n' + (args.motion or f'按训练权重抽取 {len(weights)} 个片段'))
            action = torch.zeros(env.action_space.shape, device=env.device)
            count = 0
            with torch.inference_mode():
                while not args.max_steps or count < args.max_steps:
                    start = time.monotonic()
                    if not viewer.pause.value:
                        env.step(action)
                        # Animation writes poses after physics; refresh FK before
                        # displaying the reference, without an extra physics step.
                        env.sim.forward()
                        env.scene.update(0.)
                        viewer.update()
                        count += 1
                    time.sleep(max(0., env.step_dt - (time.monotonic() - start)))
            print(f'Reference replay completed {count} steps.', flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            if viewer:
                viewer.close()
            env.close()


if __name__ == '__main__':
    main()
