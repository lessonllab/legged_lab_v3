"""Validate v3 AMP encoding against actual PhysX reference poses and histories."""
import argparse
import copy
import json
from pathlib import Path
import gymnasium as gym
import torch
import yaml
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab.utils.math import quat_apply
import legged_lab.tasks
from legged_lab.tasks.locomotion.amp.mdp.style_state import agent_style_state, encode_style, mirror_style, tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/analysis/style_v3/physical_state.json'))
    add_launcher_args(parser)
    args = parser.parse_args()
    task = 'LeggedLab-Isaac-AMP-Depth-Style-G1-Play-v0'
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    cfg.sim.physics = copy.deepcopy(cfg.sim.physics.default)
    cfg.scene.num_envs = 4
    cfg.seed = 42
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset()
            robot = env.scene['robot']
            names = yaml.safe_load(Path('scripts/tools/retarget/config/g1_29dof.yaml').read_text())['lab_dof_names']
            assert robot.joint_names == names, (robot.joint_names, names)
            ref = env.animation_manager.get_term('animation')
            term = env.observation_manager._group_obs_term_cfgs['disc_demo'][0].func
            term.mirrored[:] = torch.tensor([False, True, False, True], device=env.device)
            expected = encode_style(ref.get_root_quat(), ref.get_dof_pos(), ref.get_dof_vel(),
                ref.get_root_vel_w(), ref.get_root_ang_vel_w(), tensor(robot.data.default_joint_pos)[:, None])
            original = expected.clone()
            errors = []
            # Kinematic writes, no integration or animation update: compare identical states.
            for frame in range(10):
                pos = ref.get_root_pos_w()[:, frame] + env.scene.env_origins
                robot.write_root_pose_to_sim(torch.cat((pos, ref.get_root_quat()[:, frame]), -1))
                robot.write_joint_state_to_sim(ref.get_dof_pos()[:, frame], ref.get_dof_vel()[:, frame])
                # PhysX accepts COM velocity. Convert the reference link velocity
                # explicitly; the installed link-velocity writer does not survive
                # a forward/read round trip with the requested link velocity.
                offset = quat_apply(ref.get_root_quat()[:, frame], tensor(robot.data.body_com_pos_b)[:, 0])
                ang = ref.get_root_ang_vel_w()[:, frame]
                com_vel = ref.get_root_vel_w()[:, frame] + torch.cross(ang, offset, dim=-1)
                robot.write_root_velocity_to_sim(torch.cat((com_vel, ang), -1))
                env.sim.forward()
                env.scene.update(env.physics_dt)
                actual = agent_style_state(env)
                torch.testing.assert_close(actual, expected[:, frame], atol=2e-4, rtol=2e-4)
                errors.append((actual - expected[:, frame]).abs().max().item())
                observations = env.observation_manager.compute(update_history=True)
            torch.testing.assert_close(observations['disc'], expected, atol=2e-4, rtol=2e-4)
            reflected = mirror_style(expected, term.indices, term.signs)
            torch.testing.assert_close(observations['disc_demo'][1::2], reflected[1::2])
            torch.testing.assert_close(observations['disc_demo'][::2], expected[::2])
            torch.testing.assert_close(expected, original)
            mask = term.mirrored.clone()
            env.observation_manager.reset(torch.tensor([0], device=env.device))
            observations = env.observation_manager.compute(update_history=True)
            assert torch.equal(term.mirrored[1:], mask[1:])
            torch.testing.assert_close(observations['disc'][0], expected[0, -1:].expand(10, -1), atol=2e-4, rtol=2e-4)
            for key, value in observations.items():
                if isinstance(value, torch.Tensor):
                    assert torch.isfinite(value).all(), key
            result = {'task': task, 'device': env.device, 'num_envs': 4,
                'disc_shape': list(observations['disc'].shape), 'disc_demo_shape': list(observations['disc_demo'].shape),
                'max_state_errors_per_frame': errors, 'joint_order_matches_retarget': True,
                'history_order_matches_reference': True, 'mirror_and_partial_reset_pass': True,
                'scope': 'Kinematic observation consistency, not learned gait quality.'}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
            print('PASS', json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
