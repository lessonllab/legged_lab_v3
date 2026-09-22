"""Finite PhysX validation of the repaired visual target task; no training."""

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


EXPECTED_SHAPES = {
    "policy": (6, 495),
    "critic": (6, 600),
    "depth": (6, 8, 18, 32),
    "disc": (6, 10, 67),
    "disc_demo": (6, 10, 67),
}


def check_observations(observations):
    for name, shape in EXPECTED_SHAPES.items():
        value = observations[name]
        assert tuple(value.shape) == shape, (name, value.shape, shape)
        assert torch.isfinite(value).all(), name


def check_standing_reset(env, ids):
    """Read the actual simulated state immediately after a reset."""
    data = env.scene["robot"].data
    joint_offset = tensor(data.joint_pos)[ids] - tensor(data.default_joint_pos)[ids]
    root_local = tensor(data.root_pos_w)[ids] - env.scene.env_origins[ids]
    default_local = tensor(data.default_root_pose)[ids, :3]
    position_error = root_local - default_local
    velocity = tensor(data.root_vel_w)[ids]
    assert joint_offset.abs().max() <= .15001, joint_offset.abs().max()
    assert tensor(data.joint_vel)[ids].abs().max() < 1e-4
    assert position_error[:, :2].abs().max() <= .10001, position_error
    assert position_error[:, 2].abs().max() <= 1e-4, position_error
    assert velocity.abs().max() <= .20001, velocity.abs().max()
    return {
        "max_joint_position_offset_rad": float(joint_offset.abs().max()),
        "max_root_xy_offset_m": float(position_error[:, :2].abs().max()),
        "max_root_velocity_component": float(velocity.abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/analysis/target_v5/physical.json"))
    parser.add_argument("--steps", type=int, default=50)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    task = "LeggedLab-Isaac-AMP-Depth-Target-G1-v1"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version("rsl-rl-lib"))
    assert cfg.events.reset_from_ref is None
    assert cfg.curriculum.terrain_levels.func.__name__ == "target_progress_curriculum"
    assert agent.actor.distribution_cfg.learn_std
    assert agent.actor.distribution_cfg.init_std == .5
    assert cfg.animation.animation.num_steps_to_use == 10
    assert cfg.scene.depth_camera is not None
    cfg.sim.physics = copy.deepcopy(cfg.sim.physics.default)
    cfg.scene.num_envs = 6
    cfg.seed = 42
    cfg.scene.terrain.max_init_terrain_level = 0
    generator = cfg.scene.terrain.terrain_generator
    generator.num_rows = 2
    generator.num_cols = 6
    generator.difficulty_range = (0., 0.)
    for sub_terrain in generator.sub_terrains.values():
        sub_terrain.proportion = 1.
    with launch_simulation(cfg, args):
        raw = gym.make(task, cfg=cfg).unwrapped
        try:
            observations, _ = raw.reset()
            check_observations(observations)
            ids = torch.arange(raw.num_envs, device=raw.device)
            reset_checks = [check_standing_reset(raw, ids)]
            # Resetting the robot to stand must retain moving AMP demonstrations.
            demo_motion = (observations["disc_demo"][:, 1:, 3:32] -
                           observations["disc_demo"][:, :-1, 3:32]).abs().max()
            assert demo_motion > 1e-4, demo_motion
            command = raw.command_manager.get_term("base_velocity")
            assert command.column_names == list(generator.sub_terrains)
            assert len(set(command.column_names)) == 6
            # Exercise real partial-reset plumbing, including the curriculum,
            # animation, command manager and observation history callbacks.
            partial = torch.tensor([0, 3], device=raw.device)
            keep = torch.tensor([1, 2, 4, 5], device=raw.device)
            progress_fields = ("target_initial_distance", "target_best_progress", "_target_valid",
                               "_target_valid_reached")
            episode_metrics = ("valid_targets_reached", "target_progress_m", "moving_opportunity_s",
                               "moving_tracking_exp_vel_xy", "moving_tracking_exp_vel_yaw")
            for _ in range(2):
                saved_goals = command.pos_command_w[keep].clone()
                saved_metrics = {key: value[keep].clone() for key, value in command.metrics.items()}
                saved_progress = {key: getattr(command, key)[keep].clone() for key in progress_fields}
                # Seed prior-episode bookkeeping deliberately, so clearing zeros
                # cannot accidentally pass this reset-leak regression check.
                command.target_best_progress[partial] = .1
                command._target_valid_reached[partial] = True
                for key in episode_metrics:
                    command.metrics[key][partial] = .1
                raw._reset_idx(partial)
                raw.sim.forward()
                raw.scene.update(raw.physics_dt)
                torch.testing.assert_close(command.pos_command_w[keep], saved_goals)
                for key, before in saved_metrics.items():
                    torch.testing.assert_close(command.metrics[key][keep], before)
                for key, before in saved_progress.items():
                    torch.testing.assert_close(getattr(command, key)[keep], before)
                reset_checks.append(check_standing_reset(raw, partial))
                observations = raw.observation_manager.compute(update_history=True)
                check_observations(observations)
                assert (command.metrics["tracking_exp_vel_xy"][partial] == 0).all()
                assert (command.metrics["tracking_exp_vel_yaw"][partial] == 0).all()
                assert (command.target_best_progress[partial] == 0).all()
                assert not command._target_valid_reached[partial].any()
                for key in episode_metrics:
                    assert (command.metrics[key][partial] == 0).all(), key
                distance = (command.pos_command_w[partial, :2] -
                            tensor(raw.scene["robot"].data.root_pos_w)[partial, :2]).norm(dim=-1)
                torch.testing.assert_close(command.target_initial_distance[partial], distance)

            wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
            agent.device = raw.device
            runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=raw.device)
            distribution = runner.alg.actor.distribution
            std_parameters = dict(distribution.named_parameters())
            assert std_parameters, "The action distribution has no parameter"
            assert all(parameter.requires_grad for parameter in std_parameters.values())
            assert all(torch.isfinite(parameter).all() for parameter in std_parameters.values())
            initial_std = distribution.std_param if hasattr(distribution, "std_param") else distribution.log_std_param.exp()
            torch.testing.assert_close(initial_std, torch.full_like(initial_std, .5))
            # Zero actions make this an interface/stability check, not an audit of
            # a random untrained network's ability to walk.
            actions = torch.zeros(raw.num_envs, raw.action_manager.total_action_dim, device=raw.device)
            terminations = 0
            for _ in range(args.steps):
                observations, rewards, dones, _ = wrapped.step(actions)
                check_observations(observations)
                assert torch.isfinite(rewards).all()
                assert torch.isfinite(command.command).all()
                assert (command.command[:, 0] >= 0).all()
                assert (command.command[:, 0] <= command.speed_cap + 1e-6).all()
                assert (command.command[:, 1] == 0).all()
                assert (command.command[:, 2].abs() <= 1.).all()
                assert all(torch.isfinite(value).all() for value in command.metrics.values())
                terminations += int(dones.sum())
            result = {
                "task": task,
                "device": raw.device,
                "num_envs": raw.num_envs,
                "observation_shapes": {name: list(observations[name].shape) for name in EXPECTED_SHAPES},
                "terrain_columns": command.column_names,
                "standing_resets": reset_checks,
                "reference_joint_temporal_delta_max_rad": float(demo_motion),
                "partial_reset_preserves_other_commands_and_metrics": True,
                "learnable_std_parameters": list(std_parameters),
                "initial_action_std": float(initial_std.detach().mean()),
                "finite_steps": args.steps,
                "terminations_during_zero_action_check": terminations,
                "scope": "Configuration, simulated reset state and finite interfaces; not trained gait performance.",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("PASS", json.dumps(result), flush=True)
        finally:
            raw.close()


if __name__ == "__main__":
    main()
