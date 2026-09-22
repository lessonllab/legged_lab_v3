"""Finite PhysX validation of the Instinct-style AMP target profile; no training."""

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
from isaaclab.utils.math import quat_apply_inverse
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


def summarize(value):
    """Report distribution statistics without interpreting them as a gait score."""
    value = value.detach().float().reshape(-1)
    assert torch.isfinite(value).all()
    return {
        "count": value.numel(),
        "min": float(value.min()), "mean": float(value.mean()), "max": float(value.max()),
        "p05": float(torch.quantile(value, .05)), "p50": float(torch.quantile(value, .5)),
        "p95": float(torch.quantile(value, .95)),
    }


def measure_body_heights(env):
    """At the initial platform, origins.z is ground; heights are body-link origins."""
    robot = env.scene["robot"]
    positions = tensor(robot.data.body_pos_w)
    ground = env.scene.env_origins[:, 2]
    result = {}
    for name in ("pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"):
        index = robot.body_names.index(name)
        result[name] = summarize(positions[:, index, 2] - ground)
    result["torso_minus_pelvis_m"] = summarize(
        positions[:, robot.body_names.index("torso_link"), 2]
        - positions[:, robot.body_names.index("pelvis"), 2]
    )
    return result


def inspect_reference_dataset(env):
    """Full stored frames, unweighted by clip sampling probabilities; no retargeting."""
    robot = env.scene["robot"]
    dataset = env.animation_manager.get_term("animation").motion_data_term
    knees = [robot.joint_names.index(f"{side}_knee_joint") for side in ("left", "right")]
    gravity = torch.zeros_like(dataset.root_pos_w)
    gravity[:, 2] = -1.
    projected_gravity = quat_apply_inverse(dataset.root_quat, gravity)
    pelvis_tilt = torch.acos((-projected_gravity[:, 2]).clamp(-1., 1.))
    result = {
        "scope": "All original stored frames; frame-weighted statistics, not training sampling probabilities."
                 " Root Z is the dataset's pelvis height above its reference ground plane.",
        "root_height_m": summarize(dataset.root_pos_w[:, 2]),
        "pelvis_tilt_rad": summarize(pelvis_tilt),
        "left_knee_rad": summarize(dataset.dof_pos[:, knees[0]]),
        "right_knee_rad": summarize(dataset.dof_pos[:, knees[1]]),
        "clips": {},
    }
    for index, name in enumerate(dataset.motion_weights_dict):
        start = int(dataset.motion_start_indices[index])
        end = start + int(dataset.motion_num_frames[index])
        result["clips"][name] = {
            "sampling_probability": float(dataset.motion_weights[index]),
            "root_height_m": summarize(dataset.root_pos_w[start:end, 2]),
            "knees_rad": summarize(dataset.dof_pos[start:end, knees]),
        }
    return result


def check_spawn_safeguards_on_all_rows(env):
    """Test the real termination terms at each tile's spawn platform, without stepping."""
    terrain = env.scene.terrain
    ids = torch.arange(env.num_envs, device=env.device)
    height_cfg = env.termination_manager.get_term_cfg("base_height")
    tilt_cfg = env.termination_manager.get_term_cfg("excessive_body_tilt")
    assert height_cfg.params["minimum_height"] == .35
    assert tilt_cfg.params["limit_angle"] == 1.
    assert tilt_cfg.params["asset_cfg"].body_names == ["pelvis", "torso_link"]
    rows = []
    for level in range(terrain.terrain_origins.shape[0]):
        terrain.terrain_levels[:] = level
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        env._reset_idx(ids)
        env.sim.forward()
        env.scene.update(env.physics_dt)
        assert (terrain.terrain_levels == level).all()
        height_fail = height_cfg.func(env, **height_cfg.params)
        tilt_fail = tilt_cfg.func(env, **tilt_cfg.params)
        hits = tensor(env.scene[height_cfg.params["sensor_cfg"].name].data.ray_hits_w)[..., 2]
        valid = torch.isfinite(hits)
        assert valid.any(dim=1).all(), (level, "all height rays missed")
        ground = hits.masked_fill(~valid, 0.).sum(-1) / valid.sum(-1)
        height = tensor(env.scene["robot"].data.root_pos_w)[:, 2] - ground
        rows.append({
            "terrain_level": level,
            "pelvis_clearance_from_scanner_mean_m": height.tolist(),
            "height_termination": height_fail.tolist(),
            "body_tilt_termination": tilt_fail.tolist(),
        })
    terrain.terrain_levels[:] = 0
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env._reset_idx(ids)
    env.sim.forward()
    env.scene.update(env.physics_dt)
    assert not any(any(row["height_termination"]) or any(row["body_tilt_termination"]) for row in rows), rows
    return {
        "scope": "One random standing reset at each of 10 levels and six terrain types."
                 " Spawn platforms only; not tile transitions or learned stair-climbing behavior.",
        "terrain_order": env.command_manager.get_term("base_velocity").column_names,
        "rows": rows,
    }


def check_amp_profile(algorithm, observations, step_dt):
    """Exercise the actual discriminator reward interface on physical observations."""
    profile = algorithm.amp_cfg
    assert isinstance(algorithm.disc_optimizer, torch.optim.AdamW)
    for name, expected in {
        "disc_weight_l2_coef": 3e-4, "disc_logit_l2_coef": .04,
        "grad_penalty_data": "agent_demo", "disc_max_grad_norm": None,
        "disc_update_interval": 1,
    }.items():
        assert profile[name] == expected, (name, profile[name], expected)
    discriminator = algorithm.amp_discriminator
    assert isinstance(discriminator.disc_obs_normalizer, torch.nn.Identity)
    actual = discriminator.get_disc_obs(observations)
    torch.testing.assert_close(discriminator.normalize_disc_obs(actual), actual)
    reward, score = discriminator.predict_style_reward(actual, step_dt)
    raw = torch.clamp(1. - .25 * (score - 1.).square(), min=0.)
    torch.testing.assert_close(reward, .25 * raw)
    other_dt_reward, _ = discriminator.predict_style_reward(actual, step_dt * .5)
    torch.testing.assert_close(reward, other_dt_reward)
    task_reward = torch.linspace(-.1, .1, reward.numel(), device=reward.device)
    mixed = discriminator.lerp_reward(task_reward, reward)
    torch.testing.assert_close(mixed, task_reward + .25 * raw)
    return {
        "normalizer": type(discriminator.disc_obs_normalizer).__name__,
        "optimizer": type(algorithm.disc_optimizer).__name__,
        "gradient_penalty_data": profile["grad_penalty_data"],
        "equation": "mixed = task + 0.25 * clamp(1 - 0.25 * (D - 1)^2, min=0)",
        "style_reward_unchanged_when_dt_halved": True,
        "max_reward_formula_error": float((mixed - task_reward - .25 * raw).abs().max()),
        "untrained_agent_style_reward": summarize(reward),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/analysis/target_v6/physical.json"))
    parser.add_argument("--steps", type=int, default=50)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    task = "LeggedLab-Isaac-AMP-Depth-Target-G1-v2"
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
    generator.num_rows = 10
    generator.num_cols = 6
    generator.difficulty_range = (0., 1.)
    for sub_terrain in generator.sub_terrains.values():
        sub_terrain.proportion = 1.
    with launch_simulation(cfg, args):
        raw = gym.make(task, cfg=cfg).unwrapped
        try:
            observations, _ = raw.reset()
            check_observations(observations)
            ids = torch.arange(raw.num_envs, device=raw.device)
            reset_checks = [check_standing_reset(raw, ids)]
            standing_heights = measure_body_heights(raw)
            reference_dataset = inspect_reference_dataset(raw)
            spawn_safeguards = check_spawn_safeguards_on_all_rows(raw)
            observations = raw.observation_manager.compute(update_history=True)
            check_observations(observations)
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
            amp_profile = check_amp_profile(runner.alg, observations, raw.step_dt)
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
                "amp_profile": amp_profile,
                "standing_body_link_heights_m": standing_heights,
                "reference_dataset": reference_dataset,
                "spawn_safeguards_across_terrain_levels": spawn_safeguards,
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
