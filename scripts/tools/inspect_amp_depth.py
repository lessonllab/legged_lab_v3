"""Check the G1 depth sensor in simulation and save an actual camera frame."""

import argparse
import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import load_cfg_from_registry

import legged_lab.tasks  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="LeggedLab-Isaac-AMP-Depth-G1-Play-v0")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output_dir", type=Path, default=Path(".runtime/g1_depth_check"))
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("num_envs and steps must be positive")
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    # This small checker does not use Hydra's preset resolver like train.py does.
    cfg.sim.physics = copy.deepcopy(cfg.sim.physics.default)
    cfg.scene.num_envs = args.num_envs
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.terrain_generator.num_cols = 6
    multi_mesh = hasattr(cfg.scene.depth_camera, "update_mesh_ids")
    if multi_mesh:
        cfg.scene.depth_camera.update_mesh_ids = True
    cfg.seed = 42
    if args.device is not None:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        try:
            obs, _ = env.reset()
            shape = tuple(obs["depth"].shape)
            expected_hw = (18, 32) if "Instinct" in args.task else (36, 64)
            assert shape == (args.num_envs, 8, *expected_hw), shape
            assert torch.isfinite(obs["depth"]).all()
            sensor = env.unwrapped.scene["depth_camera"]
            raw = sensor.data.output["distance_to_image_plane"]
            if not isinstance(raw, torch.Tensor):
                raw = raw.torch
            initial = raw[0, ..., 0].clone().cpu().numpy()
            if multi_mesh:
                initial_mesh_positions = sensor._mesh_positions_w.numpy().copy()
            valid = np.isfinite(initial) & (initial >= 0.1)
            assert valid.any(), "Camera sees no valid terrain depth; check extrinsics and mesh paths."
            args.output_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.output_dir / "depth_metres.npy", initial)
            # A fixed metre scale preserves comparability between camera checks.
            from PIL import Image
            gray = np.where(valid, np.clip(np.nan_to_num(initial), 0.0, 2.5) / 2.5 * 254 + 1, 0)
            Image.fromarray(gray.astype(np.uint8)).resize((640, 360), Image.Resampling.NEAREST).save(
                args.output_dir / "depth.png"
            )
            with torch.inference_mode():
                max_pose_error = 0.0
                for _ in range(args.steps):
                    obs, reward, _, _, _ = env.step(torch.zeros_like(env.unwrapped.action_manager.action))
                    assert torch.isfinite(obs["depth"]).all()
                    assert torch.isfinite(reward).all()
                    if "Instinct" in args.task:
                        robot = env.unwrapped.scene["robot"]
                        positions = robot.data.body_pos_w.torch.cpu().numpy()
                        quaternions = robot.data.body_quat_w.torch.cpu().numpy()
                        mesh_positions = sensor._mesh_positions_w.numpy()
                        mesh_quaternions = sensor._mesh_orientations_w.numpy()
                        for index, target in enumerate(sensor._raycast_targets_cfg[1:], 1):
                            body = target.prim_expr.split("/Robot/")[1].split("/")[0]
                            body_id = robot.body_names.index(body)
                            error = float(np.linalg.norm(mesh_positions[:, index] - positions[:, body_id], axis=-1).max())
                            max_pose_error = max(max_pose_error, error)
                            assert error < 1e-5, (body, error)
                            mq, bq = mesh_quaternions[:, index], quaternions[:, body_id]
                            # PhysX quaternion norms can drift slightly; compare
                            # the actual poses modulo quaternion sign instead.
                            quat_error = np.minimum(np.linalg.norm(mq - bq, axis=-1), np.linalg.norm(mq + bq, axis=-1))
                            assert quat_error.max() < 1e-5, (body, quat_error.max())
                if multi_mesh:
                    final_mesh_positions = sensor._mesh_positions_w.numpy().copy()
                    tracked_motion = float(np.linalg.norm(
                        final_mesh_positions[:, 1:] - initial_mesh_positions[:, 1:], axis=-1
                    ).max())
                    assert tracked_motion > 1e-6, "Dynamic body mesh poses did not update"
                # A real reset must refill the observation history with the new episode.
                obs, _ = env.reset()
                torch.testing.assert_close(obs["depth"], obs["depth"][:, -1:].expand_as(obs["depth"]))
            summary = {
                "task": args.task, "depth_shape": shape, "checked_steps": args.steps,
                "valid_fraction_env0": float(valid.mean()),
                "depth_min_m": float(initial[valid].min()), "depth_max_m": float(initial[valid].max()),
                "history_reset": "passed", "image": "depth.png: black=invalid, white=2.5m or farther",
            }
            if multi_mesh:
                # Native camera exposes actual parsed targets and hit mesh IDs.
                summary["meshes_per_target"] = sensor._num_meshes_per_env
                assert sum(sensor._num_meshes_per_env.values()) > 1, "Robot meshes were not parsed"
                ids = sensor.data.image_mesh_ids.torch
                summary["unique_hit_mesh_ids"] = torch.unique(ids).cpu().tolist()
                summary["body_hit_fraction"] = float((ids > 0).float().mean())
                summary["max_tracked_body_motion_m"] = tracked_motion
                if "Instinct" in args.task:
                    summary["max_dynamic_body_pose_error_m"] = max_pose_error
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    main()
