"""Verify G1 depth target/body correspondence and single/multi-environment symmetry.

Does not train or modify the production camera pose. A second centered camera
is used only as a diagnostic. --legacy reproduces the former grouped-target bug.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab.sensors import MultiMeshRayCasterCameraCfg
from isaaclab.terrains import TerrainGeneratorCfg, MeshPlaneTerrainCfg
import legged_lab.tasks
from legged_lab.tasks.locomotion.amp.mdp.instinct_depth import preprocess_instinct_depth


def tensor(x):
    return x if isinstance(x, torch.Tensor) else x.torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--infinite_plane', action='store_true', help='Reproduce the backend huge-plane precision issue')
    parser.add_argument('--output_dir', type=Path, default=Path('docs/analysis/depth_fix_20260911'))
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error('--num_envs must be positive')
    task = 'LeggedLab-Isaac-AMP-Depth-Instinct-Warmup-G1-Play-v0'
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    cfg.sim.physics = copy.deepcopy(cfg.sim.physics.default)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    if args.infinite_plane:
        cfg.scene.terrain.terrain_type = 'plane'
        cfg.scene.terrain.terrain_generator = None
        cfg.scene.terrain.max_init_terrain_level = None
    if not args.infinite_plane:
        cfg.scene.terrain.terrain_type = 'generator'
        cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=42, size=(20., 20.), num_rows=1, num_cols=args.num_envs,
            border_width=0., curriculum=False,
            sub_terrains={'flat': MeshPlaneTerrainCfg(proportion=1.)})
        cfg.scene.terrain.max_init_terrain_level = 0
    if args.legacy:
        cfg.scene.depth_camera.mesh_prim_paths = ['/World/ground', MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
            prim_expr='{ENV_REGEX_NS}/Robot/[^/]*/visuals', track_mesh_transforms=True)]
    cfg.scene.depth_camera.update_mesh_ids = True
    cfg.scene.center_camera = copy.deepcopy(cfg.scene.depth_camera)
    cfg.scene.center_camera.offset.pos = (.0487988662332928, 0., .4378029937970051)
    cfg.scene.center_camera.offset.rot = (0., math.sin(math.radians(24)), 0., math.cos(math.radians(24)))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset()
            robot = env.scene['robot']
            base = tensor(robot.data.default_root_state).clone()
            base[:, :3] += env.scene.env_origins
            base[:, 3:7] = torch.tensor([0., 0., 0., 1.], device=env.device)
            base[:, 7:] = 0
            results, panels = {}, []
            for pose in ['symmetric_default', 'symmetric_forward', 'left_forward', 'right_forward']:
                q = tensor(robot.data.default_joint_pos).clone()
                for side in ['left', 'right']:
                    if pose == 'symmetric_forward' or pose.startswith(side):
                        q[:, robot.joint_names.index(side + '_shoulder_pitch_joint')] = -.7
                        q[:, robot.joint_names.index(side + '_elbow_joint')] = .97
                robot.write_root_state_to_sim(base)
                robot.write_joint_state_to_sim(q, torch.zeros_like(q))
                robot.set_joint_position_target(q)
                env.scene.write_data_to_sim()
                # Compare exact kinematic poses, excluding reset-dependent contact
                # warm starts and delayed actuator history from this geometry test.
                env.sim.forward()
                env.scene.update(env.physics_dt)
                results[pose] = {}
                for name in ['depth_camera', 'center_camera']:
                    sensor = env.scene[name]
                    sensor.update(env.step_dt, force_recompute=True)
                    raw = tensor(sensor.data.output['distance_to_image_plane']).clone()
                    ids_all = tensor(sensor.data.image_mesh_ids)[..., 0].cpu().numpy()
                    ids = ids_all[0]
                    crop = preprocess_instinct_depth(raw)[0].cpu().numpy()
                    depth_all = raw[..., 0].cpu().numpy()
                    depth = depth_all[0]
                    labels, max_pose_error = {}, 0.
                    if not args.legacy:
                        bodypos = tensor(robot.data.body_pos_w).cpu().numpy()
                        bodyquat = tensor(robot.data.body_quat_w).cpu().numpy()
                        meshpos = sensor._mesh_positions_w.numpy()
                        meshquat = sensor._mesh_orientations_w.numpy()
                        for index, target in enumerate(sensor._raycast_targets_cfg[1:], 1):
                            body = target.prim_expr.split('/Robot/')[1].split('/')[0]
                            labels[index] = body
                            body_id = robot.body_names.index(body)
                            assert sensor._num_meshes_per_env[target.prim_expr] == 1
                            expected_paths = [f'{p}/Robot/{body}' for p in env.scene.env_prim_paths]
                            assert list(sensor._mesh_views[index].prim_paths) == expected_paths
                            error = float(np.linalg.norm(meshpos[:, index] - bodypos[:, body_id], axis=-1).max())
                            max_pose_error = max(max_pose_error, error)
                            assert error < 1e-5, (body, error)
                            np.testing.assert_allclose(abs((meshquat[:, index] * bodyquat[:, body_id]).sum(-1)), 1., atol=1e-5)
                    def counts(pixels):
                        return {side: int(sum((pixels == i).sum() for i, label in labels.items() if label.startswith(side)))
                                for side in ('left', 'right')}
                    stats = dict(body_pixels=int((ids > 0).sum()), cropped_arm_side_pixels=counts(ids[18:, 16:-16]),
                        raw_mirror_mae_m=float(np.mean(abs(depth - depth[:, ::-1]))),
                        near_body_pixels=int(((depth > 0) & (depth < .1) & (ids > 0)).sum()),
                        max_pose_match_error_m=max_pose_error,
                        max_cross_env_depth_difference_m=float(abs(depth_all - depth_all[:1]).max()),
                        raw_min_m=float(depth.min()), raw_max_m=float(depth.max()))
                    # A ray exactly on a silhouette may change target after a
                    # float32 world translation. Check continuous surfaces
                    # separately and bound such changes to silhouette pixels.
                    same_hit = ids_all == ids_all[:1]
                    edge = np.zeros_like(ids, dtype=bool)
                    edge[1:] |= ids[1:] != ids[:-1]
                    edge[:-1] |= ids[:-1] != ids[1:]
                    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
                    edge[:, :-1] |= ids[:, :-1] != ids[:, 1:]
                    stats['max_same_surface_depth_difference_m'] = float(abs(depth_all - depth_all[:1])[same_hit].max())
                    stats['max_changed_silhouette_pixels_per_env'] = int((~same_hit).sum(axis=(1, 2)).max())
                    results[pose][name] = stats
                    np.savez(out / 'pose_debug.npz', depth=depth_all, camera_pos=tensor(sensor.data.pos_w).cpu().numpy(), camera_quat=tensor(sensor.data.quat_w_world).cpu().numpy(), origins=env.scene.env_origins.cpu().numpy(), bodypos=tensor(robot.data.body_pos_w).cpu().numpy())
                    if not args.legacy:
                        assert stats['max_same_surface_depth_difference_m'] < .002, stats
                        assert stats['max_changed_silhouette_pixels_per_env'] <= 4, stats
                        assert not ((~same_hit) & (~edge)).any(), stats
                        assert stats['near_body_pixels'] == 0, stats
                        assert depth.max() > .5
                        if name == 'center_camera' and pose == 'symmetric_forward':
                            c = stats['cropped_arm_side_pixels']
                            assert c['left'] > 0 and c['right'] > 0 and abs(c['left'] - c['right']) <= 2, c
                        if name == 'center_camera' and pose in ('left_forward', 'right_forward'):
                            side = pose.split('_')[0]
                            c = stats['cropped_arm_side_pixels']
                            assert c[side] > 0 and c['right' if side == 'left' else 'left'] == 0, c
                    np.savez(out / f'{pose}_{name}.npz', depth=depth_all, ids=ids, crop=crop)
                    color = np.full((*ids.shape, 3), 180, dtype=np.uint8)
                    for i, label in labels.items():
                        color[ids == i] = (220, 70, 50) if label.startswith('left') else (40, 100, 230) if label.startswith('right') else (70, 180, 80)
                    row = Image.new('RGB', (768, 180), 'white')
                    ImageDraw.Draw(row).text((4, 2), pose + ' / ' + name, fill='black')
                    images = [(np.clip(depth / 2.5, 0, 1) * 255).astype('uint8'), color, (crop * 255).astype('uint8')]
                    for j, img in enumerate(images):
                        row.paste(Image.fromarray(img).resize((256, 144), Image.Resampling.NEAREST), (j * 256, 28))
                    panels.append(row)
            panel = Image.new('RGB', (768, 180 * len(panels)), 'white')
            for i, row in enumerate(panels):
                panel.paste(row, (0, 180 * i))
            panel.save(out / 'comparison.png')
            panel.crop((0, 360, 768, 720)).save(out / 'symmetric_arms.png')
            (out / 'results.json').write_text(json.dumps(dict(num_envs=args.num_envs, legacy=args.legacy, poses=results), indent=2))
            print('PASS' if not args.legacy else 'LEGACY', json.dumps(results), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
