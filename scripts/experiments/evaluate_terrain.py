"""Fixed-route inference evaluation of the saved rough-terrain G1 actor.

Uses the configured Isaac Lab interpreter. No RSL-RL runner or training is used.
The compatibility aliases only redirect the legacy velocity MDP import path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import torch
import trimesh

import isaaclab_tasks.core.velocity.mdp as velocity_mdp

for name in (
    "isaaclab_tasks.manager_based",
    "isaaclab_tasks.manager_based.locomotion",
    "isaaclab_tasks.manager_based.locomotion.velocity",
):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules.setdefault(name, module)
sys.modules.setdefault("isaaclab_tasks.manager_based.locomotion.velocity.mdp", velocity_mdp)

import legged_lab.tasks  # noqa: E402,F401
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.managers import EventTermCfg  # noqa: E402
from isaaclab.envs.mdp import UniformVelocityCommandCfg  # noqa: E402
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def course_terrain(difficulty, cfg):
    """12 m tile: spawn x=2, obstacle starts x=4, goal x=8 (6 m travel)."""
    width = cfg.size[1]
    if cfg.kind == "slope":
        top = math.tan(math.radians(cfg.slope_deg)) * 3.0
        points = [(0, 0), (4, 0), (7, top), (12, top)]
    elif cfg.kind == "stairs":
        points = [(0, 0), (4, 0)]
        for i in range(4):
            x = 4 + .4 * i
            if i:
                points.append((x, cfg.step_height * i))
            points.append((x, cfg.step_height * (i + 1)))
        points.append((12, cfg.step_height * 4))
    else:
        points = [(0, 0), (12, 0)]
    vertices = [(x, y, z) for x, z in points for y in (0, width)]
    faces = []
    for i in range(len(points) - 1):
        a = 2 * i
        faces.extend([(a, a + 2, a + 1), (a + 1, a + 2, a + 3)])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return [mesh], np.array([2., width / 2, 0.])


@configclass
class CourseTerrainCfg(SubTerrainBaseCfg):
    function = course_terrain
    kind: str = "flat"
    slope_deg: float = 8.
    step_height: float = .08


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def reset_fixed(env, env_ids):
    robot = env.scene["robot"]
    state = tensor(robot.data.default_root_state)[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]
    yaw = (torch.rand(len(env_ids), device=env.device) - .5) * .10
    state[:, 3:7] = 0
    # Isaac Lab v3 uses XYZW, including the legacy write_root_pose_to_sim API.
    state[:, 5] = torch.sin(yaw / 2)
    state[:, 6] = torch.cos(yaw / 2)
    state[:, 7:] = 0
    robot.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)
    q = tensor(robot.data.default_joint_pos)[env_ids].clone()
    q += (torch.rand_like(q) - .5) * .02
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", choices=["flat", "slope", "stairs"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-reset", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-env", type=int, default=0)
    parser.add_argument("--slope-deg", type=float, default=8.)
    parser.add_argument("--step-height", type=float, default=.08)
    parser.add_argument("--continue-after-corridor", action="store_true")
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--friction", type=float, default=.8)
    parser.add_argument("--heading-feedback", action="store_true")
    parser.add_argument("--telemetry", action="store_true")
    args = parser.parse_args()
    assert 0 <= args.video_env < args.num_envs
    assert 0 <= args.slope_deg < 60 and 0 < args.step_height <= .5
    assert 0 < args.speed <= 3 and 0 < args.friction <= 2
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg("LeggedLab-Isaac-AMP-Rough-G1-Play-v0", device="cuda:0", num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.episode_length_s = max(cfg.episode_length_s, args.steps * cfg.sim.dt * cfg.decimation + 1.)
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=args.seed, size=(12., 4.), border_width=0., num_rows=1, num_cols=args.num_envs,
        curriculum=False, use_cache=False,
        sub_terrains={args.terrain: CourseTerrainCfg(kind=args.terrain, proportion=1.,
                        slope_deg=args.slope_deg, step_height=args.step_height)},
    )
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.scene.height_scanner.debug_vis = False
    cfg.curriculum.terrain_levels = None
    cfg.events.push_robot = None
    cfg.events.add_base_mass = None
    cfg.events.physics_material.params["static_friction_range"] = (args.friction, args.friction)
    cfg.events.physics_material.params["dynamic_friction_range"] = (args.friction, args.friction)
    if not args.reference_reset:
        cfg.events.reset_from_ref = EventTermCfg(func=reset_fixed, mode="reset")
    cfg.commands.base_velocity = UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(1000., 1000.), rel_standing_envs=0.,
        heading_command=args.heading_feedback, rel_heading_envs=1., heading_control_stiffness=1., debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(args.speed, args.speed), lin_vel_y=(0., 0.),
            ang_vel_z=(-1.5, 1.5) if args.heading_feedback else (0., 0.),
            heading=(0., 0.) if args.heading_feedback else None),
    )
    cfg.observations.policy.enable_corruption = False
    if args.video:
        from isaaclab_visualizers.kit import KitVisualizerCfg
        cfg.sim.visualizer_cfgs = [KitVisualizerCfg(
            headless=True, window_width=960, window_height=640,
            origin_type="asset", origin_track_path="robot/torso_link",
            origin_env_index=args.video_env,
            eye=(2., -3., 1.5), lookat=(0., 0., 0.),
        )]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = torch.nn.Sequential(
        torch.nn.Linear(682, 512), torch.nn.ELU(), torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 29),
    ).cuda().eval()
    actor.load_state_dict({k.removeprefix("mlp."): v for k, v in checkpoint["actor_state_dict"].items()
                           if k.startswith("mlp.")}, strict=True)
    metadata = dict(terrain=args.terrain, seed=args.seed, num_envs=args.num_envs, steps=args.steps,
                    checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    speed_m_s=args.speed, goal_x_m=6., corridor_half_width_m=.8,
                    robot_friction=args.friction, heading_feedback=args.heading_feedback,
                    heading_control=("wz=clip(wrap(-heading), -1.5, 1.5)" if args.heading_feedback else "wz=0"),
                    terrain_material=cfg.scene.terrain.physics_material.to_dict(),
                    episode_length_s=cfg.episode_length_s, telemetry=args.telemetry,
                    continue_after_corridor=args.continue_after_corridor,
                    containment_half_width_m=1.5, slope_deg=args.slope_deg,
                    step_height_m=args.step_height, video_env=args.video_env,
                    initialization=("original reference reset" if args.reference_reset else
                                    "default standing pose; yaw +/-0.05 rad; joint position +/-0.01 rad; zero velocity"),
                    physics="current installed Isaac Lab PhysX; legacy MDP import aliased; no training",
                    observation="rough actor's 682 channels including height scanner; noise disabled")
    metadata["evaluator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata["crash_reporting"] = "disabled; no crash reporter plugin; old dump upload disabled"
    metadata["terrain_geometry"] = {"flat": "flat ground", "slope": f"{args.slope_deg} degrees up; x=2..5 m from start",
                                    "stairs": f"4 risers of {args.step_height} m; tread 0.4 m; first riser x=2 m from start"}[args.terrain]
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2))
    from isaaclab.app import AppLauncher
    # Installed AppLauncher omits this documented SimulationApp option from its
    # forwarding allowlist. Extend only this process, never the installed package.
    AppLauncher._SIM_APP_CFG_TYPES["enable_crashreporter"] = [bool]
    launch_options = {"headless": True, "device": "cuda:0", "enable_cameras": args.video,
                      "enable_crashreporter": False,
                      "kit_args": "--/crashreporter/enabled=false --/crashreporter/skipOldDumpUpload=true "
                                  "--/crashreporter/devOnlyOverridePrivacyAndForceUpload=false "
                                  "--/privacy/extraDiagnosticDataOptIn=false"}
    with launch_simulation(cfg, launch_options):
        import gymnasium as gym
        if args.heading_feedback:
            # Import runtime command code only after the simulator is initialized.
            from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

            class ResetConsistentVelocityCommand(UniformVelocityCommand):
                """Apply feedback before reset observations enter the policy history."""

                def _resample_command(self, env_ids):
                    super()._resample_command(env_ids)
                    self._update_command()

            cfg.commands.base_velocity.class_type = ResetConsistentVelocityCommand
        env = gym.make("LeggedLab-Isaac-AMP-Rough-G1-Play-v0", cfg=cfg).unwrapped
        writer = None
        try:
            obs, _ = env.reset(seed=args.seed)
            print("OBSERVATIONS", {k: list(v.shape) for k, v in obs.items()}, flush=True)
            from isaaclab.utils.math import quat_apply
            robot = env.scene["robot"]
            up = torch.zeros((args.num_envs, 3), device=env.device)
            up[:, 2] = 1
            if not args.reference_reset:
                assert bool((quat_apply(tensor(robot.data.root_quat_w), up)[:, 2] > .95).all()), "Invalid initial orientation"
                feet = tensor(robot.data.body_pos_w)[:, [robot.body_names.index("left_ankle_roll_link"), robot.body_names.index("right_ankle_roll_link")], 2]
                assert bool((feet < tensor(robot.data.root_pos_w)[:, 2:3]).all()), "Feet must start below the pelvis"
            assert obs["policy"].shape == (args.num_envs, 682)
            assert bool(torch.isfinite(obs["policy"]).all()), "Nonfinite policy observations"
            initial = dict(root_state=tensor(robot.data.root_state_w).cpu().tolist(),
                           joint_pos=tensor(robot.data.joint_pos).cpu().tolist(),
                           origins=env.scene.env_origins.cpu().tolist())
            (args.output / "initial_state.json").write_text(json.dumps(initial))
            active = np.ones(args.num_envs, dtype=bool)
            departures = [None] * args.num_envs
            records = [None] * args.num_envs
            trajectory = []
            diagnostics = []
            telemetry = []
            foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
            foot_ids = [robot.body_names.index(n) for n in foot_names]
            sensor = env.scene["contact_forces"]
            contact_ids = [sensor.body_names.index(n) for n in foot_names]
            if args.telemetry:
                import warp as wp
                materials = wp.to_torch(robot.root_view.get_material_properties()).cpu().numpy()
                np.save(args.output / "robot_materials.npy", materials)
                assert np.allclose(materials[..., :2], args.friction), "Robot friction differs from protocol"
                (args.output / "telemetry_schema.json").write_text(json.dumps(dict(
                    dt_s=env.step_dt, sample_phase="before each control step; only active mask entries are scored",
                    feet=foot_names, joints=robot.joint_names,
                    root_quat_convention="XYZW", foot_velocity="ankle link COM velocity, not sole contact-point slip",
                    torque="simulator applied_torque; actuator estimate for implicit actuators",
                    foot_force="current PhysX backend returns net NORMAL world-frame contact force only; last physics sample"), indent=2))
            if args.video:
                import imageio.v2 as imageio
                writer = imageio.get_writer(str(args.output / f"env{args.video_env}.mp4"), fps=50, macro_block_size=1)
            for step in range(args.steps):
                if writer is not None and active[args.video_env]:
                    writer.append_data(env.sim.visualizers[0].render_rgb_array())
                before = tensor(env.scene["robot"].data.root_pos_w).clone() - env.scene.env_origins
                velocity = tensor(env.scene["robot"].data.root_lin_vel_b).clone()
                if args.telemetry:
                    if args.heading_feedback:
                        command = env.command_manager.get_command("base_velocity")
                        expected = -tensor(robot.data.heading_w)
                        expected = torch.atan2(torch.sin(expected), torch.cos(expected)).clamp(-1.5, 1.5)
                        assert torch.allclose(command[:, 2], expected, atol=2e-6), "Heading command is stale"
                    packet = dict(active=active.copy(), root_pos=before, root_quat=tensor(robot.data.root_quat_w),
                        root_vel_w=tensor(robot.data.root_lin_vel_w), heading=tensor(robot.data.heading_w),
                        command=env.command_manager.get_command("base_velocity"),
                        foot_pos=tensor(robot.data.body_pos_w)[:, foot_ids] - env.scene.env_origins[:, None, :],
                        foot_vel=tensor(robot.data.body_lin_vel_w)[:, foot_ids],
                        foot_force=tensor(sensor.data.net_forces_w)[:, contact_ids],
                        torque=tensor(robot.data.applied_torque), effort_limit=tensor(robot.data.joint_effort_limits))
                    telemetry.append({k: v.detach().cpu().numpy().copy() if isinstance(v, torch.Tensor) else v
                                      for k, v in packet.items()})
                with torch.inference_mode():
                    action = actor(obs["policy"])
                    if step < 30:
                        r = env.scene["robot"]
                        diagnostics.append(dict(step=step, action=action[0].cpu().tolist(),
                            q=tensor(r.data.joint_pos)[0].cpu().tolist(),
                            torque=tensor(r.data.applied_torque)[0].cpu().tolist(),
                            feet_z=tensor(r.data.body_pos_w)[0, [r.body_names.index("left_ankle_roll_link"), r.body_names.index("right_ankle_roll_link")], 2].cpu().tolist(),
                            ray_z=tensor(env.scene["height_scanner"].data.ray_hits_w)[0, :, 2].cpu().tolist()))
                    obs, reward, terminated, truncated, extras = env.step(action)
                pos = (tensor(env.scene["robot"].data.root_pos_w) - env.scene.env_origins).cpu().numpy()
                done = (terminated | truncated).cpu().numpy()
                prior = before.cpu().numpy()
                for i in np.flatnonzero(active):
                    p = prior[i] if done[i] else pos[i]
                    trajectory.append([step, int(i), *p.tolist(), *velocity[i].cpu().tolist()])
                    if departures[i] is None and abs(p[1]) > .8:
                        departures[i] = dict(time_s=(step + 1) * env.step_dt, position_m=p.tolist())
                    reason = None
                    if done[i]:
                        reason = "terminated" if bool(terminated[i]) else "timeout"
                    elif args.continue_after_corridor and (abs(p[1]) > 1.5 or p[0] < -.5):
                        reason = "left_containment"
                    elif not args.continue_after_corridor and (abs(p[1]) > .8 or p[0] < -.5):
                        reason = "left_corridor"
                    elif p[0] >= 6.:
                        reason = "success"
                    if reason:
                        active[i] = False
                        records[i] = dict(env=int(i), outcome=reason, time_s=(step + 1) * env.step_dt,
                                          position_m=p.tolist(), position_is_pre_reset=bool(done[i]),
                                          termination_terms=[name for name in env.termination_manager.active_terms
                                              if done[i] and bool(env.termination_manager.get_term(name)[i])])
                if step % 100 == 0:
                    print("PROGRESS", step, "active", int(active.sum()), "env0", pos[0].tolist(), flush=True)
                if not active.any():
                    break
            for i in np.flatnonzero(active):
                records[i] = dict(env=int(i), outcome="horizon_reached", time_s=args.steps * env.step_dt,
                                  position_m=pos[i].tolist(), position_is_pre_reset=False)
            for i, record in enumerate(records):
                record["first_corridor_departure"] = departures[i]
                record["route_success"] = record["outcome"] == "success" and departures[i] is None
            summary = dict(protocol=metadata, trials=records,
                           successes=sum(r["outcome"] == "success" for r in records),
                           route_successes=sum(r["route_success"] for r in records),
                           total=args.num_envs, complete_horizon=args.steps >= 700,
                           valid_evaluation=True)
            (args.output / "results.json").write_text(json.dumps(summary, indent=2))
            np.savetxt(args.output / "trajectory.csv", np.asarray(trajectory), delimiter=",",
                       header="step,env,x_m,y_m,z_m,vx_body_m_s,vy_body_m_s,vz_body_m_s", comments="")
            (args.output / "diagnostics.json").write_text(json.dumps(diagnostics))
            if args.telemetry:
                np.savez_compressed(args.output / "telemetry.npz",
                                    **{k: np.stack([p[k] for p in telemetry]) for k in telemetry[0]})
            print("RESULT", json.dumps(summary), flush=True)
        finally:
            if writer is not None:
                writer.close()
            env.close()


if __name__ == "__main__":
    main()
