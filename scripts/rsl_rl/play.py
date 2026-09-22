"""Script to play a checkpoint of an RL agent trained with RSL-RL.

Examples:
    # Record a video (Kit visualizer)
    python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Rough-G1-Play-v0 \
        --num_envs 64 --video --viz kit \
        --checkpoint logs/rsl_rl/g1_amp_rough/run_name/model_xxx.pt

    # Follow the robot with a smooth position-only chase camera (Kit only)
    python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Rough-G1-Play-v0 \
        --viz kit --checkpoint .../model_xxx.pt --follow_cam

    # Follow AND rotate with the robot's heading, damping the yaw jitter
    python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Rough-G1-Play-v0 \
        --viz kit --checkpoint .../model_xxx.pt --follow_cam --follow_yaw --follow_smooth 0.9

Note:
    The follow camera (--follow_cam) drives the Kit viewport only; the Viser backend
    manages its own camera client-side and ignores it.
"""

import warnings

warnings.warn(
    "scripts/rsl_rl/play.py is deprecated. Use "
    "`./isaaclab.sh play --rl_library rsl_rl --task <TASK>` instead.",
    DeprecationWarning,
    stacklevel=1,
)

import argparse
import contextlib
import importlib.metadata as metadata
import math
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, setup_preset_cli

try:
    from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
except ModuleNotFoundError as exc:
    if exc.name != "isaaclab.app.sim_launcher":
        raise
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

import legged_lab.tasks  # noqa: F401
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a trained RSL-RL policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--stepping_stones", action="store_true", help="Add AME-style stepping stones to G1 target terrains.")
parser.add_argument("--terrain_showcase", action="store_true", help="One robot per terrain type at fixed moderate difficulty (playback only).")
parser.add_argument("--terrain_control", action="store_true", help="One robot with interactive terrain difficulty controls.")
parser.add_argument("--stair_matrix", action="store_true", help="One robot per tile; 17 steps and stair heights 8–30 cm.")
parser.add_argument("--stair_test_speed", type=float, default=.8, help="Fixed speed cap for the stair matrix (m/s).")
parser.add_argument("--instinct_play", action="store_true", help="Target task: standing resets, 10-second episodes and close follow view, inspired by Instinct PLAY.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus."
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop playback after this many steps; 0 runs until interrupted.")
parser.add_argument("--viser_port", type=int, default=8080, help="PhysX browser viewer port.")
parser.add_argument("--viser_host", default="127.0.0.1", help="PhysX browser viewer bind address; use 0.0.0.0 for remote access.")
parser.add_argument(
    "--follow_cam",
    action="store_true",
    default=False,
    help="Make the viewport camera follow the robot each step (third-person chase view).",
)
parser.add_argument(
    "--follow_env", type=int, default=0, help="Environment index to follow when --follow_cam is set."
)
parser.add_argument(
    "--follow_body",
    type=str,
    default="torso_link",
    help="Body name to follow when --follow_cam is set (e.g. torso_link, pelvis).",
)
parser.add_argument(
    "--follow_offset",
    type=float,
    nargs=3,
    default=(0.0, -3.0, 1.5),
    metavar=("X", "Y", "Z"),
    help=(
        "Camera eye offset (m) relative to the followed body. Without --follow_yaw this is a "
        "fixed world-axis offset (camera keeps a constant viewing direction, only tracking the "
        "robot's position). With --follow_yaw it is in the robot's own frame (+X forward, +Y "
        "left, +Z up) and rotates with the robot's heading. Default (0, -3, 1.5)."
    ),
)
parser.add_argument(
    "--follow_yaw",
    action="store_true",
    default=False,
    help=(
        "Rotate the camera with the robot's heading (body-frame chase view). Off by default, "
        "since the robot's yaw changes every step and rotating with it makes the view shaky; "
        "leave off for a smooth position-only follow. Use --follow_smooth to damp the shake."
    ),
)
parser.add_argument(
    "--follow_smooth",
    type=float,
    default=0.0,
    metavar="ALPHA",
    help=(
        "Low-pass smoothing for --follow_yaw, in [0, 1). 0 = no smoothing (raw yaw each step); "
        "higher = smoother but laggier (e.g. 0.9). Only used when --follow_yaw is set."
    ),
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True

remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # Use our read-only PhysX -> Viser bridge instead of the installed
    # Newton-specific visualizer. Physics and policy inputs remain unchanged.
    physics = getattr(env_cfg.sim.physics, "default", env_cfg.sim.physics)
    physics_class = str(getattr(physics, "class_type", "")).lower()
    physics_override = getattr(args_cli, "physics", None)
    if physics_override:
        physics_class = str(physics_override).lower()
    requested = args_cli.visualizer
    use_physx_viser = bool(requested and "viser" in requested and "physx" in physics_class)
    if use_physx_viser:
        args_cli.visualizer = [name for name in requested if name != "viser"] or ["none"]
        print("[INFO] PhysX playback: using the project's Viser browser bridge (no Newton).", flush=True)
    if args_cli.max_steps < 0:
        raise ValueError("--max_steps must be non-negative")
    with launch_simulation(env_cfg, args_cli):
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.stair_matrix or args_cli.terrain_control:
            if args_cli.task != 'LeggedLab-Isaac-AMP-Stairs-Long-G1-Play-v0':
                raise ValueError('--stair_matrix requires the long-stair PLAY task')
            if args_cli.terrain_showcase or args_cli.instinct_play or args_cli.stepping_stones:
                raise ValueError('Stair matrix cannot be combined with other terrain showcases')
            from legged_lab.tasks.locomotion.amp.mdp.stair_play_matrix import configure_stair_matrix
            if args_cli.stair_matrix and args_cli.terrain_control:
                raise ValueError('Choose either matrix or interactive terrain control')
            if args_cli.terrain_control:
                from legged_lab.tasks.locomotion.amp.mdp.stair_play_matrix import configure_stair_control
                configure_stair_control(env_cfg, args_cli.stair_test_speed)
            else:
                configure_stair_matrix(env_cfg, args_cli.stair_test_speed)
        if args_cli.stepping_stones:
            from legged_lab.tasks.locomotion.amp.config.g1.terrain_variants import add_stepping_stones
            add_stepping_stones(env_cfg)
            print("[PLAY] Added AME-style stepping stones; old checkpoints have not necessarily learned this terrain.", flush=True)
        if args_cli.instinct_play:
            if args_cli.terrain_showcase:
                raise ValueError("Choose --instinct_play or --terrain_showcase")
            if "Depth-Target-G1" not in args_cli.task:
                raise ValueError("--instinct_play requires the G1 Depth-Target task")
            from isaaclab.managers import EventTermCfg
            from isaaclab.envs import mdp as base_mdp
            generator = env_cfg.scene.terrain.terrain_generator
            for subterrain in generator.sub_terrains.values():
                subterrain.proportion = 1.0
            generator.curriculum = True
            generator.num_rows = 4
            generator.num_cols = len(generator.sub_terrains)
            env_cfg.scene.num_envs = generator.num_cols
            env_cfg.scene.terrain.max_init_terrain_level = 3
            env_cfg.curriculum.terrain_levels = None
            env_cfg.episode_length_s = 10.
            env_cfg.events.reset_from_ref = None
            # v5 already defines standing reset events. Replace them instead
            # of running two root/joint reset terms on every episode boundary.
            if hasattr(env_cfg.events, "reset_base"):
                env_cfg.events.reset_base = None
            if hasattr(env_cfg.events, "reset_robot_joints"):
                env_cfg.events.reset_robot_joints = None
            env_cfg.events.play_reset_base = EventTermCfg(
                func=base_mdp.reset_root_state_uniform, mode="reset", params={
                    "pose_range": {"x": (-.1, .1), "y": (-.1, .1), "yaw": (-.1, .1)},
                    "velocity_range": {axis: (-.2, .2) for axis in ("x", "y", "z", "roll", "pitch", "yaw")}})
            env_cfg.events.play_reset_joints = EventTermCfg(
                func=base_mdp.reset_joints_by_offset, mode="reset",
                params={"position_range": (0., 0.), "velocity_range": (0., 0.)})
            args_cli.follow_cam = True
            print("[PLAY] Instinct-style standing reset; 10 s episodes; 4 difficulty rows; fall terminations retained.", flush=True)
        if args_cli.terrain_showcase:
            generator = env_cfg.scene.terrain.terrain_generator
            if generator is None:
                raise ValueError("Terrain showcase requires a generated terrain task")
            # Equal proportions and ordered columns guarantee all types appear.
            for subterrain in generator.sub_terrains.values():
                subterrain.proportion = 1.0
            generator.curriculum = True
            generator.num_rows = 1
            generator.num_cols = len(generator.sub_terrains)
            generator.difficulty_range = (0.25, 0.25)
            env_cfg.scene.num_envs = generator.num_cols
            env_cfg.scene.terrain.max_init_terrain_level = 0
            env_cfg.curriculum.terrain_levels = None
            print("[INFO] Terrain showcase robot columns:", list(enumerate(generator.sub_terrains)), flush=True)

        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")

        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        env.unwrapped.terrain_control_play = args_cli.terrain_control
        env.unwrapped.stair_matrix_play = args_cli.stair_matrix
        if args_cli.stair_matrix:
            env.unwrapped.terrain_showcase_names = env.unwrapped.command_manager.get_term('base_velocity').matrix_labels()
        if args_cli.terrain_showcase or args_cli.instinct_play:
            env.unwrapped.terrain_showcase_names = list(env_cfg.scene.terrain.terrain_generator.sub_terrains)
        env.unwrapped.instinct_play = args_cli.instinct_play
        env.unwrapped.play_checkpoint_label = os.path.basename(os.path.dirname(resume_path)) + ' / ' + os.path.basename(resume_path)

        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent
            env = multi_agent_to_single_agent(env)

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during play.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)

        if args_cli.stair_matrix or args_cli.terrain_control:
            # Evaluation owns the fixed rows/columns. Load network weights
            # strictly, but do not overwrite this layout with training groups.
            from legged_lab.rsl_rl.amp.ppo_amp import PPOAMP
            saved = torch.load(resume_path, map_location=agent_cfg.device, weights_only=False)
            PPOAMP.load(runner.alg, saved, {'actor': True, 'critic': True, 'optimizer': False}, strict=True)
            for name, value in runner.alg.get_policy().state_dict().items():
                if not torch.equal(value, saved['actor_state_dict'][name].to(value.device)):
                    raise RuntimeError(f'Matrix playback policy was not restored exactly: {name}')
            env.unwrapped.reset()
            print(f'[PLAY] {env.num_envs} robots; 17 steps; heights 8/10/12/14/16/18/20/25/30 cm; '
                  f'speed cap {args_cli.stair_test_speed} m/s; frozen layout, model weights only', flush=True)
        else:
            runner.load(resume_path, map_location=agent_cfg.device)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        if version.parse(installed_version) >= version.parse("4.0.0"):
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None
        else:
            policy_nn = runner.alg.policy if version.parse(installed_version) >= version.parse("2.3.0") else runner.alg.actor_critic
            normalizer = getattr(policy_nn, "actor_obs_normalizer", None) or getattr(policy_nn, "student_obs_normalizer", None)
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt
        obs = env.get_observations()
        timestep = 0
        browser_viewer = None
        if use_physx_viser:
            from physx_viser import PhysxViser
            browser_viewer = PhysxViser(env.unwrapped, host=args_cli.viser_host,
                                       port=args_cli.viser_port, follow=args_cli.follow_cam,
                                       selected_env=args_cli.follow_env)
            browser_viewer.update(obs)

        # -- follow camera -------------------------------------------------------
        # Manually drive the viewport camera to chase the robot each step. We do this
        # here (rather than via env_cfg.viewer's origin_type="asset_root") because the
        # ViewerCfg tracking callback fires before the articulation's physics view is
        # initialized under some visualizers, crashing on a None root_view. By the time
        # we reach this loop the robot is fully initialized, so reading its body pose and
        # calling sim.set_camera_view is safe.
        #
        # Two modes:
        #   default (--follow_yaw off): the offset is a fixed world-axis vector, so the
        #     camera keeps a constant viewing direction and only tracks the robot's
        #     position — smooth, no shake.
        #   --follow_yaw on: the offset is in the robot's frame and rotated by its heading
        #     each step (chase view). Because the robot's yaw is noisy, --follow_smooth
        #     applies a circular low-pass filter to the heading to damp the shake.
        follow_robot = None
        follow_body_id = None
        follow_offset = None
        follow_yaw_state = None  # smoothed heading (radians), lazily initialized on first step
        if args_cli.follow_cam and not use_physx_viser:
            if not 0.0 <= args_cli.follow_smooth < 1.0:
                raise ValueError(
                    f"--follow_smooth must be in [0, 1), got {args_cli.follow_smooth}."
                )
            follow_robot = env.unwrapped.scene["robot"]
            body_ids, body_names_found = follow_robot.find_bodies(args_cli.follow_body)
            if len(body_ids) == 0:
                raise ValueError(
                    f"--follow_body '{args_cli.follow_body}' is not a body of the robot. "
                    f"Available bodies: {follow_robot.body_names}."
                )
            follow_body_id = body_ids[0]
            follow_offset = args_cli.follow_offset
            mode = "heading (body-frame)" if args_cli.follow_yaw else "position-only (world-axis)"
            print(
                f"[INFO] Camera following env {args_cli.follow_env} body "
                f"'{body_names_found[0]}' in {mode} mode, eye offset {tuple(follow_offset)}."
            )

        playback_steps = 0
        try:
            while True:
                if browser_viewer is not None:
                    with torch.inference_mode():
                        changed = browser_viewer.process_inputs()
                    if browser_viewer.reset_observations:
                        obs = env.get_observations()
                        policy.reset(torch.ones(env.num_envs, device=env.unwrapped.device, dtype=torch.bool))
                        browser_viewer.reset_observations = False
                    if changed or browser_viewer.pause.value:
                        browser_viewer.update(obs)
                if browser_viewer is not None and browser_viewer.pause.value:
                    time.sleep(0.02)
                    continue
                start_time = time.time()
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    elif policy_nn is not None:
                        policy_nn.reset(dones)
                playback_steps += 1
                if browser_viewer is not None:
                    browser_viewer.record_step(dones)
                if browser_viewer is not None and playback_steps % 2 == 0:
                    browser_viewer.update(obs)
                if args_cli.follow_cam and not use_physx_viser:
                    # body_pos_w is a Warp array in this IsaacLab build; .torch gives a view.
                    target = follow_robot.data.body_pos_w.torch[args_cli.follow_env, follow_body_id]
                    target = target.detach().cpu().tolist()
                    ox, oy, oz = follow_offset
                    if args_cli.follow_yaw:
                        # Extract yaw from the root quaternion (x, y, z, w in this build).
                        qx, qy, qz, qw = follow_robot.data.root_quat_w.torch[args_cli.follow_env].tolist()
                        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                        # Circular low-pass filter on the heading to damp per-step jitter.
                        alpha = args_cli.follow_smooth
                        if follow_yaw_state is None or alpha <= 0.0:
                            follow_yaw_state = yaw
                        else:
                            # Blend in the shortest-arc direction so wrap-around never spins.
                            d = math.atan2(math.sin(yaw - follow_yaw_state), math.cos(yaw - follow_yaw_state))
                            follow_yaw_state += (1.0 - alpha) * d
                        c, s = math.cos(follow_yaw_state), math.sin(follow_yaw_state)
                        # Rotate the body-frame (x=forward, y=left) offset into world by yaw.
                        wx = c * ox - s * oy
                        wy = s * ox + c * oy
                    else:
                        wx, wy = ox, oy
                    eye = (target[0] + wx, target[1] + wy, target[2] + oz)
                    env.unwrapped.sim.set_camera_view(eye=eye, target=tuple(target))
                if args_cli.video:
                    timestep += 1
                    if timestep == args_cli.video_length:
                        break
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)
                if args_cli.max_steps and playback_steps >= args_cli.max_steps:
                    print(f"[INFO] Playback completed {playback_steps} steps.", flush=True)
                    break
        except KeyboardInterrupt:
            pass
        finally:
            if browser_viewer is not None:
                browser_viewer.close()
            env.close()


if __name__ == "__main__":
    main()
