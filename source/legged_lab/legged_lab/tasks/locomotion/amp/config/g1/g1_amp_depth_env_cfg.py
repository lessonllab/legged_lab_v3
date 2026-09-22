"""G1 visual AMP: head-mounted depth actor, privileged height-scan critic.

The nominal camera pose/FOV follows the G1 head camera documented by InstinctLab:
https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py
The ray-cast prototype sees static terrain, not moving robot-body occlusions.
Calibrate extrinsics and add sensor noise/delay before real-robot deployment.
"""

import copy
import math

import torch

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.sensors import RayCasterCameraCfg, patterns
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from legged_lab.tasks.locomotion.amp.mdp.depth import depth_image

from .g1_amp_rough_env_cfg import G1AmpRoughEnvCfg


@configclass
class DepthObservationsCfg(ObservationGroupCfg):
    # NHW per frame -> NTHW history, interpreted as NCHW by RSL-RL CNNModel.
    image = ObservationTermCfg(
        func=depth_image,
        params={"sensor_name": "depth_camera", "near": 0.1, "far": 2.5},
        history_length=8,
        flatten_history_dim=False,
    )

    def __post_init__(self):
        self.concatenate_terms = True
        self.concatenate_dim = 1
        self.enable_corruption = False
        self.history_length = None


@configclass
class G1AmpDepthEnvCfg(G1AmpRoughEnvCfg):
    """Train a visual actor directly with PPO+AMP on rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        # Avoid mutating the terrain preset used by existing rough/flat configs.
        self.scene.terrain.terrain_generator = copy.deepcopy(self.scene.terrain.terrain_generator)
        self.scene.terrain.max_init_terrain_level = 0

        # Approximate nominal G1 head extrinsics relative to torso_link. Construct
        # the quaternion with the installed math API to respect its component order.
        camera_rot = math_utils.quat_from_euler_xyz(
            torch.tensor(0.0), torch.tensor(math.radians(48.0)), torch.tensor(0.0)
        ).tolist()
        self.scene.depth_camera = RayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            mesh_prim_paths=["/World/ground"],
            update_period=0.04,  # 25 Hz camera; the controller remains 50 Hz.
            offset=RayCasterCameraCfg.OffsetCfg(
                pos=(0.0488, 0.01, 0.4378), rot=tuple(camera_rot), convention="world"
            ),
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=1.0,
                horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),
                vertical_aperture=2 * math.tan(math.radians(58.29) / 2),
                width=64,
                height=36,
            ),
            data_types=["distance_to_image_plane"],
            max_distance=3.0,
            depth_clipping_behavior="none",
            debug_vis=False,
        )
        # Repeated frames are intentionally held between camera updates. Histories
        # contain eight CONTROL ticks (160 ms), not eight independent camera frames.
        self.observations.depth = DepthObservationsCfg()
        self.observations.policy.height_scan = None
        self.scene.height_scanner.debug_vis = False

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.events.push_robot = None


@configclass
class G1AmpDepthEnvCfg_PLAY(G1AmpDepthEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_rows = 4
        self.scene.terrain.terrain_generator.num_cols = 6
