"""Target v5: consistent turn rewards, verified progress and standing resets.

Adapted from InstinctLab parkour (CC BY-NC 4.0):
https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py
The progress curriculum is a local correction, not Instinct's published gate.
"""
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from isaaclab.utils.configclass import configclass

from legged_lab.tasks.locomotion.amp.mdp.commands.target_velocity import target_progress_curriculum
from legged_lab.tasks.locomotion.amp.mdp.rewards import target_heading_error
from .g1_amp_depth_target_env_cfg import (
    G1AmpDepthTargetEnvCfg, G1AmpDepthTargetEnvCfg_PLAY, TargetVelocityCommandCfg,
)


@configclass
class TargetV5VelocityCommandCfg(TargetVelocityCommandCfg):
    target_initial_distance_margin: float = .2
    target_min_arrival_progress: float = .2


def configure_target_v5(cfg, play=False):
    cfg.commands.base_velocity = TargetV5VelocityCommandCfg()
    cfg.rewards.track_lin_vel_xy_exp.weight = 2.0
    cfg.rewards.track_ang_vel_z_exp.weight = 2.0
    cfg.rewards.heading_error = RewardTermCfg(
        func=target_heading_error, weight=-1.0, params={"command_name": "base_velocity"},
    )
    for name in ("joint_deviation_hip", "joint_deviation_arms", "joint_deviation_waist"):
        getattr(cfg.rewards, name).params.update(command_threshold=.15, angular_threshold=.15)

    # Robot spawn and AMP reference sampling are independent. Retain the full
    # mirrored motion dataset and 10-frame discriminator observations.
    cfg.events.reset_from_ref = None
    cfg.events.reset_base = EventTermCfg(
        func=base_mdp.reset_root_state_uniform, mode="reset", params={
            "pose_range": {"x": (-.1, .1), "y": (-.1, .1), "yaw": (-.1, .1)},
            "velocity_range": {axis: (-.2, .2) for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )
    cfg.events.reset_robot_joints = EventTermCfg(
        func=base_mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (0., 0.) if play else (-.15, .15), "velocity_range": (0., 0.)},
    )
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = None if play else CurriculumTermCfg(
        func=target_progress_curriculum, params={
            "xy_threshold": .6, "yaw_threshold": .5,
            "demote_xy_threshold": .3, "demote_yaw_threshold": .3,
            "min_opportunity_s": 2., "min_progress_m": 1.,
        },
    )


@configclass
class G1AmpDepthTargetV5EnvCfg(G1AmpDepthTargetEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_target_v5(self)


@configclass
class G1AmpDepthTargetV5EnvCfg_PLAY(G1AmpDepthTargetEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        configure_target_v5(self, play=True)
