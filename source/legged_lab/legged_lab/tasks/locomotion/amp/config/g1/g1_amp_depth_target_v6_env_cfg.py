"""Visual target locomotion with Instinct-style gait terms and AMP balance."""
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from legged_lab.tasks.locomotion.amp.mdp.rewards import (
    instinct_feet_air_time, stand_still, body_orientation_l2,
)
from legged_lab.tasks.locomotion.amp.mdp.terminations import body_tilt_exceeds
from .g1_amp_depth_target_v5_env_cfg import G1AmpDepthTargetV5EnvCfg, G1AmpDepthTargetV5EnvCfg_PLAY


def configure_target_v6(cfg):
    cfg.rewards.feet_air_time = RewardTermCfg(
        func=instinct_feet_air_time, weight=.5,
        params={"command_name": "base_velocity", "vel_threshold": .15,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    # Use one whole-body standing term, not three overlapping partial terms.
    for name in ("joint_deviation_hip", "joint_deviation_arms", "joint_deviation_waist"):
        setattr(cfg.rewards, name, None)
    cfg.rewards.stand_still = RewardTermCfg(
        func=stand_still, weight=-.3,
        params={"command_name": "base_velocity", "threshold": .15, "offset": 4.},
    )
    bodies = SceneEntityCfg("robot", body_names=["pelvis", "torso_link"], preserve_order=True)
    cfg.rewards.flat_orientation_l2 = None
    cfg.rewards.upright_body = RewardTermCfg(
        func=body_orientation_l2, weight=-3., params={"asset_cfg": bodies},
    )
    cfg.terminations.excessive_body_tilt = TerminationTermCfg(
        func=body_tilt_exceeds, params={"asset_cfg": bodies, "limit_angle": 1.},
    )
    # This model's root is pelvis, whereas Instinct's root is torso. Calibrated
    # against the local 30-clip dataset (minimum pelvis Z .603 m), with >.25 m
    # clearance margin for terrain adaptation; this is not a fixed height target.
    cfg.terminations.base_height.params["minimum_height"] = .35


@configclass
class G1AmpDepthTargetV6EnvCfg(G1AmpDepthTargetV5EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_target_v6(self)


@configclass
class G1AmpDepthTargetV6EnvCfg_PLAY(G1AmpDepthTargetV5EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        configure_target_v6(self)
