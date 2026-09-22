"""Progress-gated visual locomotion, retaining the complete walk/run reference set."""
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass
from .g1_amp_depth_target_v6_env_cfg import G1AmpDepthTargetV6EnvCfg, G1AmpDepthTargetV6EnvCfg_PLAY
from legged_lab.tasks.locomotion.amp.mdp.locomotion_progress import (
    style_progress_gate, track_linear_progress, track_angular_progress, low_pelvis_height,
)


@configclass
class ProgressGateObs(ObservationGroupCfg):
    progress = ObservationTermCfg(func=style_progress_gate)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = None


def configure_v7(cfg):
    cfg.observations.style_gate = ProgressGateObs()
    cfg.terminal_obs_groups = ('disc', 'style_gate')
    cfg.rewards.track_lin_vel_xy_exp.func = track_linear_progress
    cfg.rewards.track_ang_vel_z_exp.func = track_angular_progress
    # The source actor walks in a deep squat. Preserve v5's actual fall floor
    # while providing a continuous gradient out of crouching; v6's .35 m hard
    # cut-off terminated >90% of early transfer episodes before adaptation.
    cfg.terminations.base_height.params['minimum_height'] = .20
    cfg.rewards.low_pelvis_height = RewardTermCfg(
        func=low_pelvis_height, weight=-6., params={'target_height': .55,
            'asset_cfg': SceneEntityCfg('robot', body_names=['left_ankle_roll_link', 'right_ankle_roll_link'])})
    # Keep all 30 reference clips and their weights. Faster locomotion belongs
    # on gentler terrain; stairs retain the existing conservative commands.
    cfg.commands.base_velocity.speed_ranges = dict(cfg.commands.base_velocity.speed_ranges)
    cfg.commands.base_velocity.speed_ranges['random_rough'] = (.45, 2.0)
    cfg.commands.base_velocity.speed_ranges['hf_pyramid_slope'] = (.45, 1.2)
    cfg.commands.base_velocity.speed_ranges['hf_pyramid_slope_inv'] = (.45, 1.2)


@configclass
class G1AmpDepthTargetV7EnvCfg(G1AmpDepthTargetV6EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_v7(self)


@configclass
class G1AmpDepthTargetV7EnvCfg_PLAY(G1AmpDepthTargetV6EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        configure_v7(self)
