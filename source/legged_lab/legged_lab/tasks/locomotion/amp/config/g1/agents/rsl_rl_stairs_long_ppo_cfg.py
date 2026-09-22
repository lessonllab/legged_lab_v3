from isaaclab.utils.configclass import configclass
from .rsl_rl_stairs_speed_ppo_cfg import G1AmpStairsSpeedPPORunnerCfg


@configclass
class G1AmpStairsLongPPORunnerCfg(G1AmpStairsSpeedPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'g1_amp_stairs_long'
        self.run_name = 'long24_flat3_stairs1p5'
        self.algorithm.amp_cfg.amp_discriminator.style_reward_scale = .25
