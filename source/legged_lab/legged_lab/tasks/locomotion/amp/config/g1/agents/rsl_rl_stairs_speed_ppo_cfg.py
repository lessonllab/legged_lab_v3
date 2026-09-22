from isaaclab.utils.configclass import configclass
from .rsl_rl_stairs_ppo_cfg import G1AmpStairsPPORunnerCfg


@configclass
class G1AmpStairsSpeedPPORunnerCfg(G1AmpStairsPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'g1_amp_stairs_speed'
        self.run_name = 'from26500_flat3_stairs1p5'
