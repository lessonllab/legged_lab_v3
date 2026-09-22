from isaaclab.utils.configclass import configclass
from .rsl_rl_balanced_ppo_cfg import G1AmpBalancedPPORunnerCfg


@configclass
class G1AmpStairsPPORunnerCfg(G1AmpBalancedPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'g1_amp_stairs_specialist'
        self.run_name = 'stairs_window_style05'
        self.algorithm.class_name = 'legged_lab.rsl_rl.amp.stairs_ppo:StairsPPOAMP'
