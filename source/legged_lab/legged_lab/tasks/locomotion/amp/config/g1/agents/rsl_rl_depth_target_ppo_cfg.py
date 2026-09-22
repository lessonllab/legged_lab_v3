from isaaclab.utils.configclass import configclass
from .rsl_rl_depth_style_ppo_cfg import G1AmpDepthStylePPORunnerCfg


@configclass
class G1AmpDepthTargetPPORunnerCfg(G1AmpDepthStylePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'g1_amp_depth_target_v4'
