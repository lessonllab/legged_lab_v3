from isaaclab.utils.configclass import configclass
from .rsl_rl_depth_instinct_ppo_cfg import G1AmpDepthInstinctPPORunnerCfg


@configclass
class G1AmpDepthStylePPORunnerCfg(G1AmpDepthInstinctPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'g1_amp_depth_style_v3'
        # Retain PPO/AMP optimizer and reward scales for a controlled comparison.
        # Reference mirroring is independent of policy-image augmentation.
