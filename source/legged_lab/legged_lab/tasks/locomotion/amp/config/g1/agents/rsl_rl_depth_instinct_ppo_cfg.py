import copy

from isaaclab.utils.configclass import configclass

from .rsl_rl_depth_warmup_ppo_cfg import G1AmpDepthWarmupPPORunnerCfg


@configclass
class G1AmpDepthInstinctPPORunnerCfg(G1AmpDepthWarmupPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_amp_depth_instinct_v2"
        # Actor and critic have separate trainable CNNs but read the same delayed
        # images. Critic retains privileged motion data, not ideal terrain heights.
        self.critic = copy.deepcopy(self.actor)
        self.critic.distribution_cfg = None
        self.obs_groups["critic"] = ["critic", "depth"]


@configclass
class G1AmpDepthInstinctWarmupPPORunnerCfg(G1AmpDepthInstinctPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_amp_depth_instinct_warmup_v2"
