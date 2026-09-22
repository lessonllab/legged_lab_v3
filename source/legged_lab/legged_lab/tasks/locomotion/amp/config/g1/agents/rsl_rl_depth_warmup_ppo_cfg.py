from isaaclab_rl.rsl_rl import RslRlCNNModelCfg
from isaaclab.utils.configclass import configclass

from .rsl_rl_depth_ppo_cfg import G1AmpDepthPPORunnerCfg


@configclass
class FixedGaussianDistributionCfg(RslRlCNNModelCfg.GaussianDistributionCfg):
    # Supported by RSL-RL 5.4.1; not exposed by the installed Isaac Lab base cfg.
    learn_std: bool = False


@configclass
class G1AmpDepthWarmupPPORunnerCfg(G1AmpDepthPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_amp_depth_warmup"
        self.max_iterations = 10000
        self.save_interval = 500
        self.actor.distribution_cfg = FixedGaussianDistributionCfg(init_std=0.5)
        self.algorithm.entropy_coef = 0.0
        self.algorithm.amp_cfg.disc_replay_rollouts = 10
        self.algorithm.amp_cfg.disc_replay_current_fraction = 0.5
