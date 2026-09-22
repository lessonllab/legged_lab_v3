"""Target v5 runner, isolated from the stopped v4 experiment."""
from isaaclab_rl.rsl_rl import RslRlCNNModelCfg
from isaaclab.utils.configclass import configclass
from .rsl_rl_depth_target_ppo_cfg import G1AmpDepthTargetPPORunnerCfg


@configclass
class LearnedTargetGaussianCfg(RslRlCNNModelCfg.GaussianDistributionCfg):
    # RSL-RL 5.4.1 supports these even where Isaac Lab's config omits them.
    learn_std: bool = True
    std_type: str = "log"
    std_range: tuple[float, float] = (.05, 1.)


@configclass
class G1AmpDepthTargetV5PPORunnerCfg(G1AmpDepthTargetPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_amp_depth_target_v5"
        self.run_name = "target_v5_scratch"
        self.resume = False
        self.max_iterations = 30000
        self.save_interval = 500
        # Retain the local initial scale .5; learn it instead of fixing it.
        # Log-space and explicit bounds are local RSL-RL adaptations.
        self.actor.distribution_cfg = LearnedTargetGaussianCfg(init_std=.5)
        self.algorithm.entropy_coef = .006  # Instinct parkour exploration weight.
