from isaaclab.utils.configclass import configclass
from .rsl_rl_scratch_ppo_cfg import G1AmpScratchPPORunnerCfg


@configclass
class G1AmpBalancedPPORunnerCfg(G1AmpScratchPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'g1_amp_scratch_balanced'
        self.run_name = 'instinct_terrain_flat_speed'
        self.algorithm.class_name = 'legged_lab.rsl_rl.amp.balanced_ppo:BalancedPPOAMP'
        # Continuation profile: the inherited scratch exploration (.7 observed)
        # degraded tracking versus the same deterministic policy in the audit.
        self.actor.distribution_cfg.init_std = .5
        self.actor.distribution_cfg.std_range = (.05, .5)
        self.algorithm.entropy_coef = .003
