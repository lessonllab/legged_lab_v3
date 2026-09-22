from isaaclab.utils.configclass import configclass
from .rsl_rl_depth_target_v7_ppo_cfg import G1AmpDepthTargetV7PPORunnerCfg


@configclass
class G1AmpScratchPPORunnerCfg(G1AmpDepthTargetV7PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'g1_amp_scratch_instinct'
        self.run_name = 'flat_entry_style05'
        self.algorithm.class_name = 'legged_lab.rsl_rl.amp.scratch_ppo:ScratchPPOAMP'
        self.algorithm.amp_cfg.style_reward_gate_group = None
        self.algorithm.learning_rate = 1e-3
        self.algorithm.schedule = 'adaptive'
        # Instinct's actor uses init_noise_std=1.0. Our action scale is .25;
        # preserve a bounded exploration distribution on the existing actor.
        self.actor.distribution_cfg.init_std = 1.0
        self.actor.distribution_cfg.std_range = (.05, 1.5)
