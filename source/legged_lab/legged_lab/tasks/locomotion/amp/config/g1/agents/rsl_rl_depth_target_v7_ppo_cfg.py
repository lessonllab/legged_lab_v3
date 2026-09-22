"""Actor warm start with a fresh critic/discriminator and bounded exploration."""
from isaaclab.utils.configclass import configclass
from .rsl_rl_depth_target_v6_ppo_cfg import G1AmpDepthTargetV6PPORunnerCfg


@configclass
class G1AmpDepthTargetV7PPORunnerCfg(G1AmpDepthTargetV6PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'g1_amp_depth_target_v7'
        self.run_name = 'progress_walk_run_style05_floor05'
        self.save_interval = 100
        self.algorithm.amp_cfg.style_reward_gate_group = 'style_gate'
        # V6/Instinct use .25 without this additional progress/tracking gate.
        # Increase the effective style contribution while retaining the gate
        # that blocks commanded standing and discounts overspeed.
        self.algorithm.amp_cfg.amp_discriminator.style_reward_scale = .50
        self.actor.distribution_cfg.init_std = .25
        self.actor.distribution_cfg.std_range = (.05, .5)
        self.algorithm.learning_rate = 1e-4
        self.algorithm.schedule = 'fixed'
