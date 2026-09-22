"""Instinct-aligned AMP balance on the current visual actor and G1 dynamics.

Reference: project-instinct/instinct_rl/algorithms/wasabi.py and InstinctLab
parkour/config/g1/agents/instinct_rl_amp_cfg.py (CC BY-NC 4.0).
"""
from isaaclab.utils.configclass import configclass
from .rsl_rl_depth_target_v5_ppo_cfg import G1AmpDepthTargetV5PPORunnerCfg


@configclass
class G1AmpDepthTargetV6PPORunnerCfg(G1AmpDepthTargetV5PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_amp_depth_target_v6"
        self.run_name = "target_v6_scratch"
        self.resume = False
        self.max_iterations = 30000
        amp = self.algorithm.amp_cfg
        amp.disc_optimizer = "adamw"
        amp.disc_optimizer_weight_decay = .01
        amp.disc_trunk_weight_decay = 0.  # Legacy Adam groups are unused in this mode.
        amp.disc_linear_weight_decay = 0.
        amp.disc_weight_l2_coef = 3e-4
        amp.disc_logit_l2_coef = .04
        amp.grad_penalty_data = "agent_demo"
        amp.grad_penalty_scale = 5.
        amp.disc_learning_rate = 1e-4
        amp.disc_max_grad_norm = None
        amp.disc_replay_rollouts = 1
        amp.disc_replay_current_fraction = 1.
        amp.disc_update_interval = 1
        disc = amp.amp_discriminator
        disc.reward_combination = "additive"
        disc.style_reward_time_scaled = False
        disc.observation_normalization = "none"
        disc.style_reward_scale = .25
        disc.task_style_lerp = 0.
        disc.activation = "relu"
        # Preserve .5 learned exploration and entropy .006 as a baseline while
        # fixing reward units/AMP. Any lower-exploration test must be a named run.
