from dataclasses import MISSING
from typing import Literal

from isaaclab.utils.configclass import configclass


@configclass
class RslRlAmpCfg:
    """Configuration class for the AMP (Adversarial Motion Priors) in the training"""

    disc_obs_buffer_size: int = 1000
    """Legacy field, unused by PPOAMP. Use disc_replay_rollouts for capacity."""

    disc_replay_rollouts: int = 1
    """Number of rollout lengths retained. Default preserves existing task behaviour."""

    disc_replay_current_fraction: float = 0.5
    """Fraction sampled from the latest rollout when older samples are available."""

    grad_penalty_scale: float = 10.0
    """Scale for the gradient penalty in AMP training"""

    grad_penalty_data: Literal["demo", "agent_demo"] = "demo"
    """Input-gradient penalty population. agent_demo averages over both classes equally."""

    disc_optimizer: Literal["adam", "adamw"] = "adam"
    """Adam retains legacy trunk/logit parameter-group decay; AdamW uses the global decay below."""

    disc_optimizer_weight_decay: float = 0.0
    """Decoupled AdamW weight decay; ignored by the legacy Adam optimizer."""

    disc_weight_l2_coef: float = 0.0
    """Explicit loss coefficient on the sum of squared discriminator parameters, including biases."""

    disc_logit_l2_coef: float = 0.0
    """Additional explicit loss coefficient on the squared final-layer weights (excludes bias)."""

    disc_trunk_weight_decay: float = 1.0e-4
    """Weight decay for the discriminator trunk network"""

    disc_linear_weight_decay: float = 1.0e-2
    """Weight decay for the discriminator linear network"""

    disc_learning_rate: float = 1.0e-5
    """Learning rate for the discriminator networks"""

    disc_max_grad_norm: float | None = 1.0
    """Maximum gradient norm for the discriminator networks; None disables clipping."""

    disc_update_interval: int = 1
    """Discriminator update interval: perform a discriminator gradient step only once every this
    many PPO mini-batch steps. Values > 1 slow down the discriminator relative to the policy,
    helping prevent early saturation. Default 1 = update every mini-batch (original behaviour)."""

    @configclass
    class AMPDiscriminatorCfg:
        """Configuration for the AMP discriminator network."""

        hidden_dims: list[int] = MISSING
        """The hidden dimensions of the AMP discriminator network."""

        activation: str = "elu"
        """The activation function for the AMP discriminator network."""

        style_reward_scale: float = 1.0
        """Scale for the style reward in the training"""

        task_style_lerp: float = 0.0
        """Linear interpolation factor for the task style reward in the AMP training."""

        reward_combination: Literal["lerp", "additive"] = "lerp"
        """Legacy interpolation or task + style. task_style_lerp is ignored in additive mode."""

        style_reward_time_scaled: bool = True
        """Multiply the scaled style reward by step_dt. False uses a per-step style coefficient."""

        observation_normalization: Literal["empirical", "none"] = "empirical"
        """Running normalization across frames, or directly use the physical style features."""

    amp_discriminator: AMPDiscriminatorCfg = AMPDiscriminatorCfg()
    """Configuration for the AMP discriminator network."""

    loss_type: Literal["GAN", "LSGAN", "WGAN"] = "LSGAN"
    """Type of loss function used for the AMP discriminator (e.g., 'GAN', 'LSGAN', 'WGAN')"""
    style_reward_gate_group: str | None = None
    """Optional scalar progress group; None preserves earlier AMP behavior."""
