"""Official G1 PPO hyperparameters expressed with the project's RSL 5 API."""
from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class OfficialPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 100
    experiment_name = "g1_unitree_official_sim6"
    resume = False
    clip_actions = None
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO", num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1e-3, schedule="adaptive", desired_kl=.01,
        gamma=.99, lam=.95, entropy_coef=.01, max_grad_norm=1.,
        value_loss_coef=1., use_clipped_value_loss=True, clip_param=.2,
        normalize_advantage_per_mini_batch=False,
    )
