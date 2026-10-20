"""Pinned humanoid-gym training constants; no Isaac imports, testable on CPU.

Source: ezoicoder/humanoid-gym @ 9a47b7d, XBotLCfgPPO and
XBotLStoneStage1PlaneCfgPPO/XBotLStoneCfgPPO. Double critic is the repository's
README_STONES training option, with its configured 1.0/0.25 advantage weights.
G1 mechanics and upper-body posture terms come separately from Unitree.
"""
PROFILE = "beamdojo_upstream_unitree_g1_fullbody"
PPO = {
    "num_steps_per_env": 100,
    "actor_hidden_dims": [768, 512, 256],
    "critic_hidden_dims": [1024, 512, 256],
    "initial_std": 1.0,
    "num_epochs": 2,
    "num_mini_batches": 4,
    "initial_learning_rate": 1e-5,
    "learning_rate_schedule": "adaptive",
    "gamma": .994,
    "lam": .9,
    "entropy_coef": .001,
    "desired_kl": .01,
    "clip_param": .2,
    "value_loss_coef": 1.,
    "max_grad_norm": 1.,
    "use_clipped_value_loss": True,
    "advantage_weights": [1., .25],
    "observation_normalization": False,
    "clip_actions": 18.,
    "activation": "elu",
}


def configure_agent(agent):
    agent.num_steps_per_env = PPO['num_steps_per_env']
    agent.save_interval = 100
    agent.clip_actions = PPO['clip_actions']
    agent.actor.hidden_dims = list(PPO['actor_hidden_dims'])
    agent.critic.hidden_dims = list(PPO['critic_hidden_dims'])
    agent.actor.activation = agent.critic.activation = PPO['activation']
    agent.actor.obs_normalization = agent.critic.obs_normalization = False
    agent.actor.distribution_cfg.init_std = PPO['initial_std']
    agent.actor.distribution_cfg.std_type = 'scalar'
    names = {'num_learning_epochs':'num_epochs', 'learning_rate':'initial_learning_rate',
             'schedule':'learning_rate_schedule'}
    for name in ('num_learning_epochs', 'num_mini_batches', 'learning_rate', 'schedule',
                 'gamma', 'lam', 'entropy_coef', 'desired_kl', 'clip_param',
                 'value_loss_coef', 'max_grad_norm', 'use_clipped_value_loss'):
        setattr(agent.algorithm, name, PPO[names.get(name, name)])
    agent.algorithm.normalize_advantage_per_mini_batch = False
    return agent
