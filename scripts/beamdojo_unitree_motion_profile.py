"""Provenance for complete Unitree motion + BeamDojo perception/sparse critic."""
PROFILE = 'beamdojo_unitree_motion_g1_fullbody'
PPO = {
    'num_steps_per_env': 24, 'actor_hidden_dims': [512, 256, 128],
    'critic_hidden_dims': [512, 256, 128], 'initial_std': 1.,
    'num_epochs': 5, 'num_mini_batches': 4, 'initial_learning_rate': .001,
    'learning_rate_schedule': 'adaptive', 'gamma': .99, 'lam': .95,
    'entropy_coef': .01, 'desired_kl': .01, 'clip_param': .2,
    'value_loss_coef': 1., 'max_grad_norm': 1., 'use_clipped_value_loss': True,
    'advantage_weights': [1., .25], 'observation_normalization': False,
    'clip_actions': None, 'activation': 'elu',
}
