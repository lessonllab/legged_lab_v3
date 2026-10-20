"""G1 BeamDojo adaptation: full-body actions and ideal local terrain maps."""
import gymnasium as gym

gym.register(
    id='G1-BeamDojo-29dof-v0',
    entry_point=f'{__name__}.env:BeamDojoEnv',
    disable_env_checker=True,
    kwargs={'env_cfg_entry_point': f'{__name__}.env_cfg:BeamDojoEnvCfg',
            'rsl_rl_cfg_entry_point': 'legged_lab.tasks.locomotion.unitree_official.agents:OfficialPPORunnerCfg'},
)
