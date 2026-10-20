"""Unitree's G1 velocity baseline adapted to the project's Isaac Sim 6 runtime."""
import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity", entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:RobotEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:OfficialPPORunnerCfg"},
)
