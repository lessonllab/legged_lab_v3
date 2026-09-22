import gymnasium as gym


from . import agents

for _play in (False, True):
    gym.register(
        id=f"LeggedLab-Isaac-AMP-Stairs-Long-G1{'-Play' if _play else ''}-v0",
        entry_point="legged_lab.envs:ManagerBasedAmpEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_amp_stairs_long_env_cfg:G1AmpStairsLongEnvCfg{'_PLAY' if _play else ''}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_stairs_long_ppo_cfg:G1AmpStairsLongPPORunnerCfg",
        },
    )

for _play in (False, True):
    gym.register(
        id=f"LeggedLab-Isaac-AMP-Stairs-G1{'-Play' if _play else ''}-v1",
        entry_point="legged_lab.envs:ManagerBasedAmpEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_amp_stairs_speed_env_cfg:G1AmpStairsSpeedEnvCfg{'_PLAY' if _play else ''}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_stairs_speed_ppo_cfg:G1AmpStairsSpeedPPORunnerCfg",
        },
    )

for _play in (False, True):
    gym.register(
        id=f"LeggedLab-Isaac-AMP-Stairs-G1{'-Play' if _play else ''}-v0",
        entry_point="legged_lab.envs:ManagerBasedAmpEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_amp_stairs_env_cfg:G1AmpStairsEnvCfg{'_PLAY' if _play else ''}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_stairs_ppo_cfg:G1AmpStairsPPORunnerCfg",
        },
    )

for _play in (False, True):
    gym.register(
        id=f"LeggedLab-Isaac-AMP-Scratch-G1{'-Play' if _play else ''}-v1",
        entry_point="legged_lab.envs:ManagerBasedAmpEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_amp_balanced_env_cfg:G1AmpBalancedEnvCfg{'_PLAY' if _play else ''}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_balanced_ppo_cfg:G1AmpBalancedPPORunnerCfg",
        },
    )

for _play in (False, True):
    gym.register(
        id=f"LeggedLab-Isaac-AMP-Scratch-G1{'-Play' if _play else ''}-v0",
        entry_point="legged_lab.envs:ManagerBasedAmpEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_amp_scratch_env_cfg:G1AmpScratchEnvCfg{'_PLAY' if _play else ''}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_scratch_ppo_cfg:G1AmpScratchPPORunnerCfg",
        },
    )

for _play in (False, True):
    gym.register(
        id=f"LeggedLab-Isaac-AMP-Depth-Target-G1{'-Play' if _play else ''}-v3",
        entry_point="legged_lab.envs:ManagerBasedAmpEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_amp_depth_target_v7_env_cfg:G1AmpDepthTargetV7EnvCfg{'_PLAY' if _play else ''}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_target_v7_ppo_cfg:G1AmpDepthTargetV7PPORunnerCfg",
        },
    )

for _play in (False, True):
    gym.register(
        id=f"LeggedLab-Isaac-AMP-Depth-Target-G1{'-Play' if _play else ''}-v2",
        entry_point="legged_lab.envs:ManagerBasedAmpEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_amp_depth_target_v6_env_cfg:G1AmpDepthTargetV6EnvCfg{'_PLAY' if _play else ''}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_target_v6_ppo_cfg:G1AmpDepthTargetV6PPORunnerCfg",
        },
    )

for _play in (False, True):
    gym.register(
        id=f"LeggedLab-Isaac-AMP-Depth-Target-G1{'-Play' if _play else ''}-v1",
        entry_point="legged_lab.envs:ManagerBasedAmpEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_amp_depth_target_v5_env_cfg:G1AmpDepthTargetV5EnvCfg{'_PLAY' if _play else ''}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_target_v5_ppo_cfg:G1AmpDepthTargetV5PPORunnerCfg",
        },
    )

for _play in (False, True):
    gym.register(
        id=f"LeggedLab-Isaac-AMP-Depth-Target-G1{'-Play' if _play else ''}-v0",
        entry_point="legged_lab.envs:ManagerBasedAmpEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_amp_depth_target_env_cfg:G1AmpDepthTargetEnvCfg{'_PLAY' if _play else ''}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_target_ppo_cfg:G1AmpDepthTargetPPORunnerCfg",
        },
    )

for _play in (False, True):
    gym.register(
        id=f"LeggedLab-Isaac-AMP-Depth-Style-G1{'-Play' if _play else ''}-v0",
        entry_point="legged_lab.envs:ManagerBasedAmpEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_amp_depth_style_env_cfg:G1AmpDepthStyleEnvCfg{'_PLAY' if _play else ''}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_style_ppo_cfg:G1AmpDepthStylePPORunnerCfg",
        },
    )

# Separate registrations keep existing visual checkpoints and running jobs valid.
for _suffix, _env_class, _runner_class in (
    ("", "G1AmpDepthInstinctEnvCfg", "G1AmpDepthInstinctPPORunnerCfg"),
    ("-Warmup", "G1AmpDepthInstinctWarmupEnvCfg", "G1AmpDepthInstinctWarmupPPORunnerCfg"),
):
    for _play in (False, True):
        gym.register(
            id=f"LeggedLab-Isaac-AMP-Depth-Instinct{_suffix}-G1{'-Play' if _play else ''}-v0",
            entry_point="legged_lab.envs:ManagerBasedAmpEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.g1_amp_depth_instinct_env_cfg:{_env_class}{'_PLAY' if _play else ''}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_instinct_ppo_cfg:{_runner_class}",
            },
        )

# -- Flat straight-walking warm-up with depth inputs -------------------------
gym.register(
    id="LeggedLab-Isaac-AMP-Depth-Warmup-G1-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_depth_warmup_env_cfg:G1AmpDepthWarmupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_warmup_ppo_cfg:G1AmpDepthWarmupPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-Depth-Warmup-G1-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_depth_warmup_env_cfg:G1AmpDepthWarmupEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_warmup_ppo_cfg:G1AmpDepthWarmupPPORunnerCfg",
    },
)

# -- Depth-vision rough terrain ---------------------------------------------
gym.register(
    id="LeggedLab-Isaac-AMP-Depth-G1-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_depth_env_cfg:G1AmpDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_ppo_cfg:G1AmpDepthPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-Depth-G1-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_depth_env_cfg:G1AmpDepthEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_ppo_cfg:G1AmpDepthPPORunnerCfg",
    },
)

##
# Register Gym environments.
##

# -- Rough terrain (base config) ---------------------------------------------
gym.register(
    id="LeggedLab-Isaac-AMP-Rough-G1-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_rough_env_cfg:G1AmpRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1AmpRoughPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-Rough-G1-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_rough_env_cfg:G1AmpRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1AmpRoughPPORunnerCfg",
    },
)

# -- Flat terrain (derived config) -------------------------------------------
gym.register(
    id="LeggedLab-Isaac-AMP-Flat-G1-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_flat_env_cfg:G1AmpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1AmpFlatPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-Flat-G1-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_flat_env_cfg:G1AmpFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1AmpFlatPPORunnerCfg",
    },
)

# V12 rewards are opt-in; existing play/training configurations remain reproducible.
for suffix, cfg_name in (('', 'G1AmpFootholdEnvCfg'), ('-Play', 'G1AmpFootholdEnvCfg_PLAY')):
    gym.register(
        id=f'LeggedLab-Isaac-AMP-Foothold-G1{suffix}-v0',
        entry_point='legged_lab.envs:ManagerBasedAmpEnv', disable_env_checker=True,
        kwargs={'env_cfg_entry_point':f'{__name__}.g1_amp_foothold_env_cfg:{cfg_name}',
                'rsl_rl_cfg_entry_point':f'{agents.__name__}.rsl_rl_stairs_long_ppo_cfg:G1AmpStairsLongPPORunnerCfg'},
    )
