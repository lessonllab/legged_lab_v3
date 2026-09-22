"""Independent scratch entry, borrowing Instinct's dense task and AMP setup."""
from dataclasses import MISSING
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import CurriculumTermCfg, RewardTermCfg
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.trimesh.mesh_terrains import flat_terrain
from isaaclab.utils.configclass import configclass
from .g1_amp_depth_target_v7_env_cfg import G1AmpDepthTargetV7EnvCfg
from .g1_amp_depth_target_v5_env_cfg import TargetV5VelocityCommandCfg
from legged_lab.tasks.locomotion.amp.mdp.scratch_curriculum import ScratchTargetCommand, scratch_tracking_curriculum
from legged_lab.tasks.locomotion.amp.mdp.locomotion_progress import motion_state
import torch


def entry_terrain(difficulty, cfg):
    if difficulty < cfg.flat_fraction:
        return flat_terrain(difficulty, cfg)
    original = cfg.original.copy()
    original.size = cfg.size
    return original.function((difficulty - cfg.flat_fraction) / (1. - cfg.flat_fraction), original)


@configclass
class EntryTerrainCfg(SubTerrainBaseCfg):
    function = entry_terrain
    original: SubTerrainBaseCfg = MISSING
    flat_fraction: float = .1


def dense_linear_tracking(env, command_name='base_velocity', std=.5):
    command, velocity, _ = motion_state(env, command_name)
    return torch.exp(-(command[:, :2] - velocity).square().sum(-1) / std**2)


def dense_angular_tracking(env, command_name='base_velocity', std=.5):
    command, _, yaw = motion_state(env, command_name)
    return torch.exp(-(command[:, 2] - yaw).square() / std**2)


@configclass
class ScratchCommandCfg(TargetV5VelocityCommandCfg):
    class_type = ScratchTargetCommand


@configclass
class G1AmpScratchEnvCfg(G1AmpDepthTargetV7EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        ranges = dict(self.commands.base_velocity.speed_ranges)
        self.commands.base_velocity = ScratchCommandCfg(speed_ranges=ranges)
        self.observations.style_gate = None
        self.terminal_obs_groups = ('disc',)
        self.rewards.track_lin_vel_xy_exp.func = dense_linear_tracking
        self.rewards.track_ang_vel_z_exp.func = dense_angular_tracking
        self.rewards.is_alive = RewardTermCfg(func=base_mdp.is_alive, weight=3.)
        self.curriculum.terrain_levels = CurriculumTermCfg(func=scratch_tracking_curriculum)
        gen = self.scene.terrain.terrain_generator
        # Ten columns retain all six terrains with the .2/.2/.2/.2/.1/.1 mix.
        gen.num_rows, gen.num_cols = 10, 10
        gen.curriculum = True
        gen.difficulty_range = (0., 1.)
        self.scene.terrain.max_init_terrain_level = 0
        for name, original in list(gen.sub_terrains.items()):
            original = original.copy()
            if isinstance(original, HfTerrainBaseCfg):
                original.horizontal_scale = gen.horizontal_scale
                original.vertical_scale = gen.vertical_scale
                original.slope_threshold = gen.slope_threshold
            gen.sub_terrains[name] = EntryTerrainCfg(
                original=original, flat_fraction=1. / gen.num_rows,
                proportion=original.proportion, flat_patch_sampling=original.flat_patch_sampling)


@configclass
class G1AmpScratchEnvCfg_PLAY(G1AmpScratchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 6
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.reset_robot_joints.params['position_range'] = (0., 0.)
