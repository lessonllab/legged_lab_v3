"""Reward-preserving stair specialization of the balanced visual policy."""
from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils.configclass import configclass
from .g1_amp_balanced_env_cfg import G1AmpBalancedEnvCfg, BalancedCommandCfg
from .g1_amp_scratch_env_cfg import entry_terrain
from legged_lab.tasks.locomotion.amp.mdp.stair_course import StairTargetCommand, stair_curriculum, STAIR_HEIGHTS


def specialist_stairs(difficulty, cfg):
    row = min(9, int(difficulty * 10))
    if row == 0:
        return entry_terrain(difficulty, cfg)
    original = cfg.original.copy()
    original.size = cfg.size
    original.step_height_range = (STAIR_HEIGHTS[row], STAIR_HEIGHTS[row])
    return original.function(difficulty, original)


@configclass
class StairCommandCfg(BalancedCommandCfg):
    class_type = StairTargetCommand


@configclass
class G1AmpStairsEnvCfg(G1AmpBalancedEnvCfg):
    # Artificially shortened first episodes must not enter traversal windows.
    randomize_initial_episode_length: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = StairCommandCfg(speed_ranges=dict(self.commands.base_velocity.speed_ranges))
        self.curriculum.terrain_levels = CurriculumTermCfg(func=stair_curriculum)
        for name in ('pyramid_stairs', 'pyramid_stairs_inv'):
            self.scene.terrain.terrain_generator.sub_terrains[name].function = specialist_stairs


@configclass
class G1AmpStairsEnvCfg_PLAY(G1AmpStairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.reset_robot_joints.params['position_range'] = (0., 0.)
