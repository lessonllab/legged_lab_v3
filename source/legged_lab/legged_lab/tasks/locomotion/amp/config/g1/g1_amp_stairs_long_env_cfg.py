"""17-step continuation: 14 m tiles, crossing without a stop requirement."""
import numpy as np
import trimesh
from isaaclab.managers import RewardTermCfg, CurriculumTermCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass
from .g1_amp_stairs_speed_env_cfg import G1AmpStairsSpeedEnvCfg, StairSpeedCommandCfg
from .g1_amp_scratch_env_cfg import entry_terrain
from legged_lab.tasks.locomotion.amp.mdp.stair_speed_course import descent_overspeed_penalty
from legged_lab.tasks.locomotion.amp.mdp.stair_virtual_safety import virtual_stair_reward
from legged_lab.tasks.locomotion.amp.mdp.agility_course import AgilityCommandMixin
from legged_lab.tasks.locomotion.amp.mdp.adaptive_stair_course import AdaptiveStairTargetCommand, adaptive_stair_curriculum, stair_crossing_reached


def padded_entry_terrain(difficulty, cfg):
    """Retain the original 8 m non-stair task in the center of a larger tile.

    Padding avoids increasing height-field resolution/memory ninefold merely
    to lengthen stairs. All original target patches stay in the central tile.
    """
    small = cfg.copy()
    small.size = (8., 8.)
    meshes, origin = entry_terrain(difficulty, small)
    shift = np.array([(cfg.size[0] - 8.) / 2, (cfg.size[1] - 8.) / 2, 0.])
    for mesh in meshes:
        mesh.apply_translation(shift)
    x, y = cfg.size
    px, py = shift[:2]
    for dims, pos in (
        ((px, y, .1), (px / 2, y / 2, -.05)),
        ((px, y, .1), (x - px / 2, y / 2, -.05)),
        ((8., py, .1), (x / 2, py / 2, -.05)),
        ((8., py, .1), (x / 2, y - py / 2, -.05)),
    ):
        mesh = trimesh.creation.box(extents=dims)
        mesh.apply_translation(pos)
        meshes.append(mesh)
    return meshes, origin + shift


class LongStairTargetCommand(AgilityCommandMixin, AdaptiveStairTargetCommand):
    promotion_version = 11
    terrain_profile = 'stairs14_v2'
    terrain_tile_size = (14., 14.)
    expected_stair_steps = 17

@configclass
class LongStairCommandCfg(StairSpeedCommandCfg):
    class_type = LongStairTargetCommand
    heading_slowdown: bool = True


@configclass
class G1AmpStairsLongEnvCfg(G1AmpStairsSpeedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = LongStairCommandCfg(
            speed_ranges=dict(self.commands.base_velocity.speed_ranges))
        self.curriculum.terrain_levels = CurriculumTermCfg(func=adaptive_stair_curriculum)
        self.terminations.stair_crossing = TerminationTermCfg(func=stair_crossing_reached, time_out=True)
        self.episode_length_s = 60.
        # Instinct-style virtual cylinders replace ray-based edge/support costs.
        self.rewards.feet_support = None
        self.rewards.feet_edge = RewardTermCfg(func=virtual_stair_reward, weight=-4., params={'kind':'edge'})
        self.rewards.feet_toe_riser = RewardTermCfg(func=virtual_stair_reward, weight=-2., params={'kind':'toe'})
        self.rewards.feet_loaded_overhang = RewardTermCfg(func=virtual_stair_reward, weight=-.8, params={'kind':'support'})
        self.rewards.feet_swing_clearance = RewardTermCfg(func=virtual_stair_reward, weight=-1., params={'kind':'clearance'})
        self.rewards.feet_slide.weight = -.2
        self.rewards.dont_wait.params['use_yaw_frame'] = True
        self.rewards.descent_overspeed = RewardTermCfg(func=descent_overspeed_penalty, weight=-.75)
        generator = self.scene.terrain.terrain_generator
        generator.size = LongStairTargetCommand.terrain_tile_size
        for name, terrain in generator.sub_terrains.items():
            if name in ('pyramid_stairs', 'pyramid_stairs_inv'):
                terrain.flat_patch_sampling['target'].x_range = (6.7, 6.7)
            else:
                terrain.function = padded_entry_terrain


@configclass
class G1AmpStairsLongEnvCfg_PLAY(G1AmpStairsLongEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.reset_robot_joints.params['position_range'] = (0., 0.)
