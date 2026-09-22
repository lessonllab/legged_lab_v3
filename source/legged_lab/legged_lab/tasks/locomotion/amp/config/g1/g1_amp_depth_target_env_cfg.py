"""Instinct target-command adaptation on the existing visual rough G1 terrains."""
from isaaclab.managers import CommandTermCfg, CurriculumTermCfg
from isaaclab.terrains import FlatPatchSamplingCfg
from isaaclab.utils.configclass import configclass
from legged_lab.tasks.locomotion.amp.mdp.commands.target_velocity import target_tracking_curriculum
from .g1_amp_depth_style_env_cfg import G1AmpDepthStyleEnvCfg, G1AmpDepthStyleEnvCfg_PLAY


@configclass
class TargetVelocityCommandCfg(CommandTermCfg):
    class_type = 'legged_lab.tasks.locomotion.amp.mdp.commands.target_velocity:TargetVelocityCommand'
    asset_name: str = 'robot'
    resampling_time_range: tuple[float, float] = (8., 12.)
    debug_vis: bool = False  # Render goal and arrows through the PhysX Viser adapter.
    rel_standing_envs: float = .05
    velocity_control_stiffness: float = 2.
    heading_control_stiffness: float = 2.
    target_dis_threshold: float = .4
    yaw_limit: float = 1.
    speed_ranges: dict = {
        'pyramid_stairs': (.45, .8), 'pyramid_stairs_inv': (.45, .8),
        'boxes': (.45, .8), 'random_rough': (.45, 1.),
        'hf_pyramid_slope': (.45, .8), 'hf_pyramid_slope_inv': (.45, .8),
    }


def configure_target(cfg, play=False):
    cfg.commands.base_velocity = TargetVelocityCommandCfg()
    generator = cfg.scene.terrain.terrain_generator
    # Ordered columns are necessary to match terrain names to per-terrain caps.
    # PLAY keeps this layout but disables episode-level curriculum updates.
    generator.curriculum = True
    for name, terrain in generator.sub_terrains.items():
        patch = FlatPatchSamplingCfg(num_patches=50, patch_radius=[.05, .10, .15, .20],
                                     max_height_diff=.05,
                                     x_range=(-3.7, 3.7), y_range=(-3.7, 3.7))
        if name in ('pyramid_stairs', 'pyramid_stairs_inv'):
            # As in Instinct: a goal on the +X end, not a random diagonal stair path.
            patch.x_range, patch.y_range = (3.7, 3.7), (0., 0.)
        terrain.flat_patch_sampling = {'target': patch}
    cfg.curriculum.terrain_levels = None if play else CurriculumTermCfg(func=target_tracking_curriculum)


@configclass
class G1AmpDepthTargetEnvCfg(G1AmpDepthStyleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_target(self)


@configclass
class G1AmpDepthTargetEnvCfg_PLAY(G1AmpDepthStyleEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        configure_target(self, play=True)
