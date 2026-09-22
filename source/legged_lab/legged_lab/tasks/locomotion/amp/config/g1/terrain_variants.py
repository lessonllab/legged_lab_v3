"""Opt-in terrain additions for the G1 visual target task."""
from isaaclab.managers import TerminationTermCfg
from isaaclab.terrains import FlatPatchSamplingCfg
from isaaclab.utils.configclass import configclass
from isaaclab.terrains.height_field import HfTerrainBaseCfg

from legged_lab.tasks.locomotion.amp.mdp.style_state import tensor


def stepping_stones_mesh(difficulty, cfg):
    """Run AME's unchanged generator at its original height-field resolution."""
    from .ame_stakes import alternate_column_stakes_terrain
    cfg = cfg.copy()
    cfg.horizontal_scale = .05
    cfg.vertical_scale = .005
    cfg.slope_threshold = .75
    return alternate_column_stakes_terrain(difficulty, cfg)


@configclass
class SteppingStonesIslandCfg(HfTerrainBaseCfg):
    function = stepping_stones_mesh
    platform_width: float = 2.
    border_width: float = .25
    stake_side_range: tuple[float, float] = (.2, .2)
    stake_gap_range: tuple[float, float] = (.3, .3)
    column_gap_range: tuple[float, float] = (.3, .3)
    column_jitter: float = 0.
    stake_height_max: float = 0.
    holes_depth: float = -2.


def stepping_stones_fall(env, minimum_height=.25):
    """Catch falling between stones relative to the platform, not the deep pit floor."""
    command = env.command_manager.get_term('base_velocity')
    columns = tensor(env.scene.terrain.terrain_types).long()
    import torch
    stone_columns = torch.tensor([name == 'hf_steppingstones' for name in command.column_names],
                                 device=env.device, dtype=torch.bool)
    height = tensor(env.scene['robot'].data.root_pos_w)[:, 2] - env.scene.env_origins[:, 2]
    return stone_columns[columns] & (height < minimum_height)


def add_stepping_stones(cfg):
    """Use AME's alternate-column layout with larger, closer stones requested by the user.

    Source: velocity_env_cfg_29dof.py PLAY sub_terrains['stakes'].
    Task goals and the platform-relative fall termination are local additions.
    """
    generator = cfg.scene.terrain.terrain_generator
    command = cfg.commands.base_velocity
    if generator is None or not hasattr(command, 'speed_ranges'):
        raise ValueError('--stepping_stones requires a generated G1 target-command terrain task')
    if 'hf_steppingstones' in generator.sub_terrains:
        return
    generator.sub_terrains['hf_steppingstones'] = SteppingStonesIslandCfg(
        proportion=.2,
        stake_side_range=(.4, .4),
        stake_gap_range=(.1, .1),
        # A destination beyond the +X bridge induces traversal, like stair goals.
        # Arbitrary diagonal goals would send the heading controller into a pit.
        flat_patch_sampling={'target': FlatPatchSamplingCfg(
            num_patches=50, patch_radius=.08, max_height_diff=.005,
            x_range=(3.85, 3.85), y_range=(0., 0.), z_range=(-.01, .01))},
    )
    generator.num_cols += 4
    command.speed_ranges['hf_steppingstones'] = (.25, .55)
    cfg.terminations.stepping_stones_fall = TerminationTermCfg(func=stepping_stones_fall)
