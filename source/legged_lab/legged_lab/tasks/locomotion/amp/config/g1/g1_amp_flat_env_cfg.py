from isaaclab.utils.configclass import configclass

from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_rough_env_cfg import (
    G1AmpRoughEnvCfg,
    G1AmpRoughEnvCfg_PLAY,
)

# The order must align with the retarget config file scripts/tools/retarget/config/g1_29dof.yaml
ANIMATION_TERM_NAME = "animation"


def _configure_flat_commands_and_events(cfg):
    """Use the same direct velocity commands for flat training and evaluation."""
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 3.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
    # Isolate command responses from periodic external velocity perturbations.
    cfg.events.push_robot = None


@configclass
class G1AmpFlatEnvCfg(G1AmpRoughEnvCfg):
    """Configuration for the G1 AMP environment on flat terrain.

    Inherits the full rough config and strips the rough-only pieces (mirrors the official
    IsaacLab velocity ``flat_env_cfg`` deriving from ``rough_env_cfg``): reverts the terrain
    back to an infinite plane, removes the height scanner + height_scan observation, and
    disables the terrain curriculum. Direct velocity commands and no periodic pushes
    make this task a baseline for studying command tracking versus motion style.
    """

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------
        # Terrain (flat) — revert the generator terrain from the rough base
        # ------------------------------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # On plane terrain env_origins.z == 0 and the reference motion's absolute xy is
        # valid ground, so restore the original DeepMimic-style reset (keep reference xy).
        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            "height_offset": 0.1,
            "align_xy_to_origin": False,
        }
        # No terrain to perceive on flat ground: remove the height scanner, its policy/critic
        # observations, and the terrain-difficulty curriculum (mirrors the official flat cfg).
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None
        # base_height reverts to absolute world-z on flat ground (the rough base pointed it at
        # the now-removed height_scanner). None => original root_height_below_minimum behavior.
        self.terminations.base_height.params["sensor_cfg"] = None
        _configure_flat_commands_and_events(self)


@configclass
class G1AmpFlatEnvCfg_PLAY(G1AmpRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # revert terrain to plane and strip the rough-only perception (same as
        # G1AmpFlatEnvCfg, but this PLAY variant derives from the rough PLAY config)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None
        self.terminations.base_height.params["sensor_cfg"] = None
        # This PLAY variant derives from the rough PLAY chain (not G1AmpFlatEnvCfg), so it inherits
        # reset_from_ref with align_xy_to_origin=True (the rough default). On plane terrain
        # env_origins.z == 0 and the reference motion's absolute xy is valid ground, so match the
        # non-play G1AmpFlatEnvCfg and keep the reference xy (DeepMimic-style reset).
        self.events.reset_from_ref.params["align_xy_to_origin"] = False
        _configure_flat_commands_and_events(self)
