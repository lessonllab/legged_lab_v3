"""Flat straight-walking warm-up with the same visual actor/critic input sizes."""

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import RewardTermCfg
from isaaclab.utils.configclass import configclass

from legged_lab.tasks.locomotion.amp.mdp.rewards import dont_wait
from .g1_amp_depth_env_cfg import G1AmpDepthEnvCfg


@configclass
class G1AmpDepthWarmupEnvCfg(G1AmpDepthEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        # Retain both sensors and critic height input to preserve checkpoint shapes.
        # Flat depth is only an input-interface warm-up, not evidence of terrain perception.
        self.commands.base_velocity = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(1000.0, 1000.0),
            rel_standing_envs=0.0,
            heading_command=False,
            rel_heading_envs=0.0,
            debug_vis=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=None,
            ),
        )
        # Fixed commands apply immediately on reset, instead of being overwritten
        # by AmpVelocityCommand's reference-velocity alignment.
        self.events.reset_from_ref.params["align_xy_to_origin"] = True
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.dont_wait = RewardTermCfg(
            func=dont_wait, weight=-0.5,
            params={"command_name": "base_velocity", "startup_grace_s": 0.5},
        )


@configclass
class G1AmpDepthWarmupEnvCfg_PLAY(G1AmpDepthWarmupEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
