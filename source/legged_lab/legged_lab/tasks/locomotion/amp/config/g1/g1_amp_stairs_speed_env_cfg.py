"""26500-compatible continuation: safer contact, flat 3 m/s, stairs 1.5 m/s."""
from isaaclab.utils.configclass import configclass
from .g1_amp_stairs_env_cfg import G1AmpStairsEnvCfg, StairCommandCfg
from legged_lab.tasks.locomotion.amp.mdp.stair_speed_course import StairSpeedTargetCommand


@configclass
class StairSpeedCommandCfg(StairCommandCfg):
    class_type = StairSpeedTargetCommand


@configclass
class G1AmpStairsSpeedEnvCfg(G1AmpStairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = StairSpeedCommandCfg(
            speed_ranges=dict(self.commands.base_velocity.speed_ranges))
        # Moderate contact shaping; network inputs and AMP references stay compatible.
        for kind, weight in (('support', -.5), ('edge', -.2)):
            term = getattr(self.rewards, f'feet_{kind}')
            term.weight = weight
            term.params.update(support_target=.9, contact_ramp=.04)
        self.rewards.track_lin_vel_xy_exp.weight = 3.


@configclass
class G1AmpStairsSpeedEnvCfg_PLAY(G1AmpStairsSpeedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.reset_robot_joints.params['position_range'] = (0., 0.)
