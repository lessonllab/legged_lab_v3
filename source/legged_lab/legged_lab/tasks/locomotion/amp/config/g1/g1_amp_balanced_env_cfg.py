from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils.configclass import configclass
from .g1_amp_scratch_env_cfg import G1AmpScratchEnvCfg, ScratchCommandCfg
from legged_lab.tasks.locomotion.amp.mdp.balanced_course import BalancedTargetCommand, balanced_curriculum


@configclass
class BalancedCommandCfg(ScratchCommandCfg):
    class_type = BalancedTargetCommand


@configclass
class G1AmpBalancedEnvCfg(G1AmpScratchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = BalancedCommandCfg(speed_ranges=dict(self.commands.base_velocity.speed_ranges))
        self.curriculum.terrain_levels = CurriculumTermCfg(func=balanced_curriculum)


@configclass
class G1AmpBalancedEnvCfg_PLAY(G1AmpBalancedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.reset_robot_joints.params['position_range'] = (0., 0.)
