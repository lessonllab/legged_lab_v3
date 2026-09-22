"""V12 continuation: low-speed steps, contact-aware turns, block rehearsal."""
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, CurriculumTermCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.configclass import configclass
from .g1_amp_stairs_long_env_cfg import G1AmpStairsLongEnvCfg, LongStairTargetCommand
from legged_lab.tasks.locomotion.amp.mdp.foothold_course import FootholdCourseMixin, foothold_curriculum
from legged_lab.tasks.locomotion.amp.mdp import foothold_rewards


class FootholdCommand(FootholdCourseMixin,LongStairTargetCommand):
    pass


@configclass
class G1AmpFootholdEnvCfg(G1AmpStairsLongEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.class_type=FootholdCommand
        self.curriculum.terrain_levels=CurriculumTermCfg(func=foothold_curriculum)
        self.rewards.track_lin_vel_xy_exp=RewardTermCfg(func=foothold_rewards.useful_tracking,weight=3.)
        self.rewards.track_ang_vel_z_exp=RewardTermCfg(func=foothold_rewards.useful_tracking,weight=1.5,params={'angular':True})
        self.rewards.is_alive.weight=.6
        self.rewards.dont_wait=None
        # Preserve existing foot-volume edge/toe/support costs. Replace only the
        # staircase-only clearance term with a mesh-based, short-horizon probe.
        self.rewards.feet_swing_clearance=None
        for name,weight,kind in (('sustained_stall',-1.,'stall'),('turn_scuff',-.4,'turn_scuff'),
            ('turn_step',.08,'turn_step'),('terrain_clearance',-.6,'path_clearance'),('rough_impact',-.3,'rough_impact')):
            setattr(self.rewards,name,RewardTermCfg(func=foothold_rewards.foothold_signal,weight=weight,params={'kind':kind}))
        # Do not reward merely holding one leg up during a pure turn.
        self.rewards.feet_air_time.weight=.15
        self.rewards.feet_air_time.func=foothold_rewards.walking_air_time
        self.rewards.heading_error.func=foothold_rewards.navigation_heading_error
        feet=[f'{s}_ankle_roll_link' for s in ('left','right')]
        self.rewards.rough_support=RewardTermCfg(func=foothold_rewards.rough_support,weight=-.3,params={
            'asset_cfg':SceneEntityCfg('robot',body_names=feet,preserve_order=True),
            'contact_cfg':SceneEntityCfg('contact_forces',body_names=feet,preserve_order=True),
            'sensor_names':['left_sole_scanner','right_sole_scanner']})
        for side in ('left','right'):
            setattr(self.scene,f'{side}_path_scanner',RayCasterCfg(
                prim_path=f'{{ENV_REGEX_NS}}/Robot/{side}_ankle_roll_link',ray_alignment='yaw',
                pattern_cfg=patterns.PatternBaseCfg(func=foothold_rewards.path_ray_pattern),
                mesh_prim_paths=['/World/ground'],max_distance=2.,update_period=self.sim.dt*self.decimation,debug_vis=False))


@configclass
class G1AmpFootholdEnvCfg_PLAY(G1AmpFootholdEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=20
        self.curriculum.terrain_levels=None
        self.observations.policy.enable_corruption=False
        self.events.reset_robot_joints.params['position_range']=(0.,0.)
