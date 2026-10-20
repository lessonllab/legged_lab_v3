"""Two-stage G1 baseline using native Sim6 managers and official G1 actuation."""
from copy import deepcopy
import math
import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.utils.configclass import configclass
from ..unitree_official.env_cfg import RobotEnvCfg, ObservationsCfg, RewardsCfg as OfficialRewardsCfg, EventCfg as OfficialEventCfg, CommandsCfg as OfficialCommandsCfg, CurriculumCfg as OfficialCurriculumCfg, ActionsCfg as OfficialActionsCfg, TerminationsCfg as OfficialTerminationsCfg
from . import mdp, paper_mdp, paper_terrain, repo_mdp, upstream_mdp, unitree_motion_mdp, course_tracking


@configclass
class CourseTerrainCfg(terrain_gen.MeshPlaneTerrainCfg):
    function = mdp.course_terrain
    beamdojo_stage: int = 1
    beamdojo_num_stones: int = 2


@configclass
class BeamDojoObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        gait_phase = ObsTerm(func=mdp.phase)

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        gait_phase = ObsTerm(func=mdp.phase)

    @configclass
    class TerrainCfg(ObsGroup):
        ideal_heightmap = ObsTerm(func=mdp.terrain_observation)

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    terrain: TerrainCfg = TerrainCfg()


@configclass
class BeamDojoEnvCfg(RobotEnvCfg):
    beamdojo_paper_aligned: bool = False
    beamdojo_repo_aligned: bool = False
    beamdojo_upstream_aligned: bool = False
    beamdojo_unitree_motion_aligned: bool = False
    beamdojo_course_integrity: bool = False
    beamdojo_course_tracking: bool = False
    beamdojo_physical_aligned: bool = False
    beamdojo_physical_continuous: bool = False
    beamdojo_recovery: bool = False
    recovery_max_level: int = 2
    recovery_neutral_unattempted: bool = False
    recovery_yaw_half_range: float = math.pi / 12
    recovery_easy_replay_fraction: float = .2
    recovery_challenge_fraction: float = 0.
    recovery_challenge_levels: tuple = (7, 8, 9, 10)
    beamdojo_allowed_unsupported_fraction: float = .3
    beamdojo_strict_foothold: bool = False
    beamdojo_virtual_handoff: bool = False
    beamdojo_bound_joint_actions: bool = False
    beamdojo_course_schedule: str = 'legacy_9'
    beamdojo_virtual_course: bool = False
    beamdojo_match_virtual_query: bool = False
    tracking_gyro_noise_std: float = .01
    tracking_gyro_bias: float = .001
    tracking_straight_fraction: float = .4
    beamdojo_stage1_map_size: float = 32.
    beamdojo_stage2_course_length: float = 16.
    beamdojo_scanner_profile: str = 'beamdojo_paper'
    beamdojo_initial_level: int = 0
    beamdojo_curriculum: bool = True
    beamdojo_paper_noise: dict = {"height_bias":.03,"point_noise":.03,"gradient_endpoint":.03,"yaw_episode":.2,"repeat_probability":.2,"extension_probability":.6}
    beamdojo_stage: int = 1
    beamdojo_num_stones: int = 2
    observation_source: str = 'ideal_heightmap'
    beamdojo_map_noise: bool = True
    observations: BeamDojoObservationsCfg = BeamDojoObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1024
        self.scene.env_spacing = 8.
        self.episode_length_s = 14.
        self.curriculum = None
        self.compute_final_obs = True
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.visual_material = None
        self.configure_course()
        command = self.commands.base_velocity
        command.class_type = mdp.CourseVelocityCommand
        command.ranges.lin_vel_x = (.3, .5)
        command.ranges.lin_vel_y = (0., 0.)
        command.ranges.ang_vel_z = (0., 0.)
        command.limit_ranges = command.ranges.copy()
        command.rel_standing_envs = 0.
        command.debug_vis = False
        self.events.reset_base.params['pose_range'] = {
            'x': (-.02,.02), 'y': (-.01,.01), 'yaw': (-.025,.025)}
        self.events.reset_robot_joints.params['velocity_range'] = (-.1,.1)
        self.events.push_robot = None
        # Shoulder pitch remains a learned action, with a dynamic arm reference.
        self.rewards.joint_deviation_arms.params['asset_cfg'] = SceneEntityCfg(
            'robot', joint_names=['.*_shoulder_roll_joint', '.*_shoulder_yaw_joint', '.*_elbow_joint', '.*_wrist_.*'])
        self.rewards.joint_deviation_arms.weight = -.05
        self.rewards.base_height.func = mdp.relative_base_height
        self.rewards.feet_clearance = None
        feet = SceneEntityCfg('robot', body_names=['left_ankle_roll_link','right_ankle_roll_link'], preserve_order=True)
        contacts = SceneEntityCfg('contact_forces', body_names=['left_ankle_roll_link','right_ankle_roll_link'], preserve_order=True)
        self.rewards.gait = RewTerm(func=mdp.course_gait, weight=.5, params={'sensor_cfg': contacts})
        self.rewards.low_speed_flight = RewTerm(func=mdp.low_speed_flight, weight=-1., params={'sensor_cfg': contacts})
        self.rewards.foothold = RewTerm(func=mdp.foothold, weight=2., params={'asset_cfg': feet, 'sensor_cfg': contacts})
        arms = SceneEntityCfg('robot', joint_names=['.*_shoulder_.*_joint', '.*_elbow_joint', '.*_wrist_.*'])
        self.rewards.arm_excess_speed = RewTerm(func=mdp.arm_excess_speed, weight=-.02, params={'asset_cfg': arms})
        self.rewards.arm_excess_amplitude = RewTerm(func=mdp.arm_excess_amplitude, weight=-1., params={'asset_cfg': arms})
        self.rewards.natural_arm_swing = RewTerm(func=mdp.natural_arm_swing, weight=.3, params={
            'asset_cfg': SceneEntityCfg('robot', joint_names=['left_shoulder_pitch_joint','right_shoulder_pitch_joint'], preserve_order=True)})
        self.rewards.feet_air_time = RewTerm(func=mdp.feet_air_time, weight=.2, params={'sensor_cfg': contacts})
        self.rewards.feet_reference = RewTerm(func=mdp.feet_reference, weight=.3, params={'asset_cfg': feet})
        self.terminations.base_height = None
        self.terminations.left_course = DoneTerm(func=mdp.left_course)
        self.terminations.finished = DoneTerm(func=mdp.finished, params={'asset_cfg': feet, 'sensor_cfg': contacts})

    def configure_course(self):
        if self.beamdojo_upstream_aligned or self.beamdojo_unitree_motion_aligned:
            from . import upstream_terrain
            upstream_terrain.configure(self)
            return
        if self.beamdojo_paper_aligned:
            self.configure_paper_terrain()
            return
        if self.beamdojo_stage not in (1,2):
            raise ValueError('beamdojo_stage must be 1 or 2')
        mdp.course_rectangles(self.beamdojo_num_stones)
        self.scene.terrain.terrain_type = 'generator'
        self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            seed=self.seed, size=(8.,4.), border_width=0., num_rows=1, num_cols=1,
            curriculum=False, use_cache=False,
            sub_terrains={'beamdojo': CourseTerrainCfg(proportion=1.,
                beamdojo_stage=self.beamdojo_stage, beamdojo_num_stones=self.beamdojo_num_stones)})
        # Each robot receives a separate copy of the fixed course, not 1024
        # overlapping robots sharing the single terrain-generator origin.
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = self.scene.num_envs

    def configure_paper_terrain(self):
        if self.beamdojo_stage not in (1,2) or not 0<=self.beamdojo_initial_level<=8:
            raise ValueError('Paper profile requires stage 1/2 and initial_level 0..8')
        terrain=self.scene.terrain
        terrain.terrain_type='generator'
        terrain.max_init_terrain_level=0
        terrain.terrain_generator=terrain_gen.TerrainGeneratorCfg(
            seed=self.seed,size=(8.,8. if self.beamdojo_stage==1 else 2.),border_width=0.,
            num_rows=1 if self.beamdojo_stage==1 else 9,num_cols=self.scene.num_envs,
            curriculum=self.beamdojo_stage==2,use_cache=False,difficulty_range=(0.,1.),
            sub_terrains={'paper':PaperTerrainCfg(proportion=1.,beamdojo_stage=self.beamdojo_stage,layout_seed=self.seed)})

    def configure_paper_profile(self):
        self.episode_length_s=20.
        self.observations.policy.gait_phase=None
        self.observations.critic.gait_phase=None
        command=self.commands.base_velocity
        command.class_type=paper_mdp.PaperVelocityCommand
        command.ranges.lin_vel_x=(-1.,1.)
        command.ranges.lin_vel_y=(-1.,1.) if self.beamdojo_stage==1 else (0.,0.)
        command.ranges.ang_vel_z=(-1.,1.) if self.beamdojo_stage==1 else (0.,0.)
        command.limit_ranges=command.ranges.copy()
        command.rel_standing_envs=0.
        command.rel_heading_envs=0.
        self.events.reset_base.func=paper_mdp.PaperResetRoot
        self.events.reset_base.params['pose_range']={'x':(-.05,.05),'y':(-.05,.05),
            'yaw':(-3.14159,3.14159) if self.beamdojo_stage==1 else (-.025,.025)}
        self.events.physics_material.params['static_friction_range']=(.4,1.)
        self.events.physics_material.params['dynamic_friction_range']=(.4,1.)
        self.events.physics_material.params['restitution_range']=(0.,1.)
        self.events.add_base_mass.params['mass_distribution_params']=(-2.,2.)
        self.observations.policy.base_ang_vel.noise.n_min=-.5
        self.observations.policy.base_ang_vel.noise.n_max=.5
        self.observations.policy.joint_pos_rel.noise.n_min=-.05
        self.observations.policy.joint_pos_rel.noise.n_max=.05
        self.observations.policy.joint_vel_rel.noise.n_min=-2.
        self.observations.policy.joint_vel_rel.noise.n_max=2.
        r=self.rewards
        # Table VII has no periodic contact/foot-position gait target.
        for name in ('gait','feet_reference','low_speed_flight','alive','joint_deviation_legs',
                     'joint_deviation_waists','joint_deviation_arms','feet_slide','undesired_contacts'):
            setattr(r,name,None)
        r.track_lin_vel_xy.weight=1.
        r.track_ang_vel_z.weight=1.
        r.flat_orientation_l2.weight=-2.
        r.action_rate.weight=-.01
        r.joint_vel.weight=-1e-4
        r.joint_acc.weight=-2.5e-8
        r.energy=RewTerm(func=paper_mdp.joint_power,weight=-2e-5)
        r.action_smoothness=RewTerm(func=paper_mdp.action_smoothness,weight=-1e-3)
        r.stand_still=RewTerm(func=paper_mdp.stand_still,weight=-.05)
        r.joint_velocity_limits=RewTerm(func=paper_mdp.joint_velocity_limits,weight=-1e-3)
        r.feet_parallel=RewTerm(func=paper_mdp.feet_parallel,weight=-.02)
        r.feet_distance=RewTerm(func=paper_mdp.feet_distance,weight=.5)
        r.feet_air_time=RewTerm(func=paper_mdp.feet_air_time,weight=1.)
        r.feet_clearance=RewTerm(func=paper_mdp.feet_clearance,weight=-1.)
        r.foothold=RewTerm(func=paper_mdp.foothold,weight=1.)
        # User-requested 29-DoF extension, explicitly beyond the paper's 12-DoF actor.
        r.natural_arm_swing=RewTerm(func=paper_mdp.natural_arm_swing,weight=.1,params={
            'asset_cfg':SceneEntityCfg('robot',joint_names=['left_shoulder_pitch_joint','right_shoulder_pitch_joint'],preserve_order=True),
            'hip_cfg':SceneEntityCfg('robot',joint_names=['left_hip_pitch_joint','right_hip_pitch_joint'],preserve_order=True)})
        self.terminations.finished=DoneTerm(func=paper_mdp.finished)
        self.terminations.left_course=DoneTerm(func=paper_mdp.left_course)
        self.configure_paper_terrain()

    def configure_repo_profile(self):
        """Reward-aligned upstream profile; keep paper commands/terrain/PPO separate."""
        self.beamdojo_paper_aligned=True
        self.beamdojo_repo_aligned=True
        self.configure_paper_profile()
        # Replace, do not mutate, to guarantee that no legacy/paper terms remain.
        self.rewards=RepoRewardsCfg()
        self.observations.policy.gait_phase=ObsTerm(func=repo_mdp.phase)
        self.observations.critic.gait_phase=ObsTerm(func=repo_mdp.phase)


    def configure_upstream_profile(self):
        """Source configuration: ezoicoder training + Unitree G1 full-body control.

        Independent of configure_paper_profile: no paper reset, finish hold,
        handmade arm targets, or paper command/noise values leak into this mode.
        """
        from ..unitree_official import mdp as native
        from isaaclab.managers import EventTermCfg as EventTerm
        self.beamdojo_upstream_aligned = True
        self.beamdojo_repo_aligned = True
        self.beamdojo_paper_aligned = False
        self.episode_length_s = 96.
        self.curriculum = None
        self.observations = UpstreamObservationsCfg()
        self.rewards = UpstreamRewardsCfg()
        self.terminations = UpstreamTerminationsCfg()
        self.events = OfficialEventCfg()
        command = self.commands.base_velocity
        command.class_type = upstream_mdp.UpstreamVelocityCommand
        command.resampling_time_range = (8., 8.)
        command.ranges.lin_vel_x = (-.3, .6)
        command.ranges.lin_vel_y = (-.3, .3)
        command.ranges.ang_vel_z = (-.3, .3)
        command.ranges.heading = (-3.14, 3.14)
        command.limit_ranges = command.ranges.copy()
        command.heading_command = True
        command.heading_control_stiffness = .5
        command.rel_heading_envs = 1.
        command.rel_standing_envs = 0.
        command.debug_vis = False
        reset_pose = ({'x': (-1., 1.), 'y': (-1., 1.)} if self.beamdojo_stage == 1
                      else {'x': (-3.5, -3.5), 'y': (-.3, .3)})
        self.events.reset_base = EventTerm(func=upstream_mdp.UpstreamResetRoot, mode='reset',
            params={'pose_range': reset_pose, 'velocity_range': {}})
        self.events.reset_robot_joints.func = native.reset_joints_by_offset
        self.events.reset_robot_joints.params = {'position_range': (-.1, .1), 'velocity_range': (0., 0.)}
        self.events.physics_material = EventTerm(func=upstream_mdp.randomize_friction, mode='startup')
        # G1 body-mass adaptation: retain the complete Unitree torso event.
        # XBot root +/-5 kg can make G1's lighter pelvis mass negative.
        self.events.add_base_mass = deepcopy(OfficialEventCfg().add_base_mass)
        self.events.push_robot = EventTerm(func=upstream_mdp.push_robot, mode='interval', interval_range_s=(4., 4.))
        self.configure_course()


    def configure_unitree_motion_profile(self):
        """All 19 pinned Unitree locomotion rewards plus BeamDojo foothold.

        Unitree commit 4960b847's motion configuration is copied as a whole.
        BeamDojo owns terrain generation, perception and terrain progression;
        Unitree's linear-command curriculum remains unchanged. Its yaw limit
        range is not a yaw curriculum: training stays at [-0.1, 0.1].
        """
        self.beamdojo_unitree_motion_aligned = True
        self.beamdojo_upstream_aligned = False
        self.beamdojo_repo_aligned = False
        self.beamdojo_paper_aligned = False
        self.episode_length_s = 20.
        self.actions = deepcopy(OfficialActionsCfg())
        self.commands = deepcopy(OfficialCommandsCfg())
        self.commands.base_velocity.debug_vis = False
        self.rewards = deepcopy(OfficialRewardsCfg())
        self.rewards.foothold = RewTerm(func=unitree_motion_mdp.foothold,
                                       weight=repo_mdp.REWARD_WEIGHTS['foothold'])
        self.terminations = deepcopy(OfficialTerminationsCfg())
        self.events = deepcopy(OfficialEventCfg())
        # Keep Unitree root reset ranges while updating BeamDojo terrain origins.
        self.events.reset_base.func = upstream_mdp.UpstreamResetRoot
        if self.beamdojo_stage == 2:
            # Source stepping-stone start platform; the center is a pit.
            self.events.reset_base.params['pose_range'] = {
                'x': (-3.5, -3.5), 'y': (-.3, .3), 'yaw': (0., 0.)}
        self.observations = UnitreeMotionObservationsCfg()
        self.curriculum = deepcopy(OfficialCurriculumCfg())
        self.curriculum.terrain_levels = None  # BeamDojo terrain curriculum
        self.configure_course()

    def configure_course_integrity(self):
        """Task boundaries and finish conditions; official motion rewards stay intact."""
        from . import upstream_terrain
        if not self.beamdojo_unitree_motion_aligned:
            raise ValueError('Course integrity currently requires the Unitree-motion profile')
        self.beamdojo_course_integrity = True
        self.is_finite_horizon = False
        self.terminations.course_boundary = DoneTerm(
            func=upstream_terrain.course_boundary, time_out=True)
        self.terminations.course_finished = DoneTerm(
            func=upstream_terrain.course_finished, time_out=True)
        if self.beamdojo_stage == 2:
            from .course_command import CourseVelocityCommand
            self.commands.base_velocity.class_type = CourseVelocityCommand
        self.configure_course()

    def configure_course_tracking(self):
        """IMU-only actor feedback; simulation pose is reserved for reward shaping."""
        if not self.beamdojo_unitree_motion_aligned or not self.beamdojo_course_integrity:
            raise ValueError('Course tracking requires the corrected Unitree-motion course')
        self.beamdojo_course_tracking = True
        from .course_command import CourseTrackingVelocityCommand
        self.commands.base_velocity.class_type = CourseTrackingVelocityCommand
        self.observations.course_tracking = CourseTrackingObservationsCfg()
        self.rewards.course_heading = RewTerm(func=course_tracking.heading_error_l2,
                                              weight=-.5, params={'scale': .5})
        self.rewards.course_lateral = RewTerm(func=course_tracking.lateral_error_l2,
                                              weight=-.5, params={'scale': .3})

    def configure_physical_alignment(self):
        """Use the virtual raster as real colliders; partial soles are allowed."""
        if not self.beamdojo_course_tracking or not self.beamdojo_course_integrity:
            raise ValueError('Physical alignment requires the IMU corrected course')
        self.beamdojo_physical_aligned = True
        self.configure_course()
        if self.beamdojo_stage == 2:
            self.scene.terrain.max_init_terrain_level = 2
            self.events.reset_base.params['pose_range'] = {
                'x': (-.15, .15), 'y': (-.15, .15), 'yaw': (0., 0.)}

    def configure_continuous_physical(self):
        """Continue past the passage milestone and randomize initial heading."""
        if not self.beamdojo_physical_aligned or self.beamdojo_stage != 2:
            raise ValueError('Continuous passage requires aligned physical Stage 2')
        self.beamdojo_physical_continuous = True
        self.events.reset_base.params['pose_range']['yaw'] = (-math.pi, math.pi)

    def configure_recovery(self):
        """Opt-in adaptation of a walking actor to the continuous physical task."""
        if not self.beamdojo_physical_continuous or not self.beamdojo_unitree_motion_aligned:
            raise ValueError('Recovery requires continuous Unitree-motion physical training')
        from . import recovery_rewards
        self.beamdojo_recovery = True
        # Retain all official numerical weights and parameters. These wrappers
        # remove positive lift/gait credit for failing a translation command.
        self.rewards.feet_clearance.func = recovery_rewards.feet_clearance_moving
        self.rewards.gait.func = recovery_rewards.gait_moving
        self.rewards.command_stall = RewTerm(func=recovery_rewards.command_stall,weight=-1.)

    def configure_virtual_handoff(self):
        """Restore official motion rewards while retaining the source inputs/commands."""
        if not (self.beamdojo_stage == 2 and self.beamdojo_unitree_motion_aligned
                and self.beamdojo_course_tracking and self.beamdojo_physical_continuous
                and self.beamdojo_recovery and self.beamdojo_strict_foothold):
            raise ValueError('Virtual handoff requires strict continuous physical recovery Stage 2')
        self.beamdojo_virtual_handoff = True
        self.beamdojo_match_virtual_query = True
        self.rewards = deepcopy(OfficialRewardsCfg())
        self.rewards.foothold = RewTerm(func=unitree_motion_mdp.foothold,
                                       weight=repo_mdp.REWARD_WEIGHTS['foothold'])

    def configure_action_bounds(self):
        """Declare opt-in execution bounds without changing the PPO action space."""
        if not (self.beamdojo_virtual_handoff and self.beamdojo_stage == 2):
            raise ValueError('Joint action bounds require the virtual-handoff Stage 2 task')
        self.beamdojo_bound_joint_actions = True

    def configure_virtual_course(self):
        """Retain the continuous course protocol on flat collision geometry."""
        if not (self.beamdojo_stage == 2 and self.beamdojo_virtual_handoff
                and self.beamdojo_bound_joint_actions
                and self.beamdojo_course_schedule == 'smooth_top_v1'
                and not self.beamdojo_recovery_speed_gate):
            raise ValueError('Virtual course requires the v16 smooth bounded handoff task')
        self.beamdojo_virtual_course = True
        self.configure_course()


@configclass
class RepoRewardsCfg:
    """Exactly 23 upstream active terms, plus three named full-body extensions."""
    joint_pos=RewTerm(func=repo_mdp.joint_pos,weight=repo_mdp.REWARD_WEIGHTS['joint_pos'])
    feet_clearance=RewTerm(func=repo_mdp.feet_clearance,weight=repo_mdp.REWARD_WEIGHTS['feet_clearance'])
    feet_contact_number=RewTerm(func=repo_mdp.feet_contact_number,weight=repo_mdp.REWARD_WEIGHTS['feet_contact_number'])
    feet_air_time=RewTerm(func=repo_mdp.feet_air_time,weight=repo_mdp.REWARD_WEIGHTS['feet_air_time'])
    foot_slip=RewTerm(func=repo_mdp.foot_slip,weight=repo_mdp.REWARD_WEIGHTS['foot_slip'])
    feet_distance=RewTerm(func=repo_mdp.feet_distance,weight=repo_mdp.REWARD_WEIGHTS['feet_distance'])
    knee_distance=RewTerm(func=repo_mdp.knee_distance,weight=repo_mdp.REWARD_WEIGHTS['knee_distance'])
    feet_contact_forces=RewTerm(func=repo_mdp.feet_contact_forces,weight=repo_mdp.REWARD_WEIGHTS['feet_contact_forces'])
    tracking_lin_vel=RewTerm(func=repo_mdp.tracking_lin_vel,weight=repo_mdp.REWARD_WEIGHTS['tracking_lin_vel'])
    tracking_ang_vel=RewTerm(func=repo_mdp.tracking_ang_vel,weight=repo_mdp.REWARD_WEIGHTS['tracking_ang_vel'])
    vel_mismatch_exp=RewTerm(func=repo_mdp.vel_mismatch_exp,weight=repo_mdp.REWARD_WEIGHTS['vel_mismatch_exp'])
    low_speed=RewTerm(func=repo_mdp.low_speed,weight=repo_mdp.REWARD_WEIGHTS['low_speed'])
    track_vel_hard=RewTerm(func=repo_mdp.track_vel_hard,weight=repo_mdp.REWARD_WEIGHTS['track_vel_hard'])
    default_joint_pos=RewTerm(func=repo_mdp.default_joint_pos,weight=repo_mdp.REWARD_WEIGHTS['default_joint_pos'])
    orientation=RewTerm(func=repo_mdp.orientation,weight=repo_mdp.REWARD_WEIGHTS['orientation'])
    base_height=RewTerm(func=repo_mdp.base_height,weight=repo_mdp.REWARD_WEIGHTS['base_height'])
    base_acc=RewTerm(func=repo_mdp.base_acc,weight=repo_mdp.REWARD_WEIGHTS['base_acc'])
    action_smoothness=RewTerm(func=repo_mdp.action_smoothness,weight=repo_mdp.REWARD_WEIGHTS['action_smoothness'])
    torques=RewTerm(func=repo_mdp.torques,weight=repo_mdp.REWARD_WEIGHTS['torques'])
    dof_vel=RewTerm(func=repo_mdp.dof_vel,weight=repo_mdp.REWARD_WEIGHTS['dof_vel'])
    dof_acc=RewTerm(func=repo_mdp.dof_acc,weight=repo_mdp.REWARD_WEIGHTS['dof_acc'])
    collision=RewTerm(func=repo_mdp.collision,weight=repo_mdp.REWARD_WEIGHTS['collision'])
    foothold=RewTerm(func=repo_mdp.foothold,weight=repo_mdp.REWARD_WEIGHTS['foothold'])
    extension_natural_arm_swing=RewTerm(func=paper_mdp.natural_arm_swing,weight=.1,params={
        'asset_cfg':SceneEntityCfg('robot',joint_names=['left_shoulder_pitch_joint','right_shoulder_pitch_joint'],preserve_order=True),
        'hip_cfg':SceneEntityCfg('robot',joint_names=['left_hip_pitch_joint','right_hip_pitch_joint'],preserve_order=True)})
    extension_arm_excess_speed=RewTerm(func=mdp.arm_excess_speed,weight=-.02,params={
        'asset_cfg':SceneEntityCfg('robot',joint_names=['.*_shoulder_.*_joint','.*_elbow_joint','.*_wrist_.*'])})
    extension_arm_excess_amplitude=RewTerm(func=mdp.arm_excess_amplitude,weight=-1.,params={
        'asset_cfg':SceneEntityCfg('robot',joint_names=['.*_shoulder_.*_joint','.*_elbow_joint','.*_wrist_.*'])})


@configclass
class UpstreamRewardsCfg(RepoRewardsCfg):
    """23 ezoicoder terms plus the two unchanged Unitree upper-body terms."""
    extension_natural_arm_swing = None
    extension_arm_excess_speed = None
    extension_arm_excess_amplitude = None
    joint_deviation_arms = deepcopy(OfficialRewardsCfg().joint_deviation_arms)
    joint_deviation_waists = deepcopy(OfficialRewardsCfg().joint_deviation_waists)


@configclass
class UpstreamTerminationsCfg:
    # G1-specific failure conditions from Unitree's 29dof/velocity_env_cfg.py.
    # XBot base_link contact alone does not cover G1's separate torso/head
    # collisions: the robot can lie on its torso without touching its pelvis.
    # Both stages have their top support surface at world z=0, matching the
    # official root-height condition's flat-ground coordinate convention.
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    time_out = DoneTerm(func=upstream_mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=upstream_mdp.base_contact)


def upstream_heightmap(env):
    from .upstream_perception import heightmap
    return heightmap(env)


@configclass
class UpstreamObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprioception = ObsTerm(func=upstream_mdp.actor_observation)
        def __post_init__(self):
            self.history_length = 0  # source zero-filled 15 frames in observation function
            self.enable_corruption = True
            self.concatenate_terms = True
    @configclass
    class CriticCfg(ObsGroup):
        proprioception = ObsTerm(func=upstream_mdp.critic_observation)
        def __post_init__(self):
            self.history_length = 0  # source zero-filled 3 frames in observation function
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class TerrainCfg(ObsGroup):
        # Resolved lazily so the upstream map remains isolated from old modes.
        ideal_heightmap = ObsTerm(func=upstream_heightmap)
        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    terrain: TerrainCfg = TerrainCfg()


@configclass
class CourseTrackingObservationsCfg(ObsGroup):
    imu_relative_heading = ObsTerm(func=course_tracking.observation)

    def __post_init__(self):
        self.history_length = 1
        # Sensor noise lives inside the gyro estimator and is sampled once/tick.
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class UnitreeMotionObservationsCfg(ObservationsCfg):
    """Unmodified 5-frame Unitree proprioception plus current BeamDojo map."""
    terrain: UpstreamObservationsCfg.TerrainCfg = UpstreamObservationsCfg.TerrainCfg()
    course_tracking: CourseTrackingObservationsCfg | None = None


@configclass
class PaperTerrainCfg(terrain_gen.MeshPlaneTerrainCfg):
    function=paper_terrain.mesh_terrain
    beamdojo_stage:int=1
    layout_seed:int=42


@configclass
class BeamDojoStage2EnvCfg(BeamDojoEnvCfg):
    beamdojo_stage: int = 2


def build_env_cfg(stage=1, num_stones=2, num_envs=1024, device='cuda:0', seed=42,
                  paper_aligned=False, initial_level=0, curriculum=True, repo_aligned=False, upstream_aligned=False, unitree_motion_aligned=False):
    cfg = BeamDojoEnvCfg()
    cfg.beamdojo_stage = stage
    cfg.beamdojo_num_stones = num_stones
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.seed = seed
    cfg.beamdojo_paper_aligned=paper_aligned or repo_aligned
    cfg.beamdojo_repo_aligned=repo_aligned
    cfg.beamdojo_initial_level=initial_level
    cfg.beamdojo_curriculum=curriculum
    if unitree_motion_aligned:
        cfg.configure_unitree_motion_profile()
    elif upstream_aligned:
        cfg.configure_upstream_profile()
    elif repo_aligned:
        cfg.configure_repo_profile()
    elif paper_aligned:
        cfg.configure_paper_profile()
    cfg.configure_course()
    return cfg
