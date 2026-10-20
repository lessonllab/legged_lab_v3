"""Native square-course migration; imports before build remain CPU safe."""
from copy import deepcopy
import sys
import numpy as np
from beamdojo_square_course_geometry import layout, rasterize


def install_geometry(cfg, courses=None):
    import torch
    from legged_lab.tasks.locomotion.beamdojo import upstream_terrain as terrain
    import beamdojo_square_course_runtime as runtime
    courses = courses if courses is not None else [layout(cfg.seed, level=i/10) for i in range(11)]
    count = len(courses)
    # Existing mesh merger and query use [x,y]. Geometry module uses [y,x].
    grids = np.stack([rasterize(c)['observed_height_m'].T for c in courses])
    width = max(len(c['rectangles_m']) for c in courses)
    boxes = torch.zeros(count,width,4)
    display = torch.zeros(count,width,5)
    masks = torch.zeros(count,width,dtype=torch.bool)
    stones = masks.clone()
    for i,c in enumerate(courses):
        r=torch.tensor(c['rectangles_m']); n=len(r)
        boxes[i,:n]=r
        display[i,:n,:2]=(r[:,:2]+r[:,2:])/2
        display[i,:n,2:4]=r[:,2:]-r[:,:2]
        masks[i,:n]=True; stones[i,5:n]=True
    def heightfield(stage,level,seed=5,size=(8.,8.),schedule='legacy_9'):
        if tuple(size)!=(8.,8.) or not 0<=level<count:
            raise ValueError('Unexpected square terrain size/level')
        return grids[level].copy()
    def display_rectangles(stage,level,seed=5,size=(8.,8.),schedule='legacy_9'):
        return display[level,masks[level]].tolist()
    def refresh(env):
        env._upstream_heightmap_cache=None
        ensure(env)
        lv=env.beamdojo_levels
        env.beamdojo_geometry=env._upstream_display[lv]
        env.beamdojo_rectangle_mask=env._upstream_display_mask[lv]
        env.beamdojo_stone_mask=env._upstream_stone_mask[lv]
        env.beamdojo_start_platform=env.beamdojo_geometry[:,0]
        env.beamdojo_end_platform=env.beamdojo_geometry[:,1]
        env.beamdojo_square_course_rectangles=env._square_rectangles[lv]
    def ensure(env):
        if hasattr(env,'_upstream_heightfields'):return
        env._upstream_heightfields=torch.as_tensor(grids,device=env.device)
        env._square_rectangles=boxes.to(env.device)
        env._upstream_display=display.to(env.device)
        env._upstream_display_mask=masks.to(env.device)
        env._upstream_stone_mask=stones.to(env.device)
        env.beamdojo_levels=torch.full((env.num_envs,),cfg.beamdojo_initial_level,device=env.device,dtype=torch.long)
        env.beamdojo_success_streak=torch.zeros_like(env.beamdojo_levels)
        collider=env.scene.terrain
        as_tensor=runtime.tensor
        as_tensor(collider.terrain_levels)[:]=env.beamdojo_levels
        origins=as_tensor(collider.terrain_origins)[env.beamdojo_levels,as_tensor(collider.terrain_types)]
        as_tensor(collider.env_origins)[:]=origins
        as_tensor(env.scene.env_origins)[:]=origins
        env._square_course_surface_height=lambda points:query(env,points)
        runtime.ensure_state(env)
        refresh(env)
    def query(env,points,*,exact=None):
        ensure(env)
        origin=runtime.tensor(env.scene.env_origins)
        local=(points-origin.reshape((env.num_envs,)+(1,)*(points.ndim-2)+(3,))).reshape(env.num_envs,-1,3)
        xy=local[...,:2]
        inside=torch.isfinite(xy).all(-1)&(xy>=-4).all(-1)&(xy<4).all(-1)
        px=((torch.nan_to_num(xy)+4)/.02).long().clamp(0,399)
        h=env._upstream_heightfields[env.beamdojo_levels[:,None],px[...,0],px[...,1]]
        h=torch.where(inside,h,-torch.ones_like(h))
        return (h+origin[:,2:3]).reshape(points.shape[:-1]),(inside&(h>=-.1)).reshape(points.shape[:-1])
    def before_reset(env,ids):
        ensure(env)
        experienced=env._upstream_started[ids]
        wins=env.beamdojo_qualified_success[ids]
        log={'Metrics/course_success_rate':(wins&experienced).float().sum()/experienced.sum().clamp_min(1)}
        runtime.reset_course(env,ids)
        refresh(env)
        return log
    replacements={terrain.heightfield:heightfield,terrain.display_rectangles:display_rectangles,
        terrain.ensure_state:ensure,terrain._refresh_display:refresh,terrain.refresh_geometry:refresh,
        terrain.surface_height:query,terrain._update_course:runtime.update_course,
        terrain.update_success:runtime.update_course,terrain.before_reset:before_reset,
        terrain.course_diagnostics:runtime.course_metrics}
    for module in list(sys.modules.values()):
        if module is None or '.beamdojo' not in getattr(module,'__name__',''):continue
        for key,value in list(vars(module).items()):
            try: replacement=replacements.get(value)
            except TypeError:continue
            if replacement is not None:setattr(module,key,replacement)
    cfg.beamdojo_square_course_geometry=courses
    return courses


def build(parent,num_envs,seed,device,level=0):
    from beamdojo_common import build_config, scanner_profile_for_contract
    from beamdojo_corridor_profile import GPU_PAIR_CAPACITIES
    from isaaclab.managers import TerminationTermCfg
    from legged_lab.tasks.locomotion.unitree_official import mdp as official
    import beamdojo_square_course_runtime as runtime
    if parent['action_dimension']!=29 or parent['observation_source']!='ideal_heightmap':
        raise ValueError('Square course requires compatible 29-DoF heightmap actor')
    cfg,agent=build_config(2,parent['num_stones'],num_envs,seed,device,'ideal_heightmap',
        paper_aligned=True,unitree_motion_aligned=True,
        unitree_scanner_profile=scanner_profile_for_contract(parent),initial_level=level,curriculum=False,
        course_integrity=True,course_tracking=True,physical_aligned=True,continuous_physical=True,
        recovery=True,recovery_max_level=10,recovery_neutral_unattempted=True,strict_foothold=True,
        virtual_handoff=True,bound_joint_actions=True,course_schedule='smooth_top_v1',recovery_speed_gate=False)
    cfg.beamdojo_stage1_map_size=8.
    cfg.beamdojo_recovery=False;cfg.beamdojo_curriculum=False;cfg.curriculum=None
    cfg.beamdojo_match_virtual_query=False
    cfg.beamdojo_physical_continuous=False
    cfg.beamdojo_corridor_training=False;cfg.beamdojo_corridor_mixed_levels=False
    cfg.beamdojo_support_fixed_levels=False;cfg.beamdojo_contact_event_risk=False
    cfg.configure_course()
    cfg.scene.terrain.max_init_terrain_level=level
    # Fixed-level 4096 actors otherwise share only four native origins. Even
    # with env collision filtering their broadphase overlap is unnecessarily
    # expensive. More identical tile columns preserve each local task.
    cfg.scene.terrain.terrain_generator.num_cols=64 if num_envs>=512 else 4
    for name,value in GPU_PAIR_CAPACITIES.items():setattr(cfg.sim.physics,name,value)
    cfg.episode_length_s=12.
    cfg.tracking_straight_fraction=1.
    geometry=layout(seed,level/10)
    cfg.beamdojo_square_course_params=dict(start_platform_m=geometry['start_platform_m'],
        end_platform_m=geometry['end_platform_m'],target_root_x=3.25,braking_start_x=3.05,
        route_half_width=.42,stand_seconds=1.,end_half_width=.6,boundary_limit=3.9)
    command=cfg.commands.base_velocity
    command.class_type=runtime.make_command_class()
    command.heading_command=False;command.rel_standing_envs=0.
    command.resampling_time_range=(12.,12.)
    for ranges in (command.ranges,command.limit_ranges):
        ranges.lin_vel_x=(.5,1.);ranges.lin_vel_y=(0.,0.);ranges.ang_vel_z=(0.,0.)
    # Native official reset class, avoiding legacy recovery yaw/origin updates.
    cfg.events.reset_base.func=official.reset_root_state_uniform
    cfg.events.reset_base.params['pose_range']={key:(0.,0.) for key in ('x','y','z','roll','pitch','yaw')}
    cfg.events.reset_base.params['velocity_range']={key:(0.,0.) for key in ('x','y','z','roll','pitch','yaw')}
    cfg.terminations.course_finished=TerminationTermCfg(func=runtime.course_success,time_out=False)
    cfg.terminations.course_boundary=TerminationTermCfg(func=runtime.course_boundary,time_out=False)
    cfg.terminations.time_out=TerminationTermCfg(func=runtime.time_out,time_out=True)
    install_geometry(cfg)
    return cfg,agent


def make_contract(base,cfg,parent):
    from beamdojo_common import reward_parameter_contract
    from legged_lab.tasks.locomotion.unitree_official.env_cfg import RewardsCfg
    result=deepcopy(base)
    for key in ('action_dimension','joint_names','asset_sha256','observation_dimensions','policy_hz','simulation_hz'):
        if result[key]!=parent[key]:raise ValueError(f'Actor/control migration mismatch: {key}')
    active={k:v for k,v in vars(cfg.rewards).items() if v is not None and hasattr(v,'weight') and v.weight!=0}
    result['reward_weights']={k:float(v.weight) for k,v in active.items()}
    result['reward_functions']={k:f'{v.func.__module__}.{v.func.__name__}' for k,v in active.items()}
    result['reward_parameters']={k:reward_parameter_contract(v.params) for k,v in active.items()}
    official=RewardsCfg()
    expected={k:v for k,v in vars(official).items() if v is not None and hasattr(v,'weight') and v.weight!=0}
    if set(active)!=(set(expected)|{'foothold'}):raise ValueError('Unexpected movement reward terms')
    for name,term in expected.items():
        actual=active[name]
        if actual.func is not term.func or actual.weight!=term.weight or reward_parameter_contract(actual.params)!=reward_parameter_contract(term.params):
            raise ValueError(f'Official movement reward changed: {name}')
    result['square_official_reward_identity_checked']=True
    result['map']['surface_source']='same 2cm cell-center raster used for real collision boxes, sensing and sparse foothold penalty; exact cell query'
    result['square_course']=dict(version=1,physical=True,initial_level=cfg.beamdojo_initial_level,
        native_terrain_columns=cfg.scene.terrain.terrain_generator.num_cols,
        level_changes='none in first 1000-update block; evaluate before next block',
        geometry=cfg.beamdojo_square_course_geometry[cfg.beamdojo_initial_level],
        goal=cfg.beamdojo_square_course_params,
        completion='parked, zero command, both complete foot envelopes with 1cm margin on +X ring, double contact, stable 1s; failure/time limit takes precedence',
        velocity_body_m_s=[.5,1.],velocity_lateral_m_s=0.,yaw_rate_rad_s=0.,
        initialization='source actor and normalization only; fresh dual critics and optimizer; no AMP',
        motion_rewards='complete official Unitree motion terms restored; no corridor progress/heading/lateral reward additions',
        perception='ideal simulated heightmap; not validated physical LiDAR or camera')
    goal=result['square_course']['completion']
    result['terrain_family']='square_start_single_straight_branch_solid_goal_ring'
    result['physical_alignment']=dict(version=3,map_size_m=[8.,8.],
        physical_initial_levels=[cfg.beamdojo_initial_level],spawn='central platform, XY=0, yaw=0',
        partial_sole_support_allowed=False,strict_foothold=True,
        reward_only_unsupported_deadband=0.,heightmap_surface='exact 2cm collision raster cells',
        success=goal,qualification_timing='terminal success after parked stable hold',
        reset_on_distance_milestone=False,continuous_passage=False,no_absolute_pose_in_actor=True)
    result['course_integrity']=dict(version=2,map_size_m=[8.,8.],
        start_platform_m=cfg.beamdojo_square_course_params['start_platform_m'],
        goal_platform_m=cfg.beamdojo_square_course_params['end_platform_m'],
        boundary_abs_xy_m=3.9,route_half_width_after_central_m=.42,
        boundary_truncation=False,finish_hold_s=1.,finish_linear_speed_max_m_s=.15,
        finish_angular_speed_max_rad_s=.3,finish_body_tilt_max_rad=.35,
        success=goal,time_limit_s=cfg.episode_length_s,
        timeout='truncate only when no simultaneous physical/route failure; timeout never qualifies success')
    result['curriculum']=dict(enabled=False,terrain_owner='fixed square course first block',
        initial_training_levels=[cfg.beamdojo_initial_level],max_level=10,
        consecutive_passes_to_promote=None,demote_on_failure=False)
    result['course_schedule']=dict(name='square_straight_v1',level_count=11,max_level=10,
        internal_mesh_level_enumerator='smooth_top_v1',stone_side_m=[.28]*11,
        actual_same_column_gap_m=[c['metadata']['actual_same_column_gap_m'] for c in cfg.beamdojo_square_course_geometry])
    result['course_tracking'].update(command_sampling_override='all forward; zero lateral and yaw; goal parking overrides forward speed',
        motion_reward_shaping='official movement rewards; no added heading/lateral/progress terms')
    result['commands'].update(initial_ranges={'lin_vel_x':[.5,1.],'lin_vel_y':[0.,0.],'ang_vel_z':[0.,0.]},
        limit_ranges={'lin_vel_x':[.5,1.],'lin_vel_y':[0.,0.],'ang_vel_z':[0.,0.]},
        resampling_s=12.,heading_command=False,standing_fraction=0.,
        curriculum='disabled; command generator brakes and latches zero speed on corresponding goal ring')
    result['virtual_handoff']=dict(version=2,query='exact collision raster cells',
        motion_rewards='official 19 plus inherited sparse foothold',recovery_scope='disabled')
    result['recovery']=dict(enabled=False,pass_rule=goal)
    return result
