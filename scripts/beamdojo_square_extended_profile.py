"""Extended v2 mixed courses; official motion rewards/control are preserved."""
import sys
import numpy as np
from beamdojo_square_extended_geometry import tile, mesh, terrain_mesh, geometry_parameters


def column_specs(seed,phase):
    specs=[]
    for col in range(64):
        family=col//32;variant=(col%32)//4 if phase=='generalize' else 0
        layout_seed=seed+(col%4)*1009 if variant else seed
        specs.append((layout_seed,family,variant))
    return specs


def build(parent,num_envs,seed,device,phase='regular',initial_level=0,flat_fraction=0.):
    from beamdojo_square_course_profile import build as base_build
    from isaaclab.managers import TerminationTermCfg
    from isaaclab.utils.configclass import configclass
    import isaaclab.terrains as terrain_gen
    from legged_lab.tasks.locomotion.beamdojo import upstream_terrain as upstream
    import beamdojo_square_mixed_runtime as runtime
    if phase not in ('regular','generalize') or not 0<=initial_level<=10:
        raise ValueError('Invalid phase or initial terrain level')
    cfg,agent=base_build(parent,num_envs,seed,device,initial_level)
    from beamdojo_square_extended_runtime import install
    install(runtime)
    cfg.beamdojo_square_extended_params=dict(inner_half=4., outer_half=6., diagonal_corner_infill=.5, terminal_runout=1., stand_seconds=1., variant_max=7)
    cfg.beamdojo_square_course_params.update(end_platform_m=[4.,-4.,6.,4.],target_root_x=5.2,
        braking_start_x=4.2,boundary_limit=5.9)
    cfg.episode_length_s=20.
    cfg.beamdojo_stage1_map_size=12.
    cfg.commands.base_velocity.resampling_time_range=(20.,20.)
    cfg.square_mixed_phase=phase;cfg.square_mixed_active_variant=0 if phase=='regular' else 1
    if not 0<=flat_fraction<1:raise ValueError('Invalid flat rehearsal fraction')
    cfg.flat_rehearsal_fraction=float(flat_fraction)
    cfg.flat_rehearsal_count=int(num_envs*flat_fraction)
    cfg.square_mixed_column_specs=column_specs(seed,phase)+([(seed,2,0)]*16 if flat_fraction else [])
    cfg.curriculum=None;cfg.beamdojo_curriculum=False
    cfg.events.reset_base.func=runtime.reset_root_state
    cfg.commands.base_velocity.class_type=runtime.make_command_class()
    cfg.terminations.course_finished=TerminationTermCfg(func=runtime.course_success,time_out=False)
    cfg.terminations.course_boundary=TerminationTermCfg(func=runtime.course_boundary,time_out=False)
    cfg.terminations.time_out=TerminationTermCfg(func=runtime.time_out,time_out=True)
    @configclass
    class MixedTileCfg(terrain_gen.MeshPlaneTerrainCfg):
        function=terrain_mesh
        layout_seed:int=42
        mixed_family:int=0
        mixed_variant:int=0
    generator=cfg.scene.terrain.terrain_generator
    generator.size=(12.,12.)
    generator.num_cols=len(cfg.square_mixed_column_specs);generator.num_rows=11;generator.use_cache=False
    generator.sub_terrains={f'col_{col:02d}':MixedTileCfg(proportion=1/generator.num_cols,layout_seed=s,mixed_family=f,mixed_variant=v)
                            for col,(s,f,v) in enumerate(cfg.square_mixed_column_specs)}
    courses=[[tile(s,f,v,l)[0] for l in range(11)] for s,f,v in cfg.square_mixed_column_specs]
    masks=np.stack([[tile(s,f,v,l)[1] for l in range(11)] for s,f,v in cfg.square_mixed_column_specs])
    width=max(len(c['rectangles_m']) for group in courses for c in group)
    import torch
    boxes=torch.zeros(generator.num_cols,11,width,4);display=torch.zeros(generator.num_cols,11,width,5)
    valid=torch.zeros(generator.num_cols,11,width,dtype=torch.bool);stones=valid.clone()
    for col,group in enumerate(courses):
        for l,c in enumerate(group):
            n=len(c['rectangles_m']);r=torch.tensor(c['rectangles_m'])
            boxes[col,l,:n]=r;display[col,l,:n,:2]=(r[:,:2]+r[:,2:])/2
            display[col,l,:n,2:4]=r[:,2:]-r[:,:2];valid[col,l,:n]=True
            for route in c['routes']:stones[col,l,route['stone_polygon_indices']]=True
    def choose_variants(env,ids):
        family=env.square_family[ids]
        strength=env.cfg.square_mixed_active_variant
        if phase=='regular':variants=torch.zeros_like(family)
        else:
            # 25% regular replay; remaining exposure grows from variants1 to7.
            replay=torch.rand(len(ids),device=env.device)<.25
            variants=torch.randint(1,strength+1,(len(ids),),device=env.device)
            variants[replay]=0
        env.square_variant[ids]=variants
        replica=torch.randint(0,32 if phase=='regular' else 4,(len(ids),),device=env.device)
        cols=family*32+variants*4+replica
        if flat_fraction:
            flat=env.flat_rehearsal_mask[ids];cols=torch.where(flat,64+ids%16,cols);env.square_variant[ids]=torch.where(flat,0,env.square_variant[ids]);env.beamdojo_levels[ids[flat]]=0
        runtime.tensor(env.scene.terrain.terrain_types)[ids]=cols
    def ensure(env):
        if hasattr(env,'_square_mixed_grids'):return
        n=env.num_envs;dev=env.device;index=torch.arange(n,device=dev)
        env._square_mixed_grids=torch.as_tensor(masks,device=dev,dtype=torch.bool)
        env._upstream_heightfields=env._square_mixed_grids[0].float()-1
        env._square_mixed_boxes=boxes.to(dev);env._square_mixed_display=display.to(dev)
        env._square_mixed_mask=valid.to(dev);env._square_mixed_stones=stones.to(dev)
        env.beamdojo_levels=torch.full((n,),initial_level,device=dev,dtype=torch.long)
        env.beamdojo_success_streak=torch.zeros_like(env.beamdojo_levels)
        env.square_family=(index%8)//4;env.square_route=index%4
        env.square_variant=torch.zeros_like(index)
        env.flat_rehearsal_mask=index<cfg.flat_rehearsal_count
        env.flat_rehearsal_mode=index%8
        env.square_refresh_geometry=refresh
        env.square_choose_variants=lambda ids:choose_variants(env,ids)
        env._square_course_surface_height=lambda points:query(env,points)
        runtime.ensure_state(env)
        choose_variants(env,index);refresh(env,index)
    def refresh(env,ids=None):
        ensure(env)
        ids=torch.arange(env.num_envs,device=env.device) if ids is None else ids
        terrain=env.scene.terrain
        levels=env.beamdojo_levels[ids];cols=runtime.tensor(terrain.terrain_types)[ids]
        runtime.tensor(terrain.terrain_levels)[ids]=levels
        origins=runtime.tensor(terrain.terrain_origins)[levels,cols]
        runtime.tensor(terrain.env_origins)[ids]=origins;runtime.tensor(env.scene.env_origins)[ids]=origins
        # Native observation/query caches are shared; reset publication cache.
        env._upstream_heightmap_cache=None;env._upstream_critic_frame_cache=None
        all_cols=runtime.tensor(terrain.terrain_types);all_lv=env.beamdojo_levels
        env.beamdojo_geometry=env._square_mixed_display[all_cols,all_lv]
        env.beamdojo_rectangle_mask=env._square_mixed_mask[all_cols,all_lv]
        env.beamdojo_stone_mask=env._square_mixed_stones[all_cols,all_lv]
        env.beamdojo_start_platform=env.beamdojo_geometry[:,0]
        env.beamdojo_end_platform=env.beamdojo_geometry[:,1]
        env.beamdojo_square_course_rectangles=env._square_mixed_boxes[all_cols,all_lv]
    def query(env,points,*,exact=None):
        ensure(env)
        origins=runtime.tensor(env.scene.env_origins)
        local=points-origins.reshape((env.num_envs,)+(1,)*(points.ndim-2)+(3,))
        xy=local[...,:2].reshape(env.num_envs,-1,2)
        inside=torch.isfinite(xy).all(-1)&(xy>=-6).all(-1)&(xy<6).all(-1)
        px=((torch.nan_to_num(xy)+6)/.02).long().clamp(0,599)
        cols=runtime.tensor(env.scene.terrain.terrain_types)
        support=env._square_mixed_grids[cols[:,None],env.beamdojo_levels[:,None],px[...,0],px[...,1]] & inside
        height=support.to(points.dtype)-1+origins[:,2:3]
        return height.reshape(points.shape[:-1]),support.reshape(points.shape[:-1])
    runtime.initialize_geometry=ensure
    replacements={upstream.ensure_state:ensure,upstream.refresh_geometry:refresh,
        upstream._refresh_display:refresh,upstream.surface_height:query,
        upstream._update_course:runtime.update_course,upstream.update_success:runtime.update_course,
        upstream.before_reset:runtime.before_reset,upstream.course_diagnostics:runtime.course_metrics}
    for module in list(sys.modules.values()):
        if module is None or '.beamdojo' not in getattr(module,'__name__',''):continue
        for key,value in list(vars(module).items()):
            try:replacement=replacements.get(value)
            except TypeError:continue
            if replacement is not None:setattr(module,key,replacement)
    cfg.square_mixed_geometry=dict(families=['cardinal','diagonal'],directions=8,columns=generator.num_cols,
        replicas_per_family_variant=4,phase=phase,seed=seed,
        specs=[list(spec) for spec in cfg.square_mixed_column_specs],
        generalization_profiles=geometry_parameters()['variants'],
        geometry_revision=2,heights_m=0.,replay_fraction=.25 if phase=='generalize' else 1.)
    if flat_fraction:
        cfg.flat_rehearsal_ranges={'lin_vel_x':[-.5,1.],'lin_vel_y':[-.3,.3],'ang_vel_z':[-.2,.2]}
    return cfg,agent


def make_contract(base,cfg,parent):
    from beamdojo_square_course_profile import make_contract as base_contract
    from beamdojo_square_extended_runtime import REVISION
    import hashlib
    result=base_contract(base,cfg,parent)
    result['terrain_family']='square_extended_cardinal_diagonal_mixed_route_facing'
    result['square_mixed']=dict(version=2,**cfg.square_mixed_geometry,
        curriculum={'difficulty_range':[0,10],'promote_consecutive_arrivals':3,
                    'demote_consecutive_nonarrivals':2,'speed_error_is_promotion_gate':False,
                    'standing_is_promotion_gate':False,'pass_rule':'first complete-foot arrival on selected outer ring before physical failure'},
        spawn='centered; body yaw equals assigned route yaw; pure body-forward command')
    result['square_course']['initialization']='full actor/normalization/dualcritic/optimizer resume'
    result['square_course']['level_changes']='3 consecutive outer-ring arrivals promote; 2 nonarrival episodes demote; parking quality is logged separately'
    result['square_course']['geometry']=dict(**geometry_parameters(),four_branches=True)
    result['square_course']['terminal']=dict(revision=REVISION,runout_m=1.,entry_clearance_m=.2,hold_s=1.,
        trigger='complete-foot outer-ring entry and root beyond entry+0.2m',speed='finite distance taper to zero; stop latched until reset')
    result['square_course']['completion']='selected-route outer ring, latched zero command, complete soles and double contact, stable1s; failure/timeout precedence'
    result['physical_alignment']['map_size_m']=[12.,12.]
    result['course_integrity'].update(map_size_m=[12.,12.],boundary_abs_xy_m=5.9,time_limit_s=cfg.episode_length_s,
        goal_platform_m=[4.,-4.,6.,4.])
    result['physical_alignment']['success']=result['square_course']['completion']
    result['course_integrity']['success']=result['square_course']['completion']
    result['course_schedule'].update(name='square_extended_v2',
        actual_same_column_gap_m=[tile(cfg.seed,0,0,l)[0]['metadata']['actual_same_column_gap_m'] for l in range(11)],
        stone_counts_per_route={name:[tile(cfg.seed,family,0,l)[0]['metadata']['stone_count_per_route'] for l in range(11)]
                                for family,name in enumerate(('cardinal','diagonal'))})
    result['square_course']['episode_length_s']=cfg.episode_length_s
    result['commands']['resampling_s']=cfg.episode_length_s
    result['square_course']['goal']='selected route outer ring/full sole support + parked double contact stable1s'
    result['physical_alignment']['spawn']='centered with per-env route-facing yaw'
    result['curriculum'].update(enabled=True,terrain_owner='mixed_before_reset',demote_on_failure=True,
        consecutive_passes_to_promote=3,consecutive_failures_to_demote=2,
        pass_rule='complete-foot outer-ring arrival; standstill not required',standing_is_promotion_gate=False)
    digest=hashlib.sha256()
    for s,f,v in cfg.square_mixed_column_specs:
        for l in range(11):digest.update(tile(s,f,v,l)[1].tobytes())
    result['square_mixed']['support_raster_sha256']=digest.hexdigest()
    if cfg.flat_rehearsal_fraction:
        result['flat_rehearsal']=dict(version=1,fraction=cfg.flat_rehearsal_fraction,count=cfg.flat_rehearsal_count,stone_count=cfg.scene.num_envs-cfg.flat_rehearsal_count,columns=list(range(64,80)),plane_size_m=[12.,12.],
            command_ranges=cfg.flat_rehearsal_ranges,modes=['forward','backward','left','right','turn_left','turn_right','stand','mixed'],
            resampling_s=[4.,8.],motion_rewards='official Unitree weights/functions/parameters unchanged',sparse_foothold='zero on supported flat cells',
            termination='physical failure; timeout20s or tile boundary5.5m is neutral truncation; no goal-ring task',curriculum='flat stays level0; stone promotion/demotion unchanged')
    return result
