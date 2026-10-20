"""V3 two-contact constant-height centroidal diagnostic, A -> B -> A-prime.

Seven fixed-load phases share common H=.6. The same legal weighted per-foot
CoPs and contact forces advance COM, velocity, and total xyz angular momentum
about current COM exactly within each phase; neither touchdown resets L.
A-prime has no force until the second touchdown. Both candidate point sets
are hypothetical terrain-supported patches, not measured future contacts.
Two Euclidean hull projections are a feasible weighted-CoP heuristic, not an
optimal weighted-Minkowski projection or a whole-body feasibility certificate.
The model omits impacts, variable height, leg IK/collision and actuator limits.
Raw vertical velocity is separately reported and gated; modeled vz is zero.
Objective units/scales and friction .5 remain the frozen one-contact model's.
"""
from __future__ import annotations
import math
import torch
from .contact_transition_planner import project_support_hull, _cross2


@torch.no_grad()
def plan_two_contact_transition(
    support_points, support_mask, com, com_velocity, angular_momentum, mass,
    command_xy, new_support_foot, candidate_points, candidate_mask,
    touchdown_times, foot_positions, second_candidate_points, second_candidate_mask, second_touchdown_times, *,
    candidate_centers=None, second_candidate_centers=None, second_contact_enabled=True,
    planning_horizon=.6, double_support_duration=.08,
    min_intermediate_support_duration=.04, min_final_support_duration=.04, friction=.5, max_swing_speed=2.5,
    max_step_distance=.7, height_range=(.25,1.2), plane_tolerance=.025, max_vertical_speed=.2,
    length_scale=.5, momentum_reference_duration=.3,
    angular_momentum_band_ratio=(.04,.04,.08), max_horizontal_moment_ratio=.15,
    gravity=9.81, chunk_size=32,
):
    """Plan pre-enumerated A -> B -> A-prime combinations on a common horizon.

    The first 12 arguments match v1. Additional second points/mask/time have
    shapes [N,C,P,3], [N,C,P], [N,C]. Both times are from the SAME decision.
    Centers optionally specify foot-link origins [N,C,3]. C is the complete
    joint two-step combination axis, not two independent candidate axes.
    Output foot arrays always order [old A (later A-prime), new B], regardless
    of the robot's left/right index; first/second_support_foot identify them.
    Invalid durations are explicitly rejected and receive NO rollout: their
    arithmetic steps are zero, force/path outputs masked, cost=1, best=-1 if
    no other candidate passes. The raw requested phase_durations are retained.
    Finite but physically constrained trajectories remain diagnostic evidence.
    second_contact_enabled=False is the paired seven-phase hold-B control:
    phase times and repeated CoP selection stay identical, but A-prime never
    applies force and its geometry/second-swing reach gates are not enforced.
    Candidate finite-state and both timing inputs are still checked.
    """
    if type(second_contact_enabled) is not bool:raise ValueError('second_contact_enabled must be bool')
    for name,value in [('planning_horizon',planning_horizon),('double_support_duration',double_support_duration),
                       ('min_intermediate_support_duration',min_intermediate_support_duration),
                       ('min_final_support_duration',min_final_support_duration),('friction',friction),
                       ('max_swing_speed',max_swing_speed),('max_step_distance',max_step_distance),('max_vertical_speed',max_vertical_speed),
                       ('length_scale',length_scale),('momentum_reference_duration',momentum_reference_duration),
                       ('max_horizontal_moment_ratio',max_horizontal_moment_ratio),('gravity',gravity)]:
        if not math.isfinite(value) or value<=0: raise ValueError(name+' must be finite and positive')
    if len(height_range)!=2 or not 0<height_range[0]<height_range[1]: raise ValueError('invalid height_range')
    if not math.isfinite(plane_tolerance) or plane_tolerance<0: raise ValueError('invalid plane_tolerance')
    if len(angular_momentum_band_ratio)!=3 or any(not math.isfinite(x) or x<=0 for x in angular_momentum_band_ratio): raise ValueError('invalid momentum band')
    if not isinstance(chunk_size,int) or chunk_size<1: raise ValueError('invalid chunk_size')
    if support_points.ndim!=4 or support_points.shape[1]!=2 or support_points.shape[-1]!=3: raise ValueError('actual points shape [N,2,P,3] required')
    n,_,p,_=support_points.shape
    if candidate_points.ndim!=4 or candidate_points.shape[0]!=n or candidate_points.shape[2:]!=(p,3): raise ValueError('candidate points shape [N,C,P,3] required')
    ccount=candidate_points.shape[1]
    if second_candidate_points.shape!=candidate_points.shape:raise ValueError('second candidate points shape mismatch')
    if p<1 or ccount<1: raise ValueError('nonempty candidate/sample dimensions required')
    expected=[(support_mask,(n,2,p)),(com,(n,3)),(com_velocity,(n,3)),(angular_momentum,(n,3)),
              (command_xy,(n,2)),(new_support_foot,(n,)),(candidate_mask,(n,ccount,p)),
              (touchdown_times,(n,ccount)),(foot_positions,(n,2,3)),
              (second_candidate_mask,(n,ccount,p)),(second_touchdown_times,(n,ccount))]
    if any(x.shape!=shape for x,shape in expected): raise ValueError('state/mask/time tensor shape mismatch')
    if candidate_centers is not None and candidate_centers.shape!=(n,ccount,3): raise ValueError('candidate_centers shape mismatch')
    if second_candidate_centers is not None and second_candidate_centers.shape!=(n,ccount,3):raise ValueError('second centers shape mismatch')
    device=support_points.device
    values=[x.to(device=device,dtype=torch.float32) for x in (support_points,com,com_velocity,angular_momentum,command_xy,candidate_points,touchdown_times,foot_positions)]
    actual,c0,v0,L0,cmd,candidates,tau,feet=values
    actual_mask=support_mask.to(device=device,dtype=torch.bool)
    new_mask=candidate_mask.to(device=device,dtype=torch.bool)
    second_points=second_candidate_points.to(device=device,dtype=torch.float32)
    second_mask=second_candidate_mask.to(device=device,dtype=torch.bool)
    tau2=second_touchdown_times.to(device=device,dtype=torch.float32)
    second_centers=second_points.mean(-2) if second_candidate_centers is None else second_candidate_centers.to(device=device,dtype=torch.float32)
    mass=torch.as_tensor(mass,device=device,dtype=torch.float32).expand(n)
    foot=new_support_foot.to(device=device)
    centers=candidates.mean(-2) if candidate_centers is None else candidate_centers.to(device=device,dtype=torch.float32)
    valid=torch.isfinite(c0).all(-1)&torch.isfinite(v0).all(-1)&torch.isfinite(L0).all(-1)&torch.isfinite(cmd).all(-1)
    valid &= torch.isfinite(mass)&(mass>0)&(mass<=1e6)&((foot==0)|(foot==1))&torch.isfinite(feet).all((-1,-2))
    valid &= (torch.isfinite(actual).all(-1)|~actual_mask).all((-1,-2))
    finite=valid[:,None]&torch.isfinite(tau)&torch.isfinite(centers).all(-1)&(torch.isfinite(candidates).all(-1)|~new_mask).all(-1)
    finite &= torch.isfinite(tau2)&torch.isfinite(second_centers).all(-1)&(torch.isfinite(second_points).all(-1)|~second_mask).all(-1)
    clean=lambda x:torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
    actual,c0,v0,L0,cmd,candidates,tau,feet,centers=map(clean,(actual,c0,v0,L0,cmd,candidates,tau,feet,centers))
    second_points,tau2,second_centers=map(clean,(second_points,tau2,second_centers))
    mass=clean(mass).clamp(1e-6,1e6); mg=mass*gravity
    batch=torch.arange(n,device=device); which=foot.long().clamp(0,1)
    old_points=actual[batch,1-which]
    old_mask=actual_mask[batch,1-which]
    ground=(old_points[...,2]*old_mask).sum(-1)/old_mask.sum(-1).clamp_min(1)
    height=c0[:,2]-ground
    height_valid=(height>=height_range[0])&(height<=height_range[1])
    vertical_velocity_valid=v0[:,2].abs()<=max_vertical_speed
    h=height.clamp(*height_range)
    # Common-plane approximation is explicitly bounded, not silently assumed.
    old_plane=torch.where(old_mask,(old_points[...,2]-ground[:,None]).abs(),0.).amax(-1)
    new_plane=torch.where(new_mask,(candidates[...,2]-ground[:,None,None]).abs(),0.).amax(-1)
    second_plane=torch.where(second_mask,(second_points[...,2]-ground[:,None,None]).abs(),0.).amax(-1)
    second_geometry_valid=second_mask.any(-1)&(second_plane<=plane_tolerance)
    geometry_valid=old_mask.any(-1)[:,None]&new_mask.any(-1)&(old_plane[:,None]<=plane_tolerance)&(new_plane<=plane_tolerance)&height_valid[:,None]&vertical_velocity_valid[:,None]
    if second_contact_enabled:geometry_valid &= second_geometry_valid
    half=torch.full_like(tau,double_support_duration/2)
    intermediate=tau2-tau-double_support_duration
    final=planning_horizon-tau2-double_support_duration
    durations=torch.stack((tau,half,half,intermediate,half,half,final),-1)
    timing_valid=(tau>0)&(intermediate>=min_intermediate_support_duration-1e-6)&(final>=min_final_support_duration-1e-6)&(durations>0).all(-1)
    swing_distance=torch.linalg.vector_norm(centers-feet[batch,which][:,None],dim=-1)
    second_swing_distance=torch.linalg.vector_norm(second_centers-feet[batch,1-which][:,None],dim=-1)
    step_distance=torch.linalg.vector_norm(centers[...,:2]-feet[batch,1-which][:,None,:2],dim=-1)
    second_step_distance=torch.linalg.vector_norm(second_centers[...,:2]-centers[...,:2],dim=-1)
    required_swing_speed=swing_distance/tau.clamp_min(1e-4)
    second_required_swing_speed=second_swing_distance/intermediate.clamp_min(1e-4)
    first_kinematic_valid=(required_swing_speed<=max_swing_speed)&(step_distance<=max_step_distance)
    second_kinematic_valid=(second_required_swing_speed<=max_swing_speed)&(second_step_distance<=max_step_distance)
    kinematic_valid=first_kinematic_valid&(second_kinematic_valid if second_contact_enabled else torch.ones_like(second_kinematic_valid))
    finite &= (c0.abs().amax(-1)<1e6)[:,None]&(v0.abs().amax(-1)<1e3)[:,None]&(L0.abs().amax(-1)<1e6)[:,None]
    b=n*ccount
    old_xy=old_points[:,:,:2][:,None].expand(-1,ccount,-1,-1).reshape(b,p,2)
    old_valid=old_mask[:,None].expand(-1,ccount,-1).reshape(b,p)
    new_xy=candidates[...,:2].reshape(b,p,2); new_valid=new_mask.reshape(b,p)
    second_xy=second_points[...,:2].reshape(b,p,2); second_valid=second_mask.reshape(b,p)
    xy=c0[:,:2][:,None].expand(-1,ccount,-1).clone()
    velocity=v0[:,:2][:,None].expand(-1,ccount,-1).clone()
    momentum=L0[:,None].expand(-1,ccount,-1).clone()
    omega=(gravity/h).sqrt()[:,None]
    rollout_performed=timing_valid&finite
    safe_dt=torch.where(rollout_performed[...,None],durations,0.)
    shares=torch.tensor([0.,.25,.75,1.,.75,.25,0.] if second_contact_enabled else [0.,.25,.75,1.,1.,1.,1.],device=device)
    band=torch.tensor(angular_momentum_band_ratio,device=device)*mg[:,None]*length_scale*momentum_reference_duration
    com_path=[c0[:,None].expand(-1,ccount,-1).clone()]
    velocity_path=[torch.cat((velocity,torch.zeros_like(velocity[...,:1])),-1)]
    momentum_path=[momentum.clone()]
    cops=[];cmps=[];foot_cops=[];forces0=[];forces1=[];frictions=[];tracking=[];rates=[];moment_rates0=[];moment_rates1=[]
    elapsed=torch.zeros_like(tau)
    for stage in range(7):
        dt=safe_dt[...,stage]
        radius=torch.linalg.vector_norm(momentum/band[:,None],dim=-1)
        excess=momentum*(1.-1./radius.clamp_min(1.))[...,None]
        requested_rate=-excess[...,:2]/(planning_horizon-elapsed).clamp_min(1e-4)[...,None]
        maximum=max_horizontal_moment_ratio*mg*length_scale
        requested_rate *= (maximum[:,None]/torch.linalg.vector_norm(requested_rate,dim=-1).clamp_min(1e-12)).clamp_max(1.)[...,None]
        offset=torch.stack((requested_rate[...,1],-requested_rate[...,0]),-1)/mg[:,None,None]
        ch=torch.cosh(omega*dt);sh=torch.sinh(omega*dt)
        required_cmp=xy+(velocity*ch[...,None]-cmd[:,None])/(omega*sh).clamp_min(1e-8)[...,None]
        required_cop=required_cmp-offset
        a_points,a_mask=(second_xy,second_valid) if stage>=4 and second_contact_enabled else (old_xy,old_valid)
        old_cop,_=project_support_hull(a_points,a_mask,required_cop.reshape(b,2),chunk_size=chunk_size)
        new_cop,_=project_support_hull(new_xy,new_valid,required_cop.reshape(b,2),chunk_size=chunk_size)
        old_cop=old_cop.reshape(n,ccount,2);new_cop=new_cop.reshape(n,ccount,2)
        share=shares[stage]
        cop=(1-share)*old_cop+share*new_cop
        cmp=cop+offset
        displacement=xy-cmp
        next_xy=cmp+displacement*ch[...,None]+velocity*(sh/omega)[...,None]
        next_velocity=displacement*(omega*sh).clamp_min(1e-8)[...,None]+velocity*ch[...,None]
        # Same contact forces determine centroidal angular-momentum evolution.
        F0=mass[:,None,None]*omega[...,None].square()*displacement
        F1=mass[:,None,None]*omega[...,None].square()*(next_xy-cmp)
        force0=torch.cat((F0,mg[:,None,None].expand(-1,ccount,1)),-1)
        force1=torch.cat((F1,mg[:,None,None].expand(-1,ccount,1)),-1)
        weights=torch.stack((1-share,share))
        forces0.append(force0[:,:,None]*weights[None,None,:,None])
        forces1.append(force1[:,:,None]*weights[None,None,:,None])
        phase_foot_cop=torch.stack((old_cop,new_cop),-2)
        phase_foot_cop=torch.cat((phase_foot_cop,ground[:,None,None,None].expand(-1,ccount,2,1)),-1)
        foot_cops.append(phase_foot_cop)
        center0=torch.cat((xy,c0[:,2,None,None].expand(-1,ccount,1)),-1)
        center1=torch.cat((next_xy,c0[:,2,None,None].expand(-1,ccount,1)),-1)
        moment_rates0.append(torch.cross(phase_foot_cop-center0[:,:,None],forces0[-1],dim=-1).sum(-2))
        moment_rates1.append(torch.cross(phase_foot_cop-center1[:,:,None],forces1[-1],dim=-1).sum(-2))
        actual_rate_xy=mg[:,None,None]*torch.stack((cop[...,1]-cmp[...,1],cmp[...,0]-cop[...,0]),-1)
        integrated_displacement=displacement*(sh/omega)[...,None]+velocity*((ch-1)/omega.square())[...,None]
        delta_yaw=-mass[:,None]*omega.square()*_cross2(offset,integrated_displacement)
        momentum=momentum+torch.cat((actual_rate_xy*dt[...,None],delta_yaw[...,None]),-1)
        frictions.append(torch.maximum(torch.linalg.vector_norm(F0,dim=-1),torch.linalg.vector_norm(F1,dim=-1))/mg[:,None])
        tracking.append(.5*((velocity-cmd[:,None]).square().sum(-1)+(next_velocity-cmd[:,None]).square().sum(-1))*dt/planning_horizon)
        rates.append(actual_rate_xy)
        xy=next_xy;velocity=next_velocity;elapsed+=dt
        com_path.append(torch.cat((xy,c0[:,2,None,None].expand(-1,ccount,1)),-1))
        velocity_path.append(torch.cat((velocity,torch.zeros_like(velocity[...,:1])),-1))
        momentum_path.append(momentum.clone());cops.append(cop);cmps.append(cmp)
    friction_ratio=torch.stack(frictions,-1)
    friction_valid=(friction_ratio<=friction+1e-6).all(-1)
    velocity_error=torch.linalg.vector_norm(velocity-cmd[:,None],dim=-1)
    progress_target=(c0[:,:2][:,None]+cmd[:,None]*planning_horizon).expand(-1,ccount,-1)
    position_error=torch.linalg.vector_norm(xy-progress_target,dim=-1)
    final_radius=torch.linalg.vector_norm(momentum/band[:,None],dim=-1)
    objective=torch.stack(tracking,-1).sum(-1)/(.5**2)+(velocity_error/.5).square()+(position_error/.5).square()+final_radius.sub(1).clamp_min(0).square()
    objective += 10*(friction_ratio-friction).clamp_min(0).square().sum(-1)
    finite &= torch.isfinite(objective)&torch.isfinite(momentum).all(-1)&torch.isfinite(xy).all(-1)
    feasible=finite&geometry_valid&timing_valid&kinematic_valid&friction_valid
    cost=torch.where(finite,-torch.expm1(-objective.clamp_min(0)),1.).clamp(0,1)
    cost=torch.where(geometry_valid&timing_valid&kinematic_valid,cost,1.)
    ranked=torch.where(feasible,objective,float('inf'))
    best=ranked.argmin(-1)
    best=torch.where(feasible.any(-1),best,-1)
    def trajectory(value):
        value=clean(value)
        gate=rollout_performed.reshape(n,ccount,*([1]*(value.ndim-2)))
        return torch.where(gate,value,0.)
    return dict(candidate_objective=clean(objective),candidate_cost=cost,feasible=feasible,best_index=best,
                second_contact_enabled=second_contact_enabled,second_geometry_valid=second_geometry_valid,second_kinematic_checked=second_contact_enabled,
                candidate_touchdown_times=tau,second_candidate_touchdown_times=tau2,
                first_touchdown_deadline=tau,second_touchdown_deadline=tau2,rollout_performed=rollout_performed,
                first_support_foot=which,second_support_foot=1-which,total_horizon=torch.full_like(tau,planning_horizon),phase_durations=durations,
                cop=trajectory(torch.stack(cops,2)),cmp=trajectory(torch.stack(cmps,2)),foot_cop=trajectory(torch.stack(foot_cops,2)),
                com_path=trajectory(torch.stack(com_path,2)),velocity_path=trajectory(torch.stack(velocity_path,2)),momentum_path=trajectory(torch.stack(momentum_path,2)),
                foot_force_start=trajectory(torch.stack(forces0,2)),foot_force_end=trajectory(torch.stack(forces1,2)),
                new_foot_normal_load_fraction=shares[None,None].expand(n,ccount,-1),
                foot_normal_load_fractions=torch.stack((1-shares,shares),-1)[None,None].expand(n,ccount,-1,-1),
                friction_ratio=clean(friction_ratio),geometry_valid=geometry_valid,timing_valid=timing_valid,
                kinematic_valid=kinematic_valid,first_kinematic_valid=first_kinematic_valid,second_kinematic_valid=second_kinematic_valid,friction_valid=friction_valid,finite=finite,
                old_valid_points=old_mask.sum(-1),candidate_valid_points=new_mask.sum(-1),second_candidate_valid_points=second_mask.sum(-1),
                height=height,old_plane_deviation=old_plane,candidate_plane_deviation=new_plane,second_candidate_plane_deviation=second_plane,
                required_swing_speed=clean(required_swing_speed),step_distance=clean(step_distance),
                second_required_swing_speed=clean(second_required_swing_speed),second_step_distance=clean(second_step_distance),second_available_swing_time=intermediate,
                terminal_velocity_error=clean(velocity_error),terminal_position_error=clean(position_error),
                terminal_momentum_band_radius=clean(final_radius),progress_target=progress_target,
                actual_horizontal_momentum_rate=trajectory(torch.stack(rates,2)),
                stage_cop=trajectory(torch.stack(cops,2)),stage_cmp=trajectory(torch.stack(cmps,2)),
                stage_new_load=shares[None,None].expand(n,ccount,-1),
                phase_times=torch.cat((torch.zeros_like(tau)[...,None],durations.cumsum(-1)),-1),
                applied_Ldot_start=trajectory(torch.stack(moment_rates0,2)),applied_Ldot_end=trajectory(torch.stack(moment_rates1,2)),
                input_vertical_velocity=v0[:,2],vertical_velocity_valid=vertical_velocity_valid)
