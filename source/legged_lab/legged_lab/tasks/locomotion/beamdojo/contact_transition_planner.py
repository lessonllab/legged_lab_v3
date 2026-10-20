"""Deterministic reduced-order landing position/time candidate planner.

Every candidate has the SAME physical endpoint H. Four phases last
[touchdown, DS/2, DS/2, H-touchdown-DS], with new-foot normal-load shares
[0,.25,.75,1]. Each phase chooses a CoP in the weighted Minkowski sum
(1-share)*conv(actual old samples)+share*conv(hypothetical new samples).
Independent projections onto the two foot hulls give a feasible selection;
this is a heuristic, NOT the exact closest point of that Minkowski sum or
an optimal-control solution. Hypothetical terrain-supported samples are not
measured future contacts, nor are inter-sample CoPs independent gap forces.

The reduced state projects the supplied vertical velocity to zero (reported
separately); it does not model an impulse accomplishing that projection.
COM follows the exact constant-height LIPM within each phase. CoP and CMP
change between phases; position and velocity remain continuous. Normal force
is mg. Horizontal force is m*w^2*(c-CMP). A requested horizontal momentum
rate determines CMP-CoP, but reported TOTAL angular momentum ABOUT CURRENT
COM is integrated from that SAME force and CoP, including the induced yaw
moment. It is not angular momentum about the old/new support point. No
independent yaw-torque control or fictitious momentum reset is used.

Constant height/common contact plane, prescribed load shares, point-normal
forces, engineering momentum allowance and straight-line swing-speed/step
reach envelopes are approximations. Friction is checked at both phase ends;
||c-CMP|| is convex within a constant-CMP LIPM phase, so its maximum is at an
endpoint. Passing these checks is ONLY proxy validity, not whole-body, IK,
collision, actuator, impact, natural-gait, or long-term stability feasibility.
No actual future state is an input. Runtime must freeze touchdown deadlines
at decision time; repeatedly replanning a relative delay is not equivalent.
"""
from __future__ import annotations

import math
import torch


@torch.no_grad()
def project_support_hull(points, mask, query, *, chunk_size=32):
    """Project [B,2] onto hull([B,P,2]); empty hull returns zero, valid=False."""
    projections=[]
    for start in range(0,len(points),chunk_size):
        p=points[start:start+chunk_size].float()
        m=mask[start:start+chunk_size].bool()
        q=query[start:start+chunk_size].float()
        p=torch.where(m[...,None],p-q[:,None],0.)
        d=p[:,None,:,:]-p[:,:,None,:]
        pair=m[:,:,None]&m[:,None,:]
        length2=d.square().sum(-1)
        t=(-(p[:,:,None,:]*d).sum(-1)/length2.clamp_min(1e-20)).clamp(0,1)
        nearest=p[:,:,None,:]+t[...,None]*d
        dist2=torch.where(pair,nearest.square().sum(-1),float('inf'))
        chosen=dist2.flatten(1).argmin(-1)
        closest=nearest.flatten(1,2)[torch.arange(len(p),device=p.device),chosen]
        cross=d[:,:,:,None,0]*d[:,:,None,:,1]-d[:,:,:,None,1]*d[:,:,None,:,0]
        minimum=torch.where(m[:,None,None,:],cross,float('inf')).amin(-1)
        maximum=torch.where(m[:,None,None,:],cross,-float('inf')).amax(-1)
        tol=2e-7*torch.where(pair,length2,0.).amax((-1,-2)).clamp_min(1e-8)
        rank2=(pair&((maximum>tol[:,None,None])|(minimum < -tol[:,None,None]))).any((-1,-2))
        edge=pair&(length2>1e-20)&(minimum>=-tol[:,None,None])
        querycross=-d[...,0]*p[:,:,None,1]+d[...,1]*p[:,:,None,0]
        inside=((querycross>=-tol[:,None,None])|~edge).all((-1,-2))&rank2
        projected=torch.where(inside[:,None],q,closest+q)
        projections.append(torch.where(m.any(-1)[:,None],projected,0.))
    return torch.cat(projections) if projections else query.new_empty((0,2)),mask.any(-1)


def _cross2(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


@torch.no_grad()
def plan_contact_transition(
    support_points, support_mask, com, com_velocity, angular_momentum, mass,
    command_xy, new_support_foot, candidate_points, candidate_mask,
    touchdown_times, foot_positions, *, candidate_centers=None,
    planning_horizon=.6, double_support_duration=.08,
    min_final_support_duration=.1, friction=.5, max_swing_speed=2.5,
    max_step_distance=.7, height_range=(.25,1.2), plane_tolerance=.025, max_vertical_speed=.2,
    length_scale=.5, momentum_reference_duration=.3,
    angular_momentum_band_ratio=(.04,.04,.08), max_horizontal_moment_ratio=.15,
    gravity=9.81, chunk_size=32,
):
    """Inputs: actual points [N,2,P,3]/mask [N,2,P]; COM/v/total L [N,3];
    mass [N] or scalar; command [N,2]; new-foot index [N]; hypothetical points
    [N,C,P,3]/mask [N,C,P]; touchdown times [N,C]; current foot-link origins
    [N,2,3]. candidate_centers [N,C,3] optionally supplies target foot-link
    origins; otherwise the unmasked full-sole sample mean is used explicitly.

    Outputs include candidate_objective/candidate_cost/feasible [N,C],
    best_index [N] (-1 means NO passing candidate), cop/cmp [N,C,4,2],
    foot_cop [N,C,4,2,3], phase_durations [N,C,4], COM/v/L paths [N,C,5,3],
    foot forces at phase starts/ends [N,C,4,2,3], and separate constraint flags.
    Invalid/empty candidates have maximal bounded cost and cannot be selected.
    Finite but constraint-violating trajectories remain in diagnostics.
    """
    for name,value in [('planning_horizon',planning_horizon),('double_support_duration',double_support_duration),
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
    if p<1 or ccount<1: raise ValueError('nonempty candidate/sample dimensions required')
    expected=[(support_mask,(n,2,p)),(com,(n,3)),(com_velocity,(n,3)),(angular_momentum,(n,3)),
              (command_xy,(n,2)),(new_support_foot,(n,)),(candidate_mask,(n,ccount,p)),
              (touchdown_times,(n,ccount)),(foot_positions,(n,2,3))]
    if any(x.shape!=shape for x,shape in expected): raise ValueError('state/mask/time tensor shape mismatch')
    if candidate_centers is not None and candidate_centers.shape!=(n,ccount,3): raise ValueError('candidate_centers shape mismatch')
    device=support_points.device
    values=[x.to(device=device,dtype=torch.float32) for x in (support_points,com,com_velocity,angular_momentum,command_xy,candidate_points,touchdown_times,foot_positions)]
    actual,c0,v0,L0,cmd,candidates,tau,feet=values
    actual_mask=support_mask.to(device=device,dtype=torch.bool)
    new_mask=candidate_mask.to(device=device,dtype=torch.bool)
    mass=torch.as_tensor(mass,device=device,dtype=torch.float32).expand(n)
    foot=new_support_foot.to(device=device)
    centers=candidates.mean(-2) if candidate_centers is None else candidate_centers.to(device=device,dtype=torch.float32)
    valid=torch.isfinite(c0).all(-1)&torch.isfinite(v0).all(-1)&torch.isfinite(L0).all(-1)&torch.isfinite(cmd).all(-1)
    valid &= torch.isfinite(mass)&(mass>0)&(mass<=1e6)&((foot==0)|(foot==1))&torch.isfinite(feet).all((-1,-2))
    valid &= (torch.isfinite(actual).all(-1)|~actual_mask).all((-1,-2))
    finite=valid[:,None]&torch.isfinite(tau)&torch.isfinite(centers).all(-1)&(torch.isfinite(candidates).all(-1)|~new_mask).all(-1)
    clean=lambda x:torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
    actual,c0,v0,L0,cmd,candidates,tau,feet,centers=map(clean,(actual,c0,v0,L0,cmd,candidates,tau,feet,centers))
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
    geometry_valid=old_mask.any(-1)[:,None]&new_mask.any(-1)&(old_plane[:,None]<=plane_tolerance)&(new_plane<=plane_tolerance)&height_valid[:,None]&vertical_velocity_valid[:,None]
    durations=torch.stack((tau,torch.full_like(tau,double_support_duration/2),torch.full_like(tau,double_support_duration/2),planning_horizon-tau-double_support_duration),-1)
    timing_valid=(tau>0)&(durations[...,-1]>=min_final_support_duration)&(durations>0).all(-1)
    swing_distance=torch.linalg.vector_norm(centers-feet[batch,which][:,None],dim=-1)
    step_distance=torch.linalg.vector_norm(centers[...,:2]-feet[batch,1-which][:,None,:2],dim=-1)
    required_swing_speed=swing_distance/tau.clamp_min(1e-4)
    kinematic_valid=(required_swing_speed<=max_swing_speed)&(step_distance<=max_step_distance)
    finite &= (c0.abs().amax(-1)<1e6)[:,None]&(v0.abs().amax(-1)<1e3)[:,None]&(L0.abs().amax(-1)<1e6)[:,None]
    b=n*ccount
    old_xy=old_points[:,:,:2][:,None].expand(-1,ccount,-1,-1).reshape(b,p,2)
    old_valid=old_mask[:,None].expand(-1,ccount,-1).reshape(b,p)
    new_xy=candidates[...,:2].reshape(b,p,2); new_valid=new_mask.reshape(b,p)
    xy=c0[:,:2][:,None].expand(-1,ccount,-1).clone()
    velocity=v0[:,:2][:,None].expand(-1,ccount,-1).clone()
    momentum=L0[:,None].expand(-1,ccount,-1).clone()
    omega=(gravity/h).sqrt()[:,None]
    safe_dt=durations.clamp(1e-4,planning_horizon)
    shares=torch.tensor([0.,.25,.75,1.],device=device)
    band=torch.tensor(angular_momentum_band_ratio,device=device)*mg[:,None]*length_scale*momentum_reference_duration
    com_path=[c0[:,None].expand(-1,ccount,-1).clone()]
    velocity_path=[torch.cat((velocity,torch.zeros_like(velocity[...,:1])),-1)]
    momentum_path=[momentum.clone()]
    cops=[];cmps=[];foot_cops=[];forces0=[];forces1=[];frictions=[];tracking=[];rates=[];moment_rates0=[];moment_rates1=[]
    elapsed=torch.zeros_like(tau)
    for stage in range(4):
        dt=safe_dt[...,stage]
        radius=torch.linalg.vector_norm(momentum/band[:,None],dim=-1)
        excess=momentum*(1.-1./radius.clamp_min(1.))[...,None]
        requested_rate=-excess[...,:2]/(planning_horizon-elapsed).clamp_min(1e-4)[...,None]
        maximum=max_horizontal_moment_ratio*mg*length_scale
        requested_rate *= (maximum[:,None]/torch.linalg.vector_norm(requested_rate,dim=-1).clamp_min(1e-12)).clamp_max(1.)[...,None]
        offset=torch.stack((requested_rate[...,1],-requested_rate[...,0]),-1)/mg[:,None,None]
        ch=torch.cosh(omega*dt);sh=torch.sinh(omega*dt)
        required_cmp=xy+(velocity*ch[...,None]-cmd[:,None])/(omega*sh)[...,None]
        required_cop=required_cmp-offset
        old_cop,_=project_support_hull(old_xy,old_valid,required_cop.reshape(b,2),chunk_size=chunk_size)
        new_cop,_=project_support_hull(new_xy,new_valid,required_cop.reshape(b,2),chunk_size=chunk_size)
        old_cop=old_cop.reshape(n,ccount,2);new_cop=new_cop.reshape(n,ccount,2)
        share=shares[stage]
        cop=(1-share)*old_cop+share*new_cop
        cmp=cop+offset
        displacement=xy-cmp
        next_xy=cmp+displacement*ch[...,None]+velocity*(sh/omega)[...,None]
        next_velocity=displacement*(omega*sh)[...,None]+velocity*ch[...,None]
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
    return dict(candidate_objective=clean(objective),candidate_cost=cost,feasible=feasible,best_index=best,
                candidate_touchdown_times=tau,total_horizon=torch.full_like(tau,planning_horizon),phase_durations=durations,
                cop=clean(torch.stack(cops,2)),cmp=clean(torch.stack(cmps,2)),foot_cop=clean(torch.stack(foot_cops,2)),
                com_path=clean(torch.stack(com_path,2)),velocity_path=clean(torch.stack(velocity_path,2)),momentum_path=clean(torch.stack(momentum_path,2)),
                foot_force_start=clean(torch.stack(forces0,2)),foot_force_end=clean(torch.stack(forces1,2)),
                new_foot_normal_load_fraction=shares[None,None].expand(n,ccount,-1),
                friction_ratio=clean(friction_ratio),geometry_valid=geometry_valid,timing_valid=timing_valid,
                kinematic_valid=kinematic_valid,friction_valid=friction_valid,finite=finite,
                old_valid_points=old_mask.sum(-1),candidate_valid_points=new_mask.sum(-1),
                height=height,old_plane_deviation=old_plane,candidate_plane_deviation=new_plane,
                required_swing_speed=clean(required_swing_speed),step_distance=clean(step_distance),
                terminal_velocity_error=clean(velocity_error),terminal_position_error=clean(position_error),
                terminal_momentum_band_radius=clean(final_radius),progress_target=progress_target,
                actual_horizontal_momentum_rate=clean(torch.stack(rates,2)),
                stage_cop=clean(torch.stack(cops,2)),stage_cmp=clean(torch.stack(cmps,2)),
                stage_new_load=shares[None,None].expand(n,ccount,-1),
                phase_times=torch.cat((torch.zeros_like(tau)[...,None],durations.cumsum(-1)),-1),
                applied_Ldot_start=clean(torch.stack(moment_rates0,2)),applied_Ldot_end=clean(torch.stack(moment_rates1,2)),
                input_vertical_velocity=v0[:,2],vertical_velocity_valid=vertical_velocity_valid)
