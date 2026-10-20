"""Seven-phase SAME-FORCE analytic reference for the two-contact proxy.

Read-only reference sampling, not a reward/controller or feasibility claim.
The absolute input timeline is never reordered or shifted. Each query uses
one phase's fixed CoP/CMP and integrates COM, velocity and total xyz angular
momentum from that phase's stored initial state. Touchdowns introduce no
momentum reset. Modeled vertical velocity stays zero as in the planner.
"""
from __future__ import annotations
import math
import torch


@torch.no_grad()
def sample_two_contact_reference(phase_times, com_path, velocity_path, momentum_path,
                                 cop, cmp, height, mass, time, *,
                                 foot_normal_load_fractions=None, plan_valid=True, gravity=9.81):
    """World SI inputs: absolute times [N,8], paths [N,8,3], CoP/CMP [N,7,2].

    Height is [N]; mass/time accept scalar or [N]. Optional load fractions are
    [7,2] or [N,7,2], with physical slots [A (later A-prime), B]; default is
    the two-touchdown sequence. For a hold-B reference pass that planner's
    actual fractions explicitly. plan_valid is bool or boolean [N], normally
    the stored selected-candidate validity combined with episode identity.

    A phase boundary belongs to the next phase, except the final endpoint,
    which belongs to phase 6. Position/velocity/momentum remain continuous;
    contact force and Ldot may jump at these prescribed switching times.
    Invalid shape raises ValueError. Nonfinite/nonincreasing times, invalid
    mass/height/load weights, or nonfinite paths mark valid=False. Queries
    outside the absolute interval are invalid; their arithmetic dt is bounded
    only to avoid extrapolation, never to move a target's deadline.
    """
    if not math.isfinite(gravity) or gravity<=0:raise ValueError('gravity must be finite and positive')
    if phase_times.ndim!=2 or phase_times.shape[1]!=8:raise ValueError('phase_times must be [N,8]')
    n=phase_times.shape[0]
    if any(x.shape!=(n,8,3) for x in (com_path,velocity_path,momentum_path)):raise ValueError('paths must be [N,8,3]')
    if cop.shape!=(n,7,2) or cmp.shape!=(n,7,2) or height.shape!=(n,):raise ValueError('CoP/CMP/height shape mismatch')
    device=phase_times.device
    dtype=torch.float64 if any(x.dtype==torch.float64 for x in (phase_times,com_path,velocity_path,momentum_path,cop,cmp,height)) else torch.float32
    values=[x.to(device=device,dtype=dtype) for x in (phase_times,com_path,velocity_path,momentum_path,cop,cmp,height)]
    times,c,v,L,p,q,h=values
    def vector(value,name):
        value=torch.as_tensor(value,device=device,dtype=dtype)
        if value.ndim==0:return value.expand(n)
        if value.shape!=(n,):raise ValueError(name+' must be scalar or [N]')
        return value
    m=vector(mass,'mass');t=vector(time,'time')
    if type(plan_valid) is bool:gate=torch.full((n,),plan_valid,device=device,dtype=torch.bool)
    elif isinstance(plan_valid,torch.Tensor) and plan_valid.dtype==torch.bool and plan_valid.shape==(n,):gate=plan_valid.to(device=device)
    else:raise ValueError('plan_valid must be bool or boolean [N]')
    if foot_normal_load_fractions is None:
        shares=torch.tensor([[1.,0.],[.75,.25],[.25,.75],[0.,1.],[.25,.75],[.75,.25],[1.,0.]],device=device,dtype=dtype)[None].expand(n,-1,-1)
    else:
        shares=torch.as_tensor(foot_normal_load_fractions,device=device,dtype=dtype)
        if shares.shape==(7,2):shares=shares[None].expand(n,-1,-1)
        elif shares.shape!=(n,7,2):raise ValueError('foot_normal_load_fractions must be [7,2] or [N,7,2]')
    finite=torch.isfinite(times).all(-1)&((times[:,1:]-times[:,:-1])>0).all(-1)
    for value in (c,v,L,p,q):finite &= torch.isfinite(value).flatten(1).all(-1)
    finite &= torch.isfinite(h)&(h>0)&torch.isfinite(m)&(m>0)&torch.isfinite(t)
    finite &= torch.isfinite(shares).all((-1,-2))&(shares>=0).all((-1,-2))&(shares<=1).all((-1,-2))&((shares.sum(-1)-1).abs()<=1e-6).all(-1)
    inside=finite&(t>=times[:,0])&(t<=times[:,-1])
    # Sanitize only untrusted arithmetic; keep the original timelines in output.
    clean=lambda x:torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
    safe_times,c,v,L,p,q,h,m,safe_t,shares=map(clean,(times,c,v,L,p,q,h,m,t,shares))
    stage=(safe_t[:,None]>=safe_times[:,1:]).sum(-1).clamp_max(6)
    rows=torch.arange(n,device=device)
    start,end=safe_times[rows,stage],safe_times[rows,stage+1]
    dt=(safe_t-start).clamp_min(0.).minimum((end-start).clamp_min(0.))
    dt=torch.where(finite,dt,torch.zeros_like(dt))
    c0,v0,L0=c[rows,stage],v[rows,stage],L[rows,stage]
    phase_cop,phase_cmp=p[rows,stage],q[rows,stage]
    displacement=c0[:,:2]-phase_cmp;offset=phase_cmp-phase_cop
    omega=(gravity/h.clamp_min(1e-6)).sqrt()
    ch,sh=torch.cosh(omega*dt),torch.sinh(omega*dt)
    xy=phase_cmp+displacement*ch[:,None]+v0[:,:2]*(sh/omega)[:,None]
    velocity_xy=displacement*(omega*sh)[:,None]+v0[:,:2]*ch[:,None]
    integral=displacement*(sh/omega)[:,None]+v0[:,:2]*((ch-1)/omega.square())[:,None]
    mg=m*gravity
    rate_xy=mg[:,None]*torch.stack((phase_cop[:,1]-phase_cmp[:,1],phase_cmp[:,0]-phase_cop[:,0]),-1)
    yaw_delta=-m*omega.square()*(offset[:,0]*integral[:,1]-offset[:,1]*integral[:,0])
    momentum=L0+torch.cat((rate_xy*dt[:,None],yaw_delta[:,None]),-1)
    center=torch.cat((xy,c0[:,2:3]),-1)
    velocity=torch.cat((velocity_xy,torch.zeros_like(velocity_xy[:,:1])),-1)
    force=torch.cat((m[:,None]*omega.square()[:,None]*(xy-phase_cmp),mg[:,None]),-1)
    ground=c0[:,2]-h
    cop_xyz=torch.cat((phase_cop,ground[:,None]),-1)
    Ldot=torch.cross(cop_xyz-center,force,dim=-1)
    finite_result=torch.isfinite(center).all(-1)&torch.isfinite(velocity).all(-1)&torch.isfinite(momentum).all(-1)&torch.isfinite(Ldot).all(-1)&torch.isfinite(force).all(-1)
    selected_shares=shares[rows,stage]
    return dict(com=clean(center),velocity=clean(velocity),momentum=clean(momentum),Ldot=clean(Ldot),momentum_rate=clean(Ldot),
        cop=clean(phase_cop),cmp=clean(phase_cmp),contact_force=clean(force),
        foot_normal_load_fractions=selected_shares,foot_forces=clean(force[:,None]*selected_shares[...,None]),
        stage_idx=stage,stage=stage,phase_elapsed=dt,finite=finite&finite_result,
        in_interval=inside&finite_result,valid=inside&finite_result&gate,plan_valid=gate,
        sample_time=t,phase_times=times,first_touchdown_deadline=times[:,1],second_touchdown_deadline=times[:,4],fixed_reference_end=times[:,-1])


# Descriptive primary name plus compatible reference-family entry name.
sample_contact_reference=sample_two_contact_reference
