"""Sample the joint planner's piecewise reference in physical time.

This is a training-reference building block, not a controller or a reward
installed in the environment. It preserves the planner's constant-height
approximation and uses its SAME CoP/CMP to evolve COM and total momentum.
In particular, it does not interpolate independently chosen position,
velocity and angular-momentum targets. References outside the fixed physical
planning interval are marked invalid instead of moving their deadlines.
"""
from __future__ import annotations

import math
import torch


@torch.no_grad()
def sample_contact_reference(phase_times, com_path, velocity_path, momentum_path,
                             cop, cmp, height, mass, time, *, gravity=9.81):
    """Inputs selected from ONE candidate per environment, world SI units.

    phase_times [N,5] are absolute simulation times, paths [N,5,3], CoP/CMP
    [N,4,2], height/mass [N] (mass can be scalar), time scalar or [N].
    `in_interval` is a time/finite-input gate, NOT planner feasibility.
    Callers must also apply their stored plan_valid/episode/swing identity.
    """
    if not math.isfinite(gravity) or gravity <= 0:
        raise ValueError('gravity must be positive and finite')
    if phase_times.ndim != 2 or phase_times.shape[1] != 5:
        raise ValueError('phase_times must have shape [N,5]')
    n=phase_times.shape[0]
    if any(x.shape != (n,5,3) for x in (com_path,velocity_path,momentum_path)):
        raise ValueError('reference paths must have shape [N,5,3]')
    if cop.shape != (n,4,2) or cmp.shape != (n,4,2) or height.shape != (n,):
        raise ValueError('selected CoP/CMP/height shapes do not match')
    device=phase_times.device
    values=[x.to(device=device,dtype=torch.float32) for x in
            (phase_times,com_path,velocity_path,momentum_path,cop,cmp,height)]
    times,c,v,L,p,q,h=values
    m=torch.as_tensor(mass,device=device,dtype=torch.float32).expand(n)
    t=torch.as_tensor(time,device=device,dtype=torch.float32).expand(n)
    finite=torch.isfinite(times).all(-1)&((times[:,1:]-times[:,:-1])>0).all(-1)
    for x in (c,v,L,p,q):
        finite &= torch.isfinite(x).flatten(1).all(-1)
    finite &= torch.isfinite(h)&(h>0)&torch.isfinite(m)&(m>0)&torch.isfinite(t)
    inside=finite&(t>=times[:,0])&(t<=times[:,-1])
    # Invalid values are sanitized only for arithmetic; they remain rejected.
    clean=lambda x:torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
    times,c,v,L,p,q,h,m,t=map(clean,(times,c,v,L,p,q,h,m,t))
    stage=(t[:,None]>=times[:,1:]).sum(-1).clamp_max(3)
    rows=torch.arange(n,device=device)
    start,end=times[rows,stage],times[rows,stage+1]
    dt=(t-start).clamp_min(0.).minimum((end-start).clamp_min(0.))
    c0,v0,L0=c[rows,stage],v[rows,stage],L[rows,stage]
    phase_cmp,phase_cop=q[rows,stage],p[rows,stage]
    # Names below match the equations: d=c-CMP, offset=CMP-CoP.
    displacement=c0[:,:2]-phase_cmp
    offset=phase_cmp-phase_cop
    omega=(gravity/h.clamp_min(1e-6)).sqrt()
    ch,sh=torch.cosh(omega*dt),torch.sinh(omega*dt)
    xy=phase_cmp+displacement*ch[:,None]+v0[:,:2]*(sh/omega)[:,None]
    velocity_xy=displacement*(omega*sh)[:,None]+v0[:,:2]*ch[:,None]
    integral=displacement*(sh/omega)[:,None]+v0[:,:2]*((ch-1)/omega.square())[:,None]
    mg=m*gravity
    rate_xy=mg[:,None]*torch.stack((phase_cop[:,1]-phase_cmp[:,1],phase_cmp[:,0]-phase_cop[:,0]),-1)
    yaw_delta=-m*omega.square()*(offset[:,0]*integral[:,1]-offset[:,1]*integral[:,0])
    momentum=L0+torch.cat((rate_xy*dt[:,None],yaw_delta[:,None]),-1)
    com=torch.cat((xy,c0[:,2:3]),-1)
    velocity=torch.cat((velocity_xy,torch.zeros_like(velocity_xy[:,:1])),-1)
    shares=torch.tensor([0.,.25,.75,1.],device=device)[stage]
    finite_result=torch.isfinite(com).all(-1)&torch.isfinite(velocity).all(-1)&torch.isfinite(momentum).all(-1)
    return dict(com=clean(com),velocity=clean(velocity),momentum=clean(momentum),
                new_foot_normal_load_fraction=shares,stage=stage,
                phase_elapsed=dt,in_interval=inside&finite_result,
                fixed_touchdown_deadline=times[:,1],fixed_reference_end=times[:,-1])
