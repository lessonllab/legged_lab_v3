"""Analytic constant-height LIPM support-transfer proxy, independent of simulator.

The required CMP solves v(T)=v*cosh(w*T)+w*(c-CMP)*sinh(w*T), with
w=sqrt(g/h), for the current commanded terminal horizontal velocity. A radial
engineering angular-momentum allowance supplies a constant requested Ldot;
CoP=CMP+(L_excess_y,-L_excess_x)/(mass*g*T). This is not a measured natural-gait
law. The height and support set are held constant during the horizon.

The convex hull of actual supported points is the set of *combined* CoPs from
nonnegative point-normal forces. A combined CoP may lie over a gap, but does not
create an independent gap contact force. The exact Euclidean distance to this
hull gives the minimum terminal horizontal velocity error for this LIPM proxy.
The same required CoP is compared against the new foot and both feet together.
This model omits friction, normal-force/actuator capacities, vertical dynamics,
foot timing, kinematics, and whole-body feasibility. It is not a stability or
natural-gait guarantee. The caller gates legitimate support-transition events.
"""
from __future__ import annotations

import math
import torch


@torch.no_grad()
def distance_to_support_hull(points: torch.Tensor, mask: torch.Tensor,
                             query: torch.Tensor, *, chunk_size: int = 128) -> torch.Tensor:
    """Exact 2D hull distance up to float32 tolerance; inf for empty support.

    Input points [N,K,2], boolean mask [N,K], query [N,2]. Every valid pair
    supplies a segment candidate. For a rank-two hull, CCW supporting edges
    define its inside half-planes. Outside, the closest point is on a segment;
    rank-one/zero hulls use the same segment minimum directly. Duplicates and
    singleton points are handled without inventing contacts. Cubic temporary
    storage is bounded by chunk_size (70 samples => about 176 MB at 128).
    """
    if points.ndim != 3 or points.shape[-1] != 2 or points.shape[1] == 0:
        raise ValueError('points must have shape [N,K,2], K>0')
    if mask.shape != points.shape[:2] or query.shape != (points.shape[0], 2):
        raise ValueError('mask/query shapes must match points')
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    result = []
    for start in range(0, len(points), chunk_size):
        p = points[start:start+chunk_size].float()
        m = mask[start:start+chunk_size].bool()
        q = query[start:start+chunk_size].float()
        # Subtract query for translation-invariant scale and simpler distances.
        p = torch.where(m[...,None],p-q[:,None],0.)
        d = p[:,None,:,:]-p[:,:,None,:]  # j-i
        pair = m[:,:,None] & m[:,None,:]
        length2 = d.square().sum(-1)
        t = (-(p[:,:,None,:]*d).sum(-1)/length2.clamp_min(1e-20)).clamp(0,1)
        nearest = p[:,:,None,:]+t[...,None]*d
        distances2 = torch.where(pair,nearest.square().sum(-1),float('inf'))
        segment_distance = distances2.flatten(1).amin(-1).sqrt()
        rel = p[:,None,:,:]-p[:,:,None,:]  # k-i
        cross = d[:,:,:,None,0]*rel[:,:,None,:,1]-d[:,:,:,None,1]*rel[:,:,None,:,0]
        mincross = torch.where(m[:,None,None,:],cross,float('inf')).amin(-1)
        maxcross = torch.where(m[:,None,None,:],cross,-float('inf')).amax(-1)
        # Relative geometric tolerance uses support extents, not query distance.
        tol = 2e-7*torch.where(pair,length2,0.).amax(dim=(-1,-2)).clamp_min(1e-8)
        rank2 = (pair & ((maxcross > tol[:,None,None]) | (mincross < -tol[:,None,None]))).any(dim=(-1,-2))
        edge = pair & (length2 > 1e-20) & (mincross >= -tol[:,None,None])
        querycross = -d[...,0]*p[:,:,None,1]+d[...,1]*p[:,:,None,0]
        inside = ((querycross >= -tol[:,None,None]) | ~edge).all(dim=(-1,-2))
        result.append(torch.where(rank2 & inside,0.,segment_distance))
    return torch.cat(result) if result else torch.empty(0,device=points.device,dtype=torch.float32)


@torch.no_grad()
def support_transfer_lipm_cost(
    support_points: torch.Tensor, support_mask: torch.Tensor,
    com: torch.Tensor, com_velocity: torch.Tensor, angular_momentum: torch.Tensor,
    mass: torch.Tensor, command_xy: torch.Tensor, new_support_foot: torch.Tensor,
    *, horizon: float = .3, gravity: float = 9.81, length_scale: float = .5,
    angular_momentum_band_ratio: tuple[float,float,float] = (.04,.04,.08),
    velocity_error_scale: float = .5, chunk_size: int = 128,
) -> dict[str,torch.Tensor]:
    """Same state inputs as the wrench proxy; all world-frame SI quantities.

    points [N,2,P,3], mask [N,2,P], COM/velocity/total angular momentum [N,3],
    mass scalar or [N], command [N,2], new foot [N] (0/1). Height is COM z
    minus mean actually supported point z, clamped to [0.25,1.2] m. Nonfinite
    inputs, nonpositive mass/height, invalid foot, or absent support set finite
    false. New-foot emptiness separately sets new_support=false and cost=1.
    """
    for name,value in (('horizon',horizon),('gravity',gravity),('length_scale',length_scale),('velocity_error_scale',velocity_error_scale)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if len(angular_momentum_band_ratio)!=3 or any(not math.isfinite(x) or x<=0 for x in angular_momentum_band_ratio):
        raise ValueError('angular_momentum_band_ratio must contain three positive finite semiaxes')
    if support_points.ndim!=4 or support_points.shape[1]!=2 or support_points.shape[-1]!=3 or support_points.shape[2]==0:
        raise ValueError('support_points must have shape [N,2,P,3]')
    n,_,p,_=support_points.shape
    for name,value,shape in (('support_mask',support_mask,(n,2,p)),('com',com,(n,3)),('com_velocity',com_velocity,(n,3)),('angular_momentum',angular_momentum,(n,3)),('command_xy',command_xy,(n,2)),('new_support_foot',new_support_foot,(n,))):
        if value.shape!=shape:
            raise ValueError(f'{name} must have shape {shape}')
    device=support_points.device
    points,c,v,L,cmd=(x.to(device=device,dtype=torch.float32) for x in (support_points,com,com_velocity,angular_momentum,command_xy))
    mask=support_mask.to(device=device,dtype=torch.bool)
    body_mass=torch.as_tensor(mass,device=device,dtype=torch.float32).expand(n)
    foot=new_support_foot.to(device=device)
    finite=torch.isfinite(c).all(-1)&torch.isfinite(v).all(-1)&torch.isfinite(L).all(-1)&torch.isfinite(cmd).all(-1)
    finite &= torch.isfinite(body_mass)&(body_mass>0)&((foot==0)|(foot==1))
    finite &= (torch.isfinite(points).all(-1)|~mask).all(dim=(-1,-2))
    clean=lambda x:torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
    points=clean(points);c=clean(c);v=clean(v);L=clean(L);cmd=clean(cmd)
    mg=clean(body_mass).clamp_min(1e-6)*gravity
    finite &= torch.isfinite(mg)
    mg=clean(mg).clamp_min(1e-6)
    count=mask.sum(dim=(-1,-2))
    support=count>0
    ground=(points[...,2]*mask).sum(dim=(-1,-2))/count.clamp_min(1)
    raw_height=c[:,2]-ground
    finite &= support&torch.isfinite(raw_height)&(raw_height>0)
    height=raw_height.clamp(.25,1.2)
    omega=(gravity/height).sqrt()
    velocity_gain=omega*torch.sinh(omega*horizon)
    cmp=c[:,:2]+(v[:,:2]*torch.cosh(omega*horizon)[:,None]-cmd)/velocity_gain[:,None]
    normalized=L/(mg[:,None]*horizon*length_scale)
    band=torch.tensor(angular_momentum_band_ratio,device=device)
    radius=torch.linalg.vector_norm(normalized/band,dim=-1)
    excess=L*(1.-1./radius.clamp_min(1.))[:,None]
    shift=torch.stack((excess[:,1],-excess[:,0]),-1)/(mg[:,None]*horizon)
    required=cmp+shift
    finite &= torch.isfinite(required).all(-1)&torch.isfinite(excess).all(-1)&torch.isfinite(velocity_gain)
    required=clean(required);excess=clean(excess)
    flat_points=points[...,:2].reshape(n,2*p,2)
    held_mask=mask.reshape(n,2*p)&finite[:,None]
    # Gather only the selected sole for its hull: P^3 instead of (2P)^3.
    # Invalid foot values are already rejected by finite; clamp only for indexing.
    batch_index=torch.arange(n,device=device)
    selected_foot=foot.long().clamp(0,1)
    new_points=points[batch_index,selected_foot,:,:2]
    new_mask=mask[batch_index,selected_foot]&finite[:,None]
    held_distance=distance_to_support_hull(flat_points,held_mask,required,chunk_size=chunk_size)
    new_distance=distance_to_support_hull(new_points,new_mask,required,chunk_size=chunk_size)
    new_support=new_mask.any(-1)
    held_error=velocity_gain*held_distance
    new_error=velocity_gain*new_distance
    held_cost=-torch.expm1(-(held_error/velocity_error_scale).square())
    new_cost=-torch.expm1(-(new_error/velocity_error_scale).square())
    held_cost=torch.where(finite,held_cost,1.)
    new_cost=torch.where(finite,new_cost,1.)
    return dict(p_required=required,p_required_cmp=clean(cmp),new_cost=new_cost,held_cost=held_cost,
                increment=(new_cost-held_cost).clamp(0,1),finite=finite,support=support,
                new_support=new_support,valid_counts=mask.sum(-1),momentum_excess=excess,
                height=height,raw_height=raw_height,height_clamped=raw_height!=height,
                velocity_gain=velocity_gain,new_distance=new_distance,held_distance=held_distance,
                new_velocity_error=new_error,held_velocity_error=held_error)
