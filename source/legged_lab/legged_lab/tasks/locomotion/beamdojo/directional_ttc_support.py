"""Research-only command-conditioned weighting of predicted unsafe sole points.

No simulator imports or environment hook. Callers must supply actual whole-body
CoM position/velocity, not substitute root position/velocity. The LIPM expression
is a heuristic required-CoP direction, not a contact feasibility certificate.
"""
import math
import torch


def directional_ttc_cost(predicted_points, unsafe, com_xy, com_velocity_xy,
                         command_world_xy, tau, gate, *, com_height,
                         step_time=.4, alpha=.5, gravity=9.81):
    """Return (negative cost[N], diagnostics) without changing input tensors.

    Points[N,2,35,3] are world points already predicted by the caller's TTC.
    unsafe[N,2,35] and gate[N,2] are the original boolean geometry/gating result.
    Whole-body CoM, its velocity and command have shape[N,2] in world XY.
    tau is scalar or[N,2] seconds; com_height scalar or[N] metres above the
    modeled support plane. gate is reused without additional speed/phase rules.
    """
    p=predicted_points
    if p.ndim!=4 or p.shape[1:]!=(2,35,3) or not p.is_floating_point():
        raise ValueError('Expected floating world points[N,2,35,3]')
    n=p.shape[0]
    if unsafe.shape!=(n,2,35) or unsafe.dtype!=torch.bool or gate.shape!=(n,2) or gate.dtype!=torch.bool:
        raise ValueError('Original unsafe/gate boolean schema required')
    for value in (com_xy,com_velocity_xy,command_world_xy):
        if value.shape!=(n,2) or value.device!=p.device or not torch.isfinite(value).all():
            raise ValueError('Finite true CoM/velocity/world-command[N,2] required')
    if unsafe.device!=p.device or gate.device!=p.device or not torch.isfinite(p).all():
        raise ValueError('Finite colocated inputs required')
    if not all(math.isfinite(x) for x in (step_time,alpha,gravity)) or step_time<=0 or gravity<=0 or not 0<=alpha<=1:
        raise ValueError('Positive finite step_time/gravity and alpha in[0,1]')
    tau=torch.broadcast_to(torch.as_tensor(tau,dtype=p.dtype,device=p.device),(n,2))
    height=torch.broadcast_to(torch.as_tensor(com_height,dtype=p.dtype,device=p.device),(n,))
    if not torch.isfinite(tau).all() or (tau<0).any() or not torch.isfinite(height).all() or (height<=0).any():
        raise ValueError('Finite nonnegative tau and positive CoM height required')
    c=com_xy.to(p.dtype);v=com_velocity_xy.to(p.dtype);cmd=command_world_xy.to(p.dtype)
    omega=(gravity/height).sqrt()
    wt=omega*step_time
    correction=(torch.cosh(wt)[:,None]*v-cmd)/(omega*torch.sinh(wt))[:,None]
    required=c[:,None,:]+tau[:,:,None]*v[:,None,:]+correction[:,None,:]
    center=p[...,:2].mean(-2)
    direction=required-center
    u=direction/(direction.norm(dim=-1,keepdim=True)+.02)
    offsets=p[...,:2]-center[:,:,None,:]
    radius=offsets.norm(dim=-1).amax(-1,keepdim=True)
    projection=(offsets*u[:,:,None,:]).sum(-1)
    scaled=torch.where(radius>0,projection/radius.clamp_min(torch.finfo(p.dtype).tiny),torch.zeros_like(projection))
    weights=1.+alpha*scaled
    weighted_fraction=(unsafe.to(p.dtype)*weights).sum(-1)/weights.sum(-1)
    # Preserve the original uniform cost's operation/dtype exactly at alpha=0.
    if alpha==0:
        cost=-(gate.float()*unsafe.float().mean(-1)).sum(-1)
    else:
        cost=-(gate.to(p.dtype)*weighted_fraction).sum(-1)
    if not torch.isfinite(required).all() or not torch.isfinite(weights).all() or not torch.isfinite(cost).all():
        raise ValueError('Nonfinite LIPM weighting; invalid modeling scale')
    return cost,dict(required_cop_xy=required,point_weights=weights,
                     weighted_unsafe_fraction=weighted_fraction,
                     assumption='constant-height LIPM heuristic; true whole-body CoM required; not contact feasibility')
