"""Deployment-side pure Torch support-choice geometry and local differential IK.

No contact forces, world yaw, simulator Jacobians or terrain truth are read.
The 15x15 ij heightmap is a sampled observation, NOT a contact/feasibility proof.
Discrete candidates remain discrete: neither interpolation nor a hull bridges
holes. FK is an instantaneous configuration map, NEVER a future landing target.
G1 six-joint leg origins/axes below were checked against this repository's
 g1_29dof_rev_1_0.usd PhysicsRevoluteJoint local frames, 2026-10-10, using pxr
read-only (and matching Unitree g1_29dof.urdf). Joint angles are absolute radians.
"""
import math
import torch

LEG_SUFFIXES = ('hip_pitch_joint','hip_roll_joint','hip_yaw_joint','knee_joint','ankle_pitch_joint','ankle_roll_joint')


def command_basis(command_xy, minimum_speed=.2):
    """Columns forward/left in robot yaw-local coordinates; zero command invalid."""
    if command_xy.shape[-1] != 2: raise ValueError('command must end in 2')
    finite=torch.isfinite(command_xy).all(-1)
    c=torch.nan_to_num(command_xy); norm=c.norm(dim=-1)
    valid=finite&(norm>=minimum_speed)
    e=c/norm.clamp_min(1e-8)[...,None]
    e=torch.where(valid[...,None],e,torch.tensor([1.,0.],device=c.device,dtype=c.dtype))
    return torch.stack((e,torch.stack((-e[...,1],e[...,0]),-1)),-1),valid


def transform_xy(xy,basis,*,to_command=False):
    """xy [N,...,2], basis [N,2,2]. Translations must be removed by caller."""
    r=basis.transpose(-1,-2) if to_command else basis
    return torch.einsum('nij,n...j->n...i',r,xy)


def role_normalize_xy(xy,swing_foot):
    """Mirror right swing into left-swing role. This is XY only, not gyro mirroring."""
    sign=torch.where(swing_foot==1,-1.,1.).to(xy)
    shape=(len(xy),)+(1,)*(xy.ndim-2)
    return torch.stack((xy[...,0],xy[...,1]*sign.reshape(shape)),-1)


def _rotation(axis,angle):
    axis=torch.as_tensor(axis,device=angle.device,dtype=angle.dtype)
    x,y,z=axis.unbind();zero=torch.zeros_like(x)
    K=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
    eye=torch.eye(3,device=angle.device,dtype=angle.dtype)
    return eye+torch.sin(angle)[...,None,None]*K+(1-torch.cos(angle))[...,None,None]*(K@K)


def g1_leg_fk_jacobian(joint_position,joint_names,foot_offset=(0.,0.,0.)):
    """Return position[N,2,3], Jacobian[N,2,3,J], rotation[N,2,3,3].

    Root body coordinates (not yaw-level). Caller may use IMU roll/pitch to
    rotate vectors into yaw-level map coordinates, without any global heading.
    Six joints each; ankle-roll origin is the foot link origin by default.
    """
    if joint_position.ndim!=2 or len(joint_names)!=joint_position.shape[-1] or len(set(joint_names))!=len(joint_names):
        raise ValueError('unique joint_names must match [N,J]')
    n,j=joint_position.shape;dtype=joint_position.dtype;device=joint_position.device
    axes=((0,1,0),(1,0,0),(0,0,1),(0,1,0),(0,1,0),(1,0,0))
    positions=[];jacobians=[];rotations=[]
    for side,sign in (('left',1),('right',-1)):
        indices=[joint_names.index(side+'_'+name) for name in LEG_SUFFIXES]
        xyz=((0,sign*.064452,-.1027),(0,sign*.052,-.030465),(.025001,0,-.12412),(-.078273,sign*.0021489,-.17734),(0,-sign*.000094445,-.30001),(0,0,-.017558))
        pitch=(0.,-.1749,0.,.1749,0.,0.)
        R=torch.eye(3,device=device,dtype=dtype).expand(n,3,3).clone();p=torch.zeros(n,3,device=device,dtype=dtype)
        origins=[];directions=[]
        for k,index in enumerate(indices):
            p=p+torch.einsum('nij,j->ni',R,torch.tensor(xyz[k],device=device,dtype=dtype))
            R=R@_rotation((0,1,0),joint_position.new_full((n,),pitch[k]))
            origins.append(p);directions.append(torch.einsum('nij,j->ni',R,torch.tensor(axes[k],device=device,dtype=dtype)))
            R=R@_rotation(axes[k],joint_position[:,index])
        p=p+torch.einsum('nij,j->ni',R,torch.as_tensor(foot_offset,device=device,dtype=dtype))
        jac=torch.zeros(n,3,j,device=device,dtype=dtype)
        for index,origin,axis in zip(indices,origins,directions):jac[:,:,index]=torch.cross(axis,p-origin,dim=-1)
        positions.append(p);jacobians.append(jac);rotations.append(R)
    return torch.stack(positions,1),torch.stack(jacobians,1),torch.stack(rotations,1)


def level_rotation(roll,pitch):
    """R_y(pitch) R_x(roll): body->yaw-level, no global yaw needed."""
    return _rotation((0,1,0),pitch)@_rotation((1,0,0),roll)


def decode_heightmap(heightmap):
    """Observed terrain Z relative root Z; +/-5 are clipped and unknown."""
    h=heightmap.reshape(-1,15,15)
    valid=torch.isfinite(h)&(h.abs()<5.)
    return -.5-torch.nan_to_num(h)/5.,valid


def sole_offsets(device='cpu',dtype=torch.float32):
    x,y=torch.meshgrid(torch.linspace(-.055,.145,7,device=device,dtype=dtype),torch.linspace(-.035,.035,5,device=device,dtype=dtype),indexing='ij')
    return torch.stack((x.flatten(),y.flatten()),-1)


def observed_candidate_support(heightmap,candidate_xy,expected_height,*,foot_yaw=None,height_tolerance=.035,support_threshold=.65):
    """[N,C,2] centers, expected_height[N,C] relative root. Four neighboring
    samples per sole point vote on expected surface height. valid uses the
    explicit support_threshold; strict_all requires every vote and is diagnostic.
    This coarse observation fraction can miss gaps and is NOT exact full-sole
    support, contact, reach or dynamics feasibility. Unknown votes count zero.
    """
    if candidate_xy.ndim!=3 or candidate_xy.shape[-1]!=2 or expected_height.shape!=candidate_xy.shape[:2]:raise ValueError('candidate shape mismatch')
    n,c,_=candidate_xy.shape;h,hvalid=decode_heightmap(heightmap)
    offsets=sole_offsets(candidate_xy.device,candidate_xy.dtype)[None,None].expand(n,c,-1,-1)
    if foot_yaw is not None:
        co=foot_yaw.cos()[...,None];si=foot_yaw.sin()[...,None];x,y=offsets.unbind(-1)
        offsets=torch.stack((co*x-si*y,si*x+co*y),-1)
    xy=candidate_xy[:,:,None]+offsets;grid=(xy+.7)/.1
    lower=torch.floor(torch.nan_to_num(grid)).long();upper=torch.ceil(torch.nan_to_num(grid)).long()
    mask=torch.isfinite(xy).all(-1)&((grid>=0)&(grid<=14)).all(-1)
    samples=[]; known_samples=[]; supported_samples=[]
    rows=torch.arange(n,device=h.device)[:,None,None]
    for ix,iy in ((lower[...,0],lower[...,1]),(lower[...,0],upper[...,1]),(upper[...,0],lower[...,1]),(upper[...,0],upper[...,1])):
        ix=ix.clamp(0,14);iy=iy.clamp(0,14);sample=h[rows,ix,iy]
        inside=torch.isfinite(xy).all(-1)&((grid>=0)&(grid<=14)).all(-1)
        encoded=heightmap.reshape(n,15,15)[rows,ix,iy]
        # +5 means terrain_z<=-1.5 relative root; -5 means terrain_z>=+.5.
        # A clipping bound can prove INCOMPATIBILITY, never compatibility.
        clipped_incompatible=((encoded==5.)&(expected_height[...,None]>-1.5+height_tolerance))|((encoded==-5.)&(expected_height[...,None]<.5-height_tolerance))
        known=(hvalid[rows,ix,iy]|clipped_incompatible)&inside
        supported=hvalid[rows,ix,iy]&inside&((sample-expected_height[...,None]).abs()<=height_tolerance)
        mask &= supported
        known_samples.append(known);supported_samples.append(supported);samples.append(sample)
    if not 0<=support_threshold<=1:raise ValueError('support threshold must be within [0,1]')
    observed_fraction=torch.stack(supported_samples,-1).to(h.dtype).mean((-1,-2))
    known_fraction=torch.stack(known_samples,-1).to(h.dtype).mean((-1,-2))
    strict=mask.all(-1)&torch.isfinite(expected_height)
    return dict(valid=(observed_fraction>=support_threshold)&torch.isfinite(expected_height),strict_all=strict,observed_support_fraction=observed_fraction,observed_known_fraction=known_fraction,sole_sample_valid=mask,observed_heights=torch.stack(samples,-1))


def candidate_compatibility(predicted_xy,candidate_xy,candidate_valid,confidence,*,compatibility_radius=.03,confidence_threshold=.7):
    """Nearest DISCRETE candidate, no convexification. Confidence gates use of
    empirical next-landing prediction, not a probability of whole-body success.
    Empty/unknown sets disable intervention, never pretend distance is zero.
    """
    valid=candidate_valid&torch.isfinite(candidate_xy).all(-1)
    dist=(candidate_xy-predicted_xy[:,None]).norm(dim=-1).masked_fill(~valid,torch.inf)
    distance,index=dist.min(-1);known=valid.any(-1)&torch.isfinite(predicted_xy).all(-1)&torch.isfinite(confidence)&(confidence>=confidence_threshold)
    closest=candidate_xy[torch.arange(len(index),device=index.device),index]
    return dict(distance=distance,index=torch.where(known,index,-1),closest=torch.where(known[:,None],closest,torch.zeros_like(closest)),known=known,compatible=known&(distance<=compatibility_radius),modification_gate=(known&(distance>compatibility_radius)).to(candidate_xy.dtype))


def bounded_xy_joint_residual(delta_xy,jacobian_xy,joint_mask,*,confidence=None,max_joint_delta=.08,damping=.03):
    """Damped differential IK; common scaling preserves its direction. No joint
    limit, collision, contact or trajectory feasibility guarantee. Limit absolute
    target positions separately in the original controller. Units m, rad.
    """
    if damping<=0 or max_joint_delta<=0:raise ValueError('positive damping and joint bound required')
    if jacobian_xy.shape[:2]!=(len(delta_xy),2) or joint_mask.shape!=jacobian_xy.shape[::2]:raise ValueError('Jacobian/mask shape mismatch')
    finite=torch.isfinite(delta_xy).all(-1)&torch.isfinite(jacobian_xy).all((-1,-2))
    J=torch.nan_to_num(jacobian_xy)*joint_mask[:,None].to(jacobian_xy)
    dx=torch.nan_to_num(delta_xy)
    if confidence is not None:
        finite &= torch.isfinite(confidence);dx=dx*torch.nan_to_num(confidence).clamp(0,1)[:,None]
    A=J@J.transpose(-1,-2)+damping*damping*torch.eye(2,device=J.device,dtype=J.dtype)
    dq=(J.transpose(-1,-2)@torch.linalg.solve(A,dx[...,None])).squeeze(-1)
    scale=(max_joint_delta/dq.abs().amax(-1).clamp_min(1e-12)).clamp_max(1.)
    dq=dq*scale[:,None]
    return torch.where(finite[:,None],dq,torch.zeros_like(dq))


def map_support_choice(latent,context):
    """Four latent entries = left XY/right XY in command frame; only inferred
    swing foot active. Required context: swing_foot[N],confidence[N],
    jacobian_xy[N,2,2,J] in yaw-local frame,joint_mask[N,2,J],action_scale[J],
    command_xy[N,2]. Optional modification_gate[N],max_xy(.04),
    max_joint_delta(.08),damping(.03). Return residual ACTION[N,J].
    Zero latent gives exact zero even when no leg/command/confidence is valid.
    Actor adds this to frozen teacher action before unchanged action processing.
    """
    if latent.ndim!=2 or latent.shape[1]!=4:raise ValueError('latent must be [N,4]')
    foot=context['swing_foot'];rows=torch.arange(len(latent),device=latent.device);idx=foot.clamp(0,1).long()
    confidence=context['confidence'];basis,moving=command_basis(context['command_xy'])
    gate=context.get('modification_gate',torch.ones_like(confidence))
    active=((foot==0)|(foot==1))&moving&torch.isfinite(confidence)&torch.isfinite(gate)
    dx=latent.reshape(-1,2,2)[rows,idx].tanh()*float(context.get('max_xy',.04))
    dx=transform_xy(dx,basis)
    scale=torch.as_tensor(context['action_scale'],device=latent.device,dtype=latent.dtype)
    if not torch.isfinite(scale).all() or (scale<=0).any():raise ValueError('positive finite action scales required')
    weight=torch.where(active,confidence.clamp(0,1)*gate.clamp(0,1),0.)
    dq=bounded_xy_joint_residual(dx,context['jacobian_xy'][rows,idx],context['joint_mask'][rows,idx],confidence=weight,max_joint_delta=float(context.get('max_joint_delta',.08)),damping=float(context.get('damping',.03)))
    return dq/scale


def support_choice_gate(candidate_weights,observed_fraction,map_known,prediction_confidence,*,confidence_threshold=.7,min_observed_mass=.8,support_threshold=.65):
    """Reliable old choices incompatible with KNOWN terrain enable modification.

    [N,K] nonnegative empirical weights; support/known fractions [N,K] in [0,1].
    Knownness is independent of support: observed low surfaces are known failures,
    not missing data. Unknown support votes do not count as incompatibility.
    Normalize positive finite weights; no candidates/predictor confidence or
    insufficient observed mass returns zero intervention. Among observed mass,
    compatible_mass is the probability of a candidate exceeding the declared
    observation-only support threshold. Gate = 1-compatible_mass/observed_mass.
    Both masses are unconditional under the normalized empirical distribution.
    """
    if candidate_weights.shape!=observed_fraction.shape or map_known.shape!=candidate_weights.shape or candidate_weights.ndim!=2:raise ValueError('candidate inputs must share [N,K]')
    if not 0<=confidence_threshold<=1 or not 0<min_observed_mass<=1 or not 0<=support_threshold<=1:raise ValueError('invalid thresholds')
    w=candidate_weights;f=observed_fraction;k=map_known.to(w.dtype)
    good=torch.isfinite(w)&(w>=0)&torch.isfinite(f)&(f>=0)&(f<=1)&torch.isfinite(k)&(k>=0)&(k<=1)
    # A malformed positive-weight candidate makes its entire distribution unknown.
    distribution_valid=(good|(torch.isfinite(w)&(w==0))).all(-1)
    safe_w=torch.where(good,w,0.);total=safe_w.sum(-1)
    weights=safe_w/total.clamp_min(1e-12)[:,None]
    known=torch.where(good,k,0.)
    observed_mass=(weights*known).sum(-1)
    compatible_mass=(weights*known*(torch.nan_to_num(f)>=support_threshold)).sum(-1)
    predictor_valid=distribution_valid&(total>0)&torch.isfinite(prediction_confidence)&(prediction_confidence>=confidence_threshold)
    reliable=predictor_valid&(observed_mass>=min_observed_mass)
    gate=torch.where(reliable,1-compatible_mass/observed_mass.clamp_min(1e-12),0.).clamp(0,1)
    return dict(modification_gate=gate,compatible_mass=compatible_mass,observed_mass=observed_mass,predictor_valid=predictor_valid,known=reliable)
