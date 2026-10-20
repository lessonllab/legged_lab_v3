"""Optional command-conditioned terrain attention; not enabled by existing arms.

Consumes only deployed observations and estimated role probabilities. Attention
is an architectural ablation, not itself a novel algorithm or a safety gate.
"""
import math

import torch
from torch import nn


def command_frame_coordinates(points_xy, command_xy, eps=1.e-6):
    """Rotate body-frame points into commanded-forward/left coordinates.

    Inputs [P,2] or [N,P,2], command [N,2]. Standing uses body +X with zero
    directional bias. Simultaneously rotating points and commands preserves
    these coordinates for moving commands; no world yaw is required.
    """
    if command_xy.ndim != 2 or command_xy.shape[-1] != 2:
        raise ValueError('Expected command [N,2]')
    if points_xy.ndim not in (2,3) or points_xy.shape[-1] != 2:
        raise ValueError('Expected points [P,2] or [N,P,2]')
    speed=command_xy.norm(dim=-1,keepdim=True)
    fallback=torch.zeros_like(command_xy);fallback[:,0]=1.
    direction=torch.where(speed>eps,command_xy/speed.clamp_min(eps),fallback)
    points=points_xy.unsqueeze(0) if points_xy.ndim==2 else points_xy
    forward=(points*direction[:,None,:]).sum(-1)
    lateral=points[...,1]*direction[:,None,0]-points[...,0]*direction[:,None,1]
    return torch.stack((forward,lateral),-1)


class SupportChoiceAttention(nn.Module):
    """forward(policy480, terrain225, course3, estimated_roles4) -> context32, weights225.

    Terrain values are the existing clamp(relative height,-1,1)*5 encoding.
    Saturated values are flagged, but cannot be identified as actual sensor
    unknowns from this encoding alone. Nonfinite heights receive a separate
    unknown flag and a neutral numerical value; they are never called support.
    """
    def __init__(self, direction_bias=1., local_bias=.5, uniform_mass=.02):
        super().__init__()
        if (not all(math.isfinite(x) for x in (direction_bias,local_bias,uniform_mass))
                or direction_bias<0 or local_bias<0 or not 0<uniform_mass<1):
            raise ValueError('Invalid attention bias/floor configuration')
        self.direction_bias=float(direction_bias)
        self.local_bias=float(local_bias)
        self.uniform_mass=float(uniform_mass)
        axis=torch.arange(15,dtype=torch.float32)*.1-.7
        x,y=torch.meshgrid(axis,axis,indexing='ij')
        self.register_buffer('points_xy',torch.stack((x.flatten(),y.flatten()),-1))
        self.query=nn.Sequential(nn.Linear(480+3+4+3,64),nn.ELU(),nn.Linear(64,32))
        self.token=nn.Sequential(nn.Linear(5,32),nn.ELU())
        self.key=nn.Linear(32,32,bias=False)
        self.value=nn.Linear(32,32,bias=False)
        self.output=nn.Sequential(nn.Linear(64,32),nn.ELU())

    def metadata(self):
        return dict(version=1,ablation='optional_command_attention',policy_dimension=480,
            terrain_dimension=225,context_dimension=32,role_dimension=4,
            command_slice=[42,45],grid='body frame, ij 15x15, -0.7:0.1:0.7m',
            role_source='detached estimated MDN probabilities; no measured contact',
            direction_bias=self.direction_bias,local_bias=self.local_bias,
            uniform_mass=self.uniform_mass,world_yaw_input=False,
            terrain_flags='saturation proxy plus observed nonfinite; actual unknown validity unavailable')

    def forward(self,policy,terrain,course_tracking,role_probabilities):
        n=policy.shape[0] if policy.ndim==2 else -1
        if any(t.shape!=(n,width) for t,width in ((policy,480),(terrain,225),(course_tracking,3),(role_probabilities,4))):
            raise ValueError('Expected [N,480], [N,225], [N,3], [N,4] inputs')
        if not torch.isfinite(policy).all() or not torch.isfinite(course_tracking).all():
            raise ValueError('Nonfinite proprioception or course tracking')
        roles=role_probabilities.detach()
        if not torch.isfinite(roles).all() or (roles<0).any():
            raise ValueError('Estimated role probabilities must be finite and nonnegative')
        total=roles.sum(-1,keepdim=True)
        roles=torch.where(total>0,roles/total.clamp_min(1.e-8),torch.full_like(roles,.25))
        command=policy[:,42:45]
        points=command_frame_coordinates(self.points_xy.to(policy),command[:,:2])
        unknown=~torch.isfinite(terrain)
        clipped=(terrain.abs()>=5.)|unknown
        height=torch.nan_to_num(terrain,nan=0.,posinf=5.,neginf=-5.).clamp(-5.,5.)/5.
        features=torch.cat((height[...,None],points,
            clipped.to(policy.dtype)[...,None],unknown.to(policy.dtype)[...,None]),-1)
        tokens=self.token(features)
        query=self.query(torch.cat((policy,course_tracking,roles,command),-1))
        # Bounded learned logits plus a uniform floor retain every map location.
        logits=8*torch.tanh((self.key(tokens)*query[:,None,:]).sum(-1)/(8*math.sqrt(32)))
        moving=command[:,:2].norm(dim=-1,keepdim=True).clamp(0.,.2)/.2
        logits=logits+self.direction_bias*moving*torch.tanh(points[...,0]/.35)
        logits=logits+self.local_bias*torch.exp(-points.square().sum(-1)/(.3**2))
        weights=(1-self.uniform_mass)*logits.softmax(-1)+self.uniform_mass/225
        pooled=(weights[...,None]*self.value(tokens)).sum(1)
        return self.output(torch.cat((query,pooled),-1)),weights
