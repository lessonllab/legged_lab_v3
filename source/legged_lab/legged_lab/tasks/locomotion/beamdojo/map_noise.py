"""Optional conservative noisy-ideal-map training helper, not LiDAR completion.

Defaults below are adaptation starting values, NOT the BeamDojo paper values.
Paper Table IX instead reports offset/noise ±.03 m, roll/pitch endpoint ±.03 m,
yaw ±.2 rad, extension .6 and repeat .2. Call only for training ideal task maps.
Invalid cells remain invalid; extension requires caller-provided safe foothold
labels and cannot manufacture returns for a deployment sensor's missing cells.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MapNoiseCfg:
    height_bias: float = .01
    point_noise: float = .01
    yaw_episode: float = .02
    gradient_endpoint: float = .02
    repeat_probability: float = .05
    extension_probability: float = .05


class IdealMapNoise:
    """Reset-aware four-family map noise; inputs absolute heights [N,size²].

    Yaw bias stays constant in each episode. Noise is applied when a fresh map
    is published, before ElevationMapHold; don't call at physics frequency.
    safe_foothold[N,size²] is a task label, distinct from sensor valid mask.
    Repetition keeps the complete preceding map/mask; first map never repeats.
    """
    def __init__(self, num_envs, device='cpu', size=15, cfg=MapNoiseCfg(), generator=None):
        if min(cfg.height_bias,cfg.point_noise,cfg.yaw_episode,cfg.gradient_endpoint) < 0:
            raise ValueError('noise amplitudes must be nonnegative')
        if not 0 <= cfg.repeat_probability <= 1 or not 0 <= cfg.extension_probability <= 1:
            raise ValueError('probabilities must lie in [0,1]')
        self.cfg,self.size,self.generator=cfg,size,generator
        self.yaw=torch.empty(num_envs,device=device)
        self.bias=torch.empty(num_envs,device=device)
        self.previous=torch.zeros(num_envs,size*size,device=device)
        self.previous_valid=torch.zeros_like(self.previous,dtype=torch.bool)
        self.ready=torch.zeros(num_envs,device=device,dtype=torch.bool)
        self.last_repeated=torch.zeros_like(self.ready)
        self.reset()

    def _rand(self, shape):
        return torch.rand(shape,device=self.yaw.device,generator=self.generator)

    def reset(self, env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self.yaw[ids]=(2*self._rand(self.yaw[ids].shape)-1)*self.cfg.yaw_episode
        self.bias[ids]=(2*self._rand(self.bias[ids].shape)-1)*self.cfg.height_bias
        self.previous[ids]=0.
        self.previous_valid[ids]=False
        self.ready[ids]=False
        self.last_repeated[ids]=False

    def __call__(self, heights, valid, safe_foothold=None, env_ids=None, rotate_map=True):
        ids = (torch.arange(len(self.ready),device=self.yaw.device) if env_ids is None
               else torch.as_tensor(env_ids,device=self.yaw.device,dtype=torch.long))
        if heights.shape != self.previous[ids].shape or valid.shape != heights.shape:
            raise ValueError('map shape does not match noise state')
        n=len(heights); s=self.size
        valid=valid.bool() & torch.isfinite(heights)
        image=torch.where(valid,heights,0.).reshape(n,1,s,s)
        mask=valid.reshape(n,1,s,s)
        # Image rows represent local x; columns local y, so convert yaw to row/col axes.
        yaw = self.yaw[ids] if rotate_map else torch.zeros_like(self.yaw[ids])
        c,sn=yaw.cos(),yaw.sin()
        affine=torch.zeros(n,2,3,device=heights.device,dtype=heights.dtype)
        affine[:,0,0]=c; affine[:,0,1]=sn
        affine[:,1,0]=-sn; affine[:,1,1]=c
        grid=F.affine_grid(affine,image.shape,align_corners=True)
        sample=lambda a:F.grid_sample(a.float(),grid,mode='nearest',padding_mode='zeros',align_corners=True)
        image=sample(image); mask=sample(mask).bool()
        if safe_foothold is not None:
            if safe_foothold.shape != heights.shape:
                raise ValueError('safe foothold labels must match map shape')
            safe=sample(safe_foothold.reshape(n,1,self.size,self.size).bool() & valid.reshape(n,1,self.size,self.size)).bool() & mask
            neighbor=F.max_pool2d(torch.where(safe,image,-torch.inf),3,stride=1,padding=1)
            extend=(~safe)&mask&torch.isfinite(neighbor)&(self._rand(image.shape)<self.cfg.extension_probability)
            image=torch.where(extend,neighbor,image)
        axis=torch.linspace(-1.,1.,self.size,device=heights.device,dtype=heights.dtype)
        slopes=(2*self._rand((n,2))-1)*self.cfg.gradient_endpoint
        gradient=slopes[:,0,None,None]*axis[None,:,None]+slopes[:,1,None,None]*axis[None,None,:]
        bias=self.bias[ids,None,None,None]
        noise=(2*self._rand(image.shape)-1)*self.cfg.point_noise
        image=image+bias+noise+gradient[:,None]
        output=torch.where(mask,image,0.).flatten(1)
        mask=mask.flatten(1)
        repeat=self.ready[ids] & (self._rand((n,))<self.cfg.repeat_probability)
        output=torch.where(repeat[:,None],self.previous[ids],output)
        mask=torch.where(repeat[:,None],self.previous_valid[ids],mask)
        self.previous[ids]=output; self.previous_valid[ids]=mask; self.ready[ids]=True
        self.last_repeated[ids]=repeat
        return output,mask
