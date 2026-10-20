"""Pure Torch elevation interfaces, independent of Isaac and sensor transport.

Ideal terrain queries are training observations, not simulated LiDAR returns.
Point clouds must be transformed into world coordinates by the caller; this
module does not invent the unverified G1 scan-frame transform or fill occlusions.
Maps are x-major (meshgrid indexing='ij'), flattened, with absolute world z.
"""
import math
import torch


def grid_points_world(base_pos, yaw, size=15, resolution=0.1):
    """Yaw-aligned [N,size*size,3] sampling locations; z equals base z."""
    if size < 1 or resolution <= 0:
        raise ValueError('size and resolution must be positive')
    axis = (torch.arange(size, device=base_pos.device, dtype=base_pos.dtype) - (size - 1) / 2) * resolution
    x, y = torch.meshgrid(axis, axis, indexing='ij')
    x, y = x.flatten()[None], y.flatten()[None]
    c, s = yaw.cos()[:, None], yaw.sin()[:, None]
    offsets = torch.stack((c*x-s*y, s*x+c*y, torch.zeros_like(c*x)), -1)
    return base_pos[:, None] + offsets


def sparse_points_to_heightmap(points, base_pos, yaw, point_valid=None, size=15, resolution=0.1):
    """Bin world [N,P,3] points; highest finite z per cell, invalid zero + mask.

    Cells are centered on the same grid as grid_points_world, with half-open
    outer bounds. Input should already exclude robot returns and non-ground
    objects if the intended output is a ground elevation map. Highest-point
    aggregation alone cannot distinguish obstacles, overhangs and terrain.
    """
    if size < 1 or resolution <= 0:
        raise ValueError('size and resolution must be positive')
    rel = points - base_pos[:, None]
    c, s = yaw.cos()[:, None], yaw.sin()[:, None]
    xy = torch.stack((c*rel[..., 0]+s*rel[..., 1], -s*rel[..., 0]+c*rel[..., 1]), -1)
    finite = torch.isfinite(points).all(-1) & torch.isfinite(xy).all(-1)
    index = torch.floor(torch.where(finite[..., None], xy, 0.) / resolution + (size-1)/2 + 0.5).long()
    valid = finite & (index >= 0).all(-1) & (index < size).all(-1)
    if point_valid is not None:
        valid &= point_valid.bool()
    flat = (index[..., 0]*size + index[..., 1]).clamp(0, size*size-1)
    z = torch.where(valid, points[..., 2], -torch.inf)
    result = points.new_full((points.shape[0], size*size), -torch.inf)
    result.scatter_reduce_(1, flat, z, reduce='amax', include_self=True)
    mask = torch.isfinite(result)
    return torch.where(mask, result, 0.), mask


class ElevationMapHold:
    """Per-environment 10 Hz sample/hold with explicit reset and age in seconds.

    update receives candidate maps on every call; dt advances selected envs.
    First update after construction/reset publishes immediately. Subsequent
    samples publish on period boundaries. Held maps include their original
    pose implicitly: do not reinterpret them as freshly measured at a new pose.
    """
    def __init__(self, num_envs, device='cpu', period=0.1, size=15, dtype=torch.float32):
        if not math.isfinite(period) or period <= 0:
            raise ValueError('period must be finite and positive')
        self.period = period
        self.heights = torch.zeros((num_envs, size*size), device=device, dtype=dtype)
        self.valid = torch.zeros_like(self.heights, dtype=torch.bool)
        self.age = torch.full((num_envs,), torch.inf, device=device, dtype=dtype)
        self._elapsed = torch.zeros_like(self.age)
        self._fresh = torch.ones(num_envs, device=device, dtype=torch.bool)

    @property
    def due(self):
        return self._fresh | (self._elapsed >= self.period - 1e-7)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.heights[ids] = 0.
        self.valid[ids] = False
        self.age[ids] = torch.inf
        self._elapsed[ids] = 0.
        self._fresh[ids] = True

    def update(self, heights, valid, dt, env_ids=None, repeat=None):
        """Return published env IDs. Candidates have len(env_ids) rows if given."""
        if not math.isfinite(dt) or dt < 0:
            raise ValueError('dt must be finite and nonnegative')
        ids = (torch.arange(len(self.age), device=self.age.device) if env_ids is None
               else torch.as_tensor(env_ids, device=self.age.device, dtype=torch.long))
        if heights.shape != self.heights[ids].shape or valid.shape != heights.shape:
            raise ValueError('candidate maps must match selected environments and grid')
        self.age[ids] += dt
        self._elapsed[ids] += dt
        publish = self.due[ids]
        dst = ids[publish]
        repeated = torch.zeros_like(publish) if repeat is None else repeat.to(self.age.device).bool()
        if repeated.shape != publish.shape:
            raise ValueError('repeat must match selected environments')
        # A delayed publication consumes this scan slot but retains the whole
        # preceding observation and its real age. Fresh reset maps never repeat.
        fresh = publish & (~repeated | self._fresh[ids])
        fresh_dst = ids[fresh]
        mask = valid[fresh].bool() & torch.isfinite(heights[fresh])
        self.heights[fresh_dst] = torch.where(mask, heights[fresh], 0.)
        self.valid[fresh_dst] = mask
        self.age[fresh_dst] = 0.
        # Initial samples start their own publication clock, independently of reset time.
        remainder = torch.remainder(self._elapsed[dst].clamp_min(self.period), self.period)
        self._elapsed[dst] = torch.where(self._fresh[dst], 0., remainder)
        self._fresh[dst] = False
        return fresh_dst
