"""Stage-2 terrain-mesh ray LiDAR surrogate and static-world point accumulation.

This is a regular angular grid, NOT Livox's non-repetitive scan simulation.
PhysX rays target only /World/ground: there is no robot self-occlusion, moving
object mapping, measured odometry, motion distortion or FAST-LIO. World poses
are simulator poses. Coverage/missing cells remain measured by rays, never
filled with ideal terrain heights. Stage 1 must use the separate ideal/noisy
training map, since its physical ground intentionally differs from task terrain.
"""
import math
import torch
from .perception import sparse_points_to_heightmap
from .sensor_spec import MID360_SPEC


def _rpy_quat(rpy):
    r, p, y = (float(a)/2 for a in rpy)
    cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return (cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy)


def _compose_wxyz(a,b):
    w,x,y,z=a; v,i,j,k=b
    return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j, w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)


def make_lidar_cfg(mount_to_scan_rpy=None, horizontal_resolution=2.8125, channels=16,
                   stage=2, prim_path='{ENV_REGEX_NS}/Robot/torso_link'):
    """Create native IsaacLab 6 RayCasterCfg (2048 rays by default).

    Caller-supplied scan RPY is radians relative to mid360_link, an explicit
    experimental hypothesis until calibrated. None is rejected. The mount
    translation is taken from our local USD, not a different robot revision.
    """
    if stage != 2:
        raise ValueError('Physical terrain ray LiDAR is supported only in Stage 2')
    if mount_to_scan_rpy is None:
        raise ValueError('G1 mount-to-scan transform is unverified; supply explicit experimental RPY')
    if len(mount_to_scan_rpy) != 3 or not all(math.isfinite(float(v)) for v in mount_to_scan_rpy):
        raise ValueError('mount_to_scan_rpy must contain three finite radians')
    if channels < 1 or horizontal_resolution <= 0 or not math.isfinite(horizontal_resolution):
        raise ValueError('invalid ray pattern dimensions')
    rays = channels * math.ceil(360/horizontal_resolution)
    if not 1024 <= rays <= 4096:
        raise ValueError('Choose a low-cost surrogate with 1024 to 4096 rays')
    # Lazy imports keep point-map mathematics usable without a running Simulator.
    from isaaclab.sensors import RayCasterCfg, patterns
    w,x,y,z = _compose_wxyz(_rpy_quat(MID360_SPEC['mount_rpy_rad']), _rpy_quat(mount_to_scan_rpy))
    return RayCasterCfg(
        prim_path=prim_path, mesh_prim_paths=['/World/ground'], update_period=.1,
        offset=RayCasterCfg.OffsetCfg(pos=MID360_SPEC['local_usd_mount_xyz_m'], rot=(x,y,z,w)),
        ray_alignment='base', max_distance=40.,
        pattern_cfg=patterns.LidarPatternCfg(channels=channels, vertical_fov_range=(-7.,52.),
                                             horizontal_fov_range=(-180.,180.), horizontal_res=horizontal_resolution),
        debug_vis=False,
    )


def ray_hits_world(sensor):
    """Return native sensor [N,R,3] world hits and finite/range-valid mask.

    Accessing sensor.data follows native lazy sensor updates. Caller is
    responsible for adding a frame to the cache only once per new scan.
    """
    data = sensor.data
    points = data.ray_hits_w
    points = points.torch if hasattr(points, 'torch') else points
    # Native Sim6 stores the exact, offset-adjusted ray origins separately.
    # Use them rather than the tracked parent/frame position for blind-zone tests.
    if hasattr(sensor, '_ray_starts_w'):
        import warp as wp
        origins = wp.to_torch(sensor._ray_starts_w)
    else:
        origin = data.pos_w
        origin = origin.torch if hasattr(origin, 'torch') else origin
        origins = origin[:,None,:]
    distance = torch.linalg.vector_norm(points-origins, dim=-1)
    valid = torch.isfinite(points).all(-1) & (distance >= .1) & (distance <= 40.)
    return points, valid


class WorldPointCache:
    """Bounded per-environment static-world history, reprojected into current yaw.

    tick(dt) advances every environment; add() receives new frames only at
    scan_period cadence, reset() clears episode history immediately. FIFO storage
    is sized to TTL/scan_period; more frequent additions shorten effective TTL.
    Memory scales as N*ceil(TTL/period)*rays; use only small coverage runs until
    measured, rather than deploying this raw-point cache to thousands of envs.
    """
    def __init__(self, num_envs, rays_per_scan, device='cpu', ttl=3., scan_period=.1,
                 dtype=torch.float32):
        if not math.isfinite(ttl) or not math.isfinite(scan_period) or ttl <= 0 or scan_period <= 0:
            raise ValueError('ttl and scan_period must be finite positive seconds')
        self.ttl = ttl
        self.capacity = math.ceil(ttl/scan_period)+1
        self.points = torch.zeros((num_envs,self.capacity,rays_per_scan,3),device=device,dtype=dtype)
        self.valid = torch.zeros(self.points.shape[:-1],device=device,dtype=torch.bool)
        self.age = torch.full((num_envs,self.capacity),torch.inf,device=device,dtype=dtype)
        self.cursor = torch.zeros(num_envs,device=device,dtype=torch.long)

    def tick(self, dt):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError('dt must be finite and nonnegative')
        self.age += dt

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.valid[ids] = False
        self.points[ids] = 0.
        self.age[ids] = torch.inf
        self.cursor[ids] = 0

    def add(self, points_world, valid, env_ids=None):
        ids = (torch.arange(len(self.cursor),device=self.cursor.device) if env_ids is None
               else torch.as_tensor(env_ids,device=self.cursor.device,dtype=torch.long))
        expected = (len(ids),self.points.shape[2],3)
        if points_world.shape != expected or valid.shape != expected[:-1]:
            raise ValueError('scan dimensions do not match cache and selected environments')
        slot = self.cursor[ids]
        mask = valid.bool() & torch.isfinite(points_world).all(-1)
        self.points[ids,slot] = torch.where(mask[...,None],points_world,0.)
        self.valid[ids,slot] = mask
        self.age[ids,slot] = 0.
        self.cursor[ids] = (slot+1)%self.capacity

    def reproject(self, base_pos, yaw, size=15, resolution=.1):
        mask = self.valid & (self.age <= self.ttl)[...,None]
        return sparse_points_to_heightmap(self.points.flatten(1,2),base_pos,yaw,
                                          mask.flatten(1,2),size=size,resolution=resolution)


def lidar_observation(env):
    """451 ray-only elevation/mask/age values, with 10 Hz map publications.

    Experimental Stage 2 coverage interface. Unknown cells stay unknown.
    A held observation includes the pose of its publication, like the ideal
    training map. World poses used for point accumulation are simulator truth.
    """
    from .perception import ElevationMapHold
    from ..unitree_official.mdp import tensor
    data = env.scene['robot'].data
    pos, q = tensor(data.root_pos_w), tensor(data.root_quat_w)
    yaw = torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),
                      1-2*(q[:,1].square()+q[:,2].square()))
    if not hasattr(env, '_beamdojo_lidar_cache'):
        points, valid = ray_hits_world(env.scene.sensors['beamdojo_lidar'])
        env._beamdojo_lidar_cache = WorldPointCache(env.num_envs, points.shape[1], env.device)
        env._beamdojo_lidar_hold = ElevationMapHold(env.num_envs, env.device)
        env._beamdojo_lidar_tick = -1
        env._beamdojo_lidar_clock_tick = -1
    tick = env.common_step_counter
    if tick != env._beamdojo_lidar_tick:
        hold = env._beamdojo_lidar_hold
        cache = env._beamdojo_lidar_cache
        dt = env.step_dt if env._beamdojo_lidar_clock_tick != tick else 0.
        # The first scan after reset is immediate, then one scan each .1 s.
        due = hold.due | (hold._elapsed + dt >= hold.period-1e-7)
        ids = due.nonzero().flatten()
        cache.tick(dt)
        if ids.numel():
            points, valid = ray_hits_world(env.scene.sensors['beamdojo_lidar'])
            cache.add(points[ids], valid[ids], ids)
        heights, valid = cache.reproject(pos, yaw)
        relative = (pos[:,2:3]-heights-.78).clamp(-1.,1.)
        hold.update(relative, valid, dt)
        env._beamdojo_lidar_tick = tick
        env._beamdojo_lidar_clock_tick = tick
    hold = env._beamdojo_lidar_hold
    return torch.cat((hold.heights, hold.valid.to(pos.dtype), hold.age[:,None]), -1)
