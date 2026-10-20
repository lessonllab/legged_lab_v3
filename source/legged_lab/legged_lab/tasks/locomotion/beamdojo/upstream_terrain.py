# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


"""Native Sim6 port of ezoicoder terrain.py stones_everywhere{,_stage1}.

Source 9a47b7dfe85fdcc737c3c8963d80d05997376575. Pixel rounding and the
actual zero stone heights are retained (upstream's random heights are unused).
Stage 2 uses stepping_stones, the actual XBotLStoneCfg index 9.
"""
import math
import numpy as np
import torch
if __package__:
    from . import course_schedule
else:  # Retain standalone, simulator-free geometry inspection.
    import importlib.util
    from pathlib import Path
    _schedule_spec = importlib.util.spec_from_file_location(
        'beamdojo_course_schedule_math', Path(__file__).with_name('course_schedule.py'))
    course_schedule = importlib.util.module_from_spec(_schedule_spec)
    _schedule_spec.loader.exec_module(course_schedule)

RESOLUTION = .02
SIZE = 8.5
PIXELS = int(SIZE / RESOLUTION)
COURSE_BOUNDARY = SIZE / 2 - .75
COURSE_STAGE2_BOUNDARY = SIZE / 2 - .1
COURSE_MIN_DISTANCE = 3.
COURSE_MAX_ERROR = .05
COURSE_MIN_CONTACT_SECONDS = 1.  # summed contacting foot-seconds
COURSE_STAND_SECONDS = 1.
COURSE_MAX_LINEAR_SPEED = .15
COURSE_MAX_ANGULAR_SPEED = .3


def enabled(cfg):
    """Terrain reuse does not opt the Unitree profile into XBot motion code."""
    return (getattr(cfg, 'beamdojo_upstream_aligned', False)
            or getattr(cfg, 'beamdojo_unitree_motion_aligned', False))


def course_size(cfg):
    """World x/y footprint; legacy callers retain the exact 8.5m raster."""
    if getattr(cfg, 'beamdojo_physical_aligned', False):
        side = float(getattr(cfg, 'beamdojo_stage1_map_size', 32.))
        return (side, side)
    if not getattr(cfg, 'beamdojo_course_integrity', False):
        return (SIZE, SIZE)
    if cfg.beamdojo_stage == 1:
        side = float(getattr(cfg, 'beamdojo_stage1_map_size', 32.))
        return (side, side)
    return (float(getattr(cfg, 'beamdojo_stage2_course_length', 16.)), SIZE)


def layout_stage(cfg):
    """Aligned physical training materializes the existing virtual layout."""
    return 1 if getattr(cfg, 'beamdojo_physical_aligned', False) else cfg.beamdojo_stage


def rectangles(stage, level, seed=5, size=(SIZE, SIZE), schedule='legacy_9'):
    """Integer pixel rectangles; default geometry matches upstream exactly."""
    nx, ny = (int(v / RESOLUTION) for v in size)
    border = int(.25 / RESOLUTION)
    if stage == 2 and course_schedule.profile_name(schedule) != 'legacy_9':
        raise ValueError('Smooth schedule requires the shared Stage 1 raster layout')
    if stage == 2:
        rng = np.random.RandomState(seed + level)
        side = int((.8,.65,.5,.4,.35,.3,.25,.2,.2)[level]/RESOLUTION)
        gap = int((.1+.05*level)/RESOLUTION)
        mid = ny // 2
        platform_width = int(1.5/RESOLUTION)
        platform_length = int(1./RESOLUTION)
        end = nx - border - platform_length
        # Raw transverse/forward coordinates are transposed to world x/y below.
        raw = [(0,0,ny,border),(0,nx-border,ny,nx),
               (0,0,border,nx),(ny-border,0,ny,nx),
               (mid-platform_width//2,border,mid+platform_width//2,border+platform_length),
               (mid-platform_width//2,end,mid+platform_width//2,end+platform_length)]
        offset = int(.4/RESOLUTION)
        for center in (mid-offset, mid+offset):
            y = border + platform_length
            while y < end:
                rng.uniform(-.05,.05)
                low = side//2-offset
                high = max(low,offset-side//2)
                shift = rng.randint(low,high+1)
                raw.append((max(mid-50,center-side//2+shift), y,
                            min(mid+50,center+side//2+shift), min(y+side,end)))
                y += side + gap
        return [(y0,x0,y1,x1) for x0,y0,x1,y1 in raw]
    side, gap = course_schedule.stone_pixels(level, schedule)
    pitch = side + gap
    effective_x, effective_y = (int((v-.5)/RESOLUTION) for v in size)
    mid_x, mid_y = nx//2, ny//2
    start_x = mid_x-effective_x//2+(effective_x-(effective_x//pitch)*pitch)//2
    start_y = mid_y-effective_y//2+(effective_y-(effective_y//pitch)*pitch)//2
    rects = [(start_x+i*pitch,start_y+j*pitch,start_x+i*pitch+side,start_y+j*pitch+side)
             for i in range(effective_x//pitch) for j in range(effective_y//pitch)]
    rects += [(0,0,nx,border),(0,ny-border,nx,ny),
              (0,0,border,ny),(nx-border,0,nx,ny)]
    platform = int((1. if stage == 1 else 4.) / RESOLUTION)
    rects += [(mid_x-platform//2,mid_y-platform//2,mid_x+platform//2,mid_y+platform//2)]
    return rects


def heightfield(stage, level, seed=5, size=(SIZE, SIZE), schedule='legacy_9'):
    grid = np.full(tuple(int(v/RESOLUTION) for v in size), -1., dtype=np.float32)
    for x0,y0,x1,y1 in rectangles(stage, level, seed, size, schedule):
        grid[x0:x1,y0:y1] = 0.
    return grid


def mesh_terrain(difficulty, cfg):
    import trimesh
    count = course_schedule.level_count(cfg)
    level = min(count - 1, int(difficulty * count))
    # Each raster cell is constant over its 2cm footprint. Merge adjacent cells
    # into boxes, preserving the exact raster support boundary and 1m pit floor.
    size = tuple(cfg.size)
    grid = heightfield(getattr(cfg, 'layout_stage', 2), level, cfg.layout_seed, size,
                       course_schedule.profile_name(cfg))
    meshes = []
    floor = trimesh.creation.box(extents=(*size,.1))
    floor.apply_translation((size[0]/2,size[1]/2,-1.05))
    meshes.append(floor)
    active = {}
    spans = []
    for x in range(grid.shape[0]+1):
        runs=[]
        if x < grid.shape[0]:
            row=grid[x] == 0
            edges=np.flatnonzero(np.diff(np.r_[False,row,False].astype(np.int8)))
            runs=list(zip(edges[::2],edges[1::2]))
        for run in list(active):
            if run not in runs:
                spans.append((active.pop(run),x,*run))
        for run in runs:
            active.setdefault(run,x)
    for x0,x1,y0,y1 in spans:
        box=trimesh.creation.box(extents=((x1-x0)*RESOLUTION,(y1-y0)*RESOLUTION,1.))
        box.apply_translation(((x0+x1)*RESOLUTION/2,(y0+y1)*RESOLUTION/2,-.5))
        meshes.append(box)
    # A grid can contain thousands of stones. One mesh per tile keeps the USD
    # prim/collision-shape count bounded without changing a single box boundary.
    if getattr(cfg, 'layout_stage', 2) == 1:
        meshes = [trimesh.util.concatenate(meshes)]
    return meshes, np.array([size[0]/2,size[1]/2,0.])


def configure(cfg):
    import isaaclab.terrains as terrain_gen
    from isaaclab.utils.configclass import configclass
    @configclass
    class StonesCfg(terrain_gen.MeshPlaneTerrainCfg):
        function = mesh_terrain
        layout_seed: int = 5
        layout_stage: int = 2
        beamdojo_course_schedule: str = 'legacy_9'
    terrain=cfg.scene.terrain
    size = course_size(cfg)
    schedule = course_schedule.profile_name(cfg)
    if schedule != 'legacy_9' and layout_stage(cfg) != 1:
        raise ValueError('Smooth schedule requires the shared Stage 1 raster layout')
    cfg.scene.env_spacing=max(size)
    terrain.max_init_terrain_level=4
    terrain.visual_material=None
    if cfg.beamdojo_stage == 1 or getattr(cfg, 'beamdojo_virtual_course', False):
        terrain.terrain_type='plane'
        terrain.terrain_generator=None
    else:
        terrain.terrain_type='generator'
        terrain.terrain_generator=terrain_gen.TerrainGeneratorCfg(
            seed=cfg.seed,size=size,border_width=5.,num_rows=course_schedule.level_count(cfg),num_cols=4,
            curriculum=True,use_cache=False,difficulty_range=(0.,1.),
            sub_terrains={'stepping_stones':StonesCfg(proportion=1.,layout_seed=cfg.seed,
                                                   layout_stage=layout_stage(cfg),
                                                   beamdojo_course_schedule=schedule)})


def ensure_state(env):
    if hasattr(env,'_upstream_heightfields'):
        return
    source_stage = layout_stage(env.cfg)
    count = course_schedule.level_count(env.cfg)
    schedule = course_schedule.profile_name(env.cfg)
    env._upstream_heightfields=torch.tensor(np.stack([heightfield(source_stage,i,env.cfg.seed,course_size(env.cfg),schedule) for i in range(count)]),device=env.device)
    if not getattr(env.cfg,'beamdojo_curriculum',True):
        env.beamdojo_levels=torch.full((env.num_envs,),env.cfg.beamdojo_initial_level,device=env.device,dtype=torch.long)
    elif env.cfg.beamdojo_stage==2 and not getattr(env.cfg, 'beamdojo_virtual_course', False):
        env.beamdojo_levels=env.scene.terrain.terrain_levels.clone()
    else:
        env.beamdojo_levels=torch.randint(0,5,(env.num_envs,),device=env.device)
    env.beamdojo_success_streak=torch.zeros_like(env.beamdojo_levels)
    if getattr(env.cfg, 'beamdojo_support_fixed_levels', False):
        from . import support_experiment_runtime
        support_experiment_runtime.assign_levels(env)
    if getattr(env.cfg, 'beamdojo_corridor_mixed_levels', False):
        from . import corridor_mixed_runtime
        corridor_mixed_runtime.assign_levels(env)
    env.beamdojo_qualified_success=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    env._upstream_started=torch.zeros_like(env.beamdojo_qualified_success)
    layouts=[display_rectangles(source_stage,i,env.cfg.seed,course_size(env.cfg),schedule) for i in range(count)]
    if source_stage == 1 and env.num_envs > 64 and getattr(env.cfg, 'beamdojo_course_integrity', False):
        layouts = [layout[:1] for layout in layouts]
    width=max(len(rects) for rects in layouts)
    env._upstream_display=torch.zeros((count,width,5),device=env.device)
    env._upstream_display_mask=torch.zeros((count,width),device=env.device,dtype=torch.bool)
    env._upstream_stone_mask=torch.zeros_like(env._upstream_display_mask)
    for level,rects in enumerate(layouts):
        n=len(rects)
        env._upstream_display[level,:n]=torch.tensor(rects,device=env.device)
        env._upstream_display_mask[level,:n]=True
        env._upstream_stone_mask[level,1:n-1 if source_stage==2 else n]=True
    _refresh_display(env)


def display_rectangles(stage, level, seed=5, size=(SIZE, SIZE), schedule='legacy_9'):
    """Display rectangles in world-forward x, with start first/end last.

    Borders are intentionally excluded from stone visit metrics; physical
    collision/query surfaces retain the full source raster including borders.
    """
    pixels=rectangles(stage,level,seed,size,schedule)
    def convert(r):
        x0,y0,x1,y1=r
        return ((x0+x1)*RESOLUTION/2-size[0]/2,(y0+y1)*RESOLUTION/2-size[1]/2,
                (x1-x0)*RESOLUTION,(y1-y0)*RESOLUTION,0.)
    if stage==1:
        return [convert(pixels[-1])]+[convert(r) for r in pixels[:-5]]
    return [convert(pixels[4])]+[convert(r) for r in pixels[6:]]+[convert(pixels[5])]


def _refresh_display(env):
    levels=env.beamdojo_levels
    env.beamdojo_geometry=env._upstream_display[levels]
    env.beamdojo_rectangle_mask=env._upstream_display_mask[levels]
    env.beamdojo_stone_mask=env._upstream_stone_mask[levels]
    first_platform=env.beamdojo_geometry[:,0]
    last=env.beamdojo_rectangle_mask.sum(-1)-1 if layout_stage(env.cfg)==2 else torch.zeros_like(levels)
    last_platform=env.beamdojo_geometry[torch.arange(env.num_envs,device=env.device),last]
    direction = getattr(env, 'beamdojo_course_direction', torch.ones(env.num_envs, device=env.device))
    reverse = direction < 0
    env.beamdojo_start_platform=torch.where(reverse[:, None], last_platform, first_platform)
    env.beamdojo_end_platform=torch.where(reverse[:, None], first_platform, last_platform)


def refresh_geometry(env):
    # Curriculum restore/reset can change the query geometry within one tick.
    env._upstream_heightmap_cache = None
    ensure_state(env)
    _refresh_display(env)
    if env.cfg.beamdojo_stage==2 and not getattr(env.cfg, 'beamdojo_virtual_course', False):
        terrain=env.scene.terrain
        terrain.terrain_levels[:]=env.beamdojo_levels
        terrain.env_origins[:]=terrain.terrain_origins[env.beamdojo_levels,terrain.terrain_types]
        env.scene.env_origins[:]=terrain.env_origins


def surface_height(env, points, *, exact=None):
    ensure_state(env)
    origins=env.scene.env_origins
    origins=origins if isinstance(origins,torch.Tensor) else origins.torch
    shape=points.shape[:-1]
    local=(points-origins.reshape((env.num_envs,)+(1,)*(points.ndim-2)+(3,))).reshape(env.num_envs,-1,3)
    size = torch.as_tensor(course_size(env.cfg), device=points.device, dtype=points.dtype)
    inside = torch.isfinite(local[..., :2]).all(-1) & (local[..., :2] >= -size/2).all(-1) & (local[..., :2] < size/2).all(-1)
    # Clamping is only for safe indexing; outside the tile is never support.
    pixels=((local[...,:2]+size/2)/RESOLUTION).long().clamp_min(0)
    if exact is None:
        exact = (getattr(env.cfg, 'beamdojo_physical_aligned', False)
                 and not getattr(env.cfg, 'beamdojo_match_virtual_query', False))
    pixel_max = torch.as_tensor(env._upstream_heightfields.shape[1:], device=points.device) - (1 if exact else 2)
    pixels = torch.minimum(pixels, pixel_max)
    x,y=pixels.unbind(-1)
    level=env.beamdojo_levels[:,None]
    grid=env._upstream_heightfields
    if exact:
        # Constant-height raster cells exactly match the merged-box collision
        # mesh. Three-neighbour erosion would falsely mark a 2cm stone rim void.
        heights=grid[level,x,y]
    else:
        heights=torch.minimum(torch.minimum(grid[level,x,y],grid[level,x+1,y]),grid[level,x,y+1])
    heights = torch.where(inside, heights, -torch.ones_like(heights))
    return (heights+origins[:,2:3]).reshape(shape), (inside & (heights>=-.1)).reshape(shape)


def update_success(env):
    if getattr(env.cfg, 'beamdojo_course_integrity', False):
        _update_course(env)
        if getattr(env.cfg, 'beamdojo_corridor_training', False):
            from . import corridor_runtime
            env.beamdojo_qualified_success &= corridor_runtime.qualified_mask(env)
        return
    ensure_state(env)
    data=env.scene['robot'].data.root_pos_w
    data=data if isinstance(data,torch.Tensor) else data.torch
    origins=env.scene.env_origins
    origins=origins if isinstance(origins,torch.Tensor) else origins.torch
    relative=data[:,:2]-origins[:,:2]
    success=relative.norm(dim=-1)>3.75 if env.cfg.beamdojo_stage==1 else relative[:,0]>1.
    env.beamdojo_qualified_success |= success


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def _ensure_course(env):
    ensure_state(env)
    if hasattr(env, 'beamdojo_course_distance'):
        return
    for name in ('distance', 'contact_samples', 'unsafe_samples', 'stand_time'):
        setattr(env, 'beamdojo_course_' + name, torch.zeros(env.num_envs, device=env.device))
    env.beamdojo_course_failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._course_initialized = torch.zeros_like(env.beamdojo_course_failed)
    env._course_last_xy = torch.zeros((env.num_envs, 2), device=env.device)
    env._course_start_xy = torch.zeros_like(env._course_last_xy)
    env._course_start_yaw = torch.zeros(env.num_envs, device=env.device)
    env._course_initial_direction = torch.ones(env.num_envs, device=env.device)
    env.beamdojo_course_signed_progress = torch.zeros(env.num_envs, device=env.device)
    env.beamdojo_course_net_displacement = torch.zeros(env.num_envs, device=env.device)
    env.beamdojo_course_valid_contact_seconds = torch.zeros(env.num_envs, device=env.device)
    env._course_boundary = torch.zeros_like(env.beamdojo_course_failed)
    env._course_finished = torch.zeros_like(env.beamdojo_course_failed)
    env._course_milestone = torch.zeros_like(env.beamdojo_course_failed)
    env._course_step = None


def _failure_limits(env):
    terms = getattr(env.cfg, 'terminations', None)
    height = getattr(getattr(terms, 'base_height', None), 'params', {})
    tilt = getattr(getattr(terms, 'bad_orientation', None), 'params', {})
    return height.get('minimum_height', .2), tilt.get('limit_angle', .8)


def _update_course(env):
    """One contact-conditioned observation per control frame, without rewards.

    Legacy qualification requires full sole support. Aligned physical traversal
    permits partial support but requires measured contact on a real stone top;
    virtual overlap or pit-floor contact alone cannot accumulate valid travel.
    """
    _ensure_course(env)
    step = env.common_step_counter
    if env._course_step == step:
        return
    env._course_step = step
    from .geometry import g1_sole_points, transform_sole_points
    robot = env.scene['robot']
    sensor = env.scene.sensors['contact_forces']
    if not hasattr(env, '_course_foot_query'):
        names = ['left_ankle_roll_link', 'right_ankle_roll_link']
        env._course_foot_query = (robot.find_bodies(names, preserve_order=True)[0],
                                  sensor.find_bodies(names, preserve_order=True)[0],
                                  g1_sole_points(device=env.device))
    feet, contacts, points = env._course_foot_query
    data = robot.data
    origins = _tensor(env.scene.env_origins)
    root = _tensor(data.root_pos_w) - origins
    soles = transform_sole_points(_tensor(data.body_pos_w)[:, feet],
        _tensor(data.body_quat_w)[:, feet][..., [3, 0, 1, 2]], points)
    surface, safe = surface_height(env, soles)
    contact = _tensor(sensor.data.net_forces_w)[:, contacts, 2] > 5.
    if getattr(env.cfg, 'beamdojo_corridor_boundary_version', 1) == 2:
        from . import corridor_runtime
        env._course_boundary[:] = corridor_runtime.boundary_mask(env, root)
    else:
        margin = .75 if env.cfg.beamdojo_stage == 1 else .1
        boundary_limit = torch.as_tensor(course_size(env.cfg), device=env.device) / 2 - margin
        env._course_boundary[:] = (root[:, :2].abs() >= boundary_limit).any(-1)
    # Match evaluation: the boundary frame ends the measured traversal.
    measured_contact = contact & ~env._course_boundary[:, None]
    env.beamdojo_course_contact_samples += measured_contact.sum(-1) * points.shape[0]
    env.beamdojo_course_unsafe_samples += ((~safe) & measured_contact[..., None]).sum((-1, -2))
    minimum_height, limit_angle = _failure_limits(env)
    gravity = _tensor(data.projected_gravity_b)
    failed = ((root[:, 2] < minimum_height)
              | (-gravity[:, 2] < math.cos(limit_angle))
              | ~torch.isfinite(root).all(-1) | ~torch.isfinite(gravity).all(-1))
    env.beamdojo_course_failed |= failed
    aligned = getattr(env.cfg, 'beamdojo_physical_aligned', False)
    if aligned:
        # Direction is frozen at episode initialization, not changed by a later
        # command resample. Native Sim6 root quaternion storage is XYZW.
        x, y, z, w = _tensor(data.root_quat_w).unbind(-1)
        yaw = torch.atan2(2. * (w * z + x * y), 1. - 2. * (y.square() + z.square()))
        direction = getattr(env, 'beamdojo_course_direction', torch.ones_like(yaw))
        env._course_start_yaw[:] = torch.where(env._course_initialized, env._course_start_yaw, yaw)
        env._course_initial_direction[:] = torch.where(
            env._course_initialized, env._course_initial_direction, direction)
    env._course_start_xy[:] = torch.where(env._course_initialized[:, None], env._course_start_xy, root[:, :2])
    displacement = (root[:, :2] - env._course_last_xy).norm(dim=-1)
    if aligned:
        # Physical support follows collision cells, independently of the
        # conservative virtual query used by rewards and contact-error counts.
        surface, physical_safe = surface_height(env, soles, exact=True)
        tolerance = float(getattr(env.cfg, 'beamdojo_physical_surface_tolerance', .06))
        near_top = (soles[..., 2] - surface).abs() <= tolerance
        top_contact = contact & (physical_safe & near_top).any(-1)
        supported_contact = top_contact & ~env._course_boundary[:, None]
    else:
        supported_contact = measured_contact & safe.all(-1)
    valid_motion = env._course_initialized & supported_contact.any(-1) & ~env.beamdojo_course_failed
    env.beamdojo_course_distance += torch.where(valid_motion, displacement, 0.)
    env._course_last_xy[:] = root[:, :2]
    env._course_initialized[:] = True
    dt = env.step_dt
    if aligned:
        env.beamdojo_course_valid_contact_seconds += (
            supported_contact.sum(-1) * dt * (~env.beamdojo_course_failed))
        relative = root[:, :2] - env._course_start_xy
        env.beamdojo_course_net_displacement[:] = relative.norm(dim=-1)
        env.beamdojo_course_signed_progress[:] = (
            relative[:, 0] * torch.cos(env._course_start_yaw)
            + relative[:, 1] * torch.sin(env._course_start_yaw)) * env._course_initial_direction
        minimum_distance = float(getattr(env.cfg, 'beamdojo_physical_min_distance', COURSE_MIN_DISTANCE))
        continuous = getattr(env.cfg, 'beamdojo_physical_continuous', False)
        progress = (env.beamdojo_course_net_displacement if continuous
                    else env.beamdojo_course_signed_progress)
        travel_qualified = (
            (env.beamdojo_course_distance >= minimum_distance)
            & (progress >= minimum_distance)
            & (env.beamdojo_course_valid_contact_seconds >= COURSE_MIN_CONTACT_SECONDS)
            & ~env.beamdojo_course_failed)
        if (getattr(env.cfg, 'beamdojo_strict_foothold', False)
                and getattr(env.cfg, 'beamdojo_curriculum', False)):
            # Match virtual-course contact accuracy. This is an episode
            # qualification threshold, not a reward deadband: every unsafe
            # contacting sample is penalized by the strict foothold reward.
            contact_error = (env.beamdojo_course_unsafe_samples
                             / env.beamdojo_course_contact_samples.clamp_min(1))
            travel_qualified &= contact_error <= COURSE_MAX_ERROR
        env._course_milestone[:] = (
            travel_qualified & supported_contact.any(-1) & ~env._course_boundary)
        if continuous:
            # Three metres is a measurement milestone, not an episode end.
            # A timeout or map-boundary reset can qualify, but only with actual
            # stone-top support in that final frame. Boundary frames still add
            # no contact time or supported path. No earlier milestone is latched
            # across a fall, loss of support, or retreat below three metres.
            timeout = getattr(env, 'episode_length_buf', torch.zeros(env.num_envs, device=env.device)) >= getattr(env, 'max_episode_length', float('inf'))
            env._course_finished[:] = False
            env.beamdojo_qualified_success[:] = (
                (timeout | env._course_boundary) & travel_qualified & top_contact.any(-1))
        else:
            env._course_finished[:] = env._course_milestone
            env.beamdojo_qualified_success[:] = env._course_finished
        return
    enough = env.beamdojo_course_contact_samples * dt >= COURSE_MIN_CONTACT_SECONDS * points.shape[0]
    error = env.beamdojo_course_unsafe_samples / env.beamdojo_course_contact_samples.clamp_min(1)
    quality = enough & (error <= COURSE_MAX_ERROR) & ~env.beamdojo_course_failed
    env._course_finished[:] = False
    if env.cfg.beamdojo_stage == 2:
        end = env.beamdojo_end_platform
        local = soles[..., :2] - origins[:, None, None, :2]
        on_end = ((local - end[:, None, None, :2]).abs() <= end[:, None, None, 2:4] / 2).all((-1, -2, -3))
        slow = (_tensor(data.root_lin_vel_w).norm(dim=-1) <= COURSE_MAX_LINEAR_SPEED) & (_tensor(data.root_ang_vel_w).norm(dim=-1) <= COURSE_MAX_ANGULAR_SPEED)
        standing = on_end & contact.all(-1) & safe.all((-1, -2)) & slow & ~failed
        env.beamdojo_course_stand_time[:] = torch.where(standing, env.beamdojo_course_stand_time + dt, 0.)
        env._course_finished[:] = (env.beamdojo_course_stand_time >= COURSE_STAND_SECONDS - 1.e-6) & ~env.beamdojo_course_failed & ~env._course_boundary
    net_distance = (root[:, :2] - env._course_start_xy).norm(dim=-1)
    timeout = getattr(env, 'episode_length_buf', torch.zeros(env.num_envs, device=env.device)) >= getattr(env, 'max_episode_length', float('inf'))
    terminal = env._course_boundary | timeout
    # No success latch: falling after reaching a target cannot retain a pass.
    env.beamdojo_qualified_success[:] = quality & (env._course_finished if env.cfg.beamdojo_stage == 2 else
        terminal & (env.beamdojo_course_distance >= COURSE_MIN_DISTANCE) & (net_distance >= COURSE_MIN_DISTANCE))


def course_boundary(env):
    _update_course(env)
    return env._course_boundary & ~env.beamdojo_course_failed


def course_finished(env):
    _update_course(env)
    return env._course_finished


def course_diagnostics(env):
    _update_course(env)
    return {
        'Metrics/course_valid_distance': env.beamdojo_course_distance.mean(),
        'Metrics/course_contact_error': (env.beamdojo_course_unsafe_samples / env.beamdojo_course_contact_samples.clamp_min(1)).mean(),
        'Metrics/course_contact_samples': env.beamdojo_course_contact_samples.mean(),
        'Metrics/course_stand_time': env.beamdojo_course_stand_time.mean(),
        'Metrics/course_signed_progress': env.beamdojo_course_signed_progress.mean(),
        'Metrics/course_net_displacement': env.beamdojo_course_net_displacement.mean(),
        'Metrics/course_valid_contact_seconds': env.beamdojo_course_valid_contact_seconds.mean(),
        'Metrics/course_milestone_fraction': env._course_milestone.float().mean(),
    }


def before_reset(env, ids):
    ensure_state(env)
    experienced=env._upstream_started[ids]
    wins=env.beamdojo_qualified_success[ids]
    skip=getattr(env,'_upstream_skip_curriculum_once',torch.zeros_like(env._upstream_started))[ids]
    eligible=experienced & ~skip
    streak=torch.where(wins,env.beamdojo_success_streak[ids]+1,0)
    streak=torch.where(eligible,streak,env.beamdojo_success_streak[ids])
    if getattr(env.cfg, 'beamdojo_corridor_mixed_levels', False):
        from . import corridor_mixed_runtime
        corridor_mixed_runtime.before_reset(env, ids)
    elif getattr(env.cfg, 'beamdojo_support_fixed_levels', False):
        from . import support_experiment_runtime
        support_experiment_runtime.before_reset(env, ids)
    elif getattr(env.cfg, 'beamdojo_recovery', False) and env.cfg.beamdojo_curriculum:
        from . import recovery_curriculum
        wins = recovery_curriculum.before_reset(env, ids, wins, eligible)
    elif env.cfg.beamdojo_curriculum:
        promote=(streak>=3)&eligible
        levels=env.beamdojo_levels[ids]+promote.long()
        count = course_schedule.level_count(env.cfg)
        env.beamdojo_levels[ids]=torch.where(levels>=count,torch.randint_like(levels,count),levels)
        env.beamdojo_success_streak[ids]=torch.where(promote,0,streak)
    refresh_geometry(env)
    # Avoid a GPU scalar->Python condition and boolean-index nonzero.
    rate=(wins & experienced).float().sum() / experienced.sum().clamp_min(1)
    if getattr(env.cfg, 'beamdojo_course_integrity', False):
        _ensure_course(env)
        for name in ('distance', 'contact_samples', 'unsafe_samples', 'stand_time', 'failed'):
            getattr(env, 'beamdojo_course_' + name)[ids] = 0
        env._course_initialized[ids] = False
        env._course_start_xy[ids] = 0.
        env._course_start_yaw[ids] = 0.
        env._course_initial_direction[ids] = 1.
        env.beamdojo_course_signed_progress[ids] = 0.
        env.beamdojo_course_net_displacement[ids] = 0.
        env.beamdojo_course_valid_contact_seconds[ids] = 0.
        env._course_boundary[ids] = False
        env._course_finished[ids] = False
        env._course_milestone[ids] = False
        # Keep the step guard: reset observations must not sample stale contacts.
    env.beamdojo_qualified_success[ids]=False
    env._upstream_started[ids]=True
    if hasattr(env,'_upstream_skip_curriculum_once'):
        env._upstream_skip_curriculum_once[ids]=False
    return {'Metrics/course_success_rate':rate}
