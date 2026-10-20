"""Longer, seeded four-route square courses with exact support geometry.

This v2 module never modifies v1 geometry. All top surfaces remain at z=0.
Witnesses certify continuous full-sole placements, including preparatory steps
and occasional two-feet-on-one-stone transitions. They do NOT certify IK,
dynamics, swing clearance, a natural gait, or policy success.
"""
from functools import lru_cache
import math
import numpy as np
from beamdojo_square_diagram_playback import rotation
from beamdojo_square_course_geometry import support_mask
from beamdojo_feasibility_geometry import foot_polygon, polygons_overlap

MAP_HALF = 6.
INNER_HALF = 4.
CENTER_HALF = .7
CORNER_INFILL = .5
RESOLUTION = .02
ANGLES = {'cardinal': (0, 90, 180, -90),
          'diagonal': (45, 135, -135, -45)}
VARIANT_PROFILES = {
    0: dict(name='regular', length=[.28, .28], width=[.28, .28], square=True,
            longitudinal_jitter=0., lateral_jitter=0., rotation_degrees=0.),
    1: dict(name='independent_square_sizes', length=[.25, .31], width=[.25, .31], square=True,
            longitudinal_jitter=0., lateral_jitter=0., rotation_degrees=0.),
    2: dict(name='independent_rectangles', length=[.25, .32], width=[.24, .32], square=False,
            longitudinal_jitter=0., lateral_jitter=0., rotation_degrees=0.),
    3: dict(name='irregular_longitudinal_spacing', length=[.28, .28], width=[.28, .28], square=True,
            longitudinal_jitter=.035, lateral_jitter=0., rotation_degrees=0.),
    4: dict(name='lateral_offsets', length=[.28, .28], width=[.28, .28], square=True,
            longitudinal_jitter=0., lateral_jitter=.025, rotation_degrees=0.),
    5: dict(name='rotated_stones', length=[.28, .28], width=[.28, .28], square=True,
            longitudinal_jitter=0., lateral_jitter=0., rotation_degrees=5.),
    6: dict(name='size_and_spacing', length=[.25, .32], width=[.24, .32], square=False,
            longitudinal_jitter=.025, lateral_jitter=.02, rotation_degrees=0.),
    7: dict(name='combined_size_spacing_rotation', length=[.25, .32], width=[.24, .32], square=False,
            longitudinal_jitter=.025, lateral_jitter=.02, rotation_degrees=5.),
}


def geometry_parameters():
    """JSON-ready construction parameters for experiment contracts."""
    return dict(version=2, schema='beamdojo_square_extended_geometry_v2',
                central_side_m=1.4, outer_side_m=12., inner_side_m=8.,
                ring_width_m=2., corner_landing_m=.5, raster_resolution_m=.02,
                raster_shape=[600, 600], mesh_origin_m=[6., 6., 0.],
                nominal_stone_side_m=.28, nominal_lateral_edge_gap_m=.12,
                same_column_requested_gap_range_m=[.07, .20],
                variants={str(k): dict(v) for k, v in VARIANT_PROFILES.items()},
                perturbation_scope='interior stones; first two and last two stones preserve regular interfaces',
                witness_scope='continuous sole geometry only; no IK/dynamics/gait/policy guarantee')


def _inside(foot, polygon):
    polygon = np.asarray(polygon)
    edges = np.roll(polygon, -1, axis=0)-polygon
    delta = foot[None, :, :]-polygon[:, None, :]
    return bool(np.all(edges[:, None, 0]*delta[:, :, 1]-
                       edges[:, None, 1]*delta[:, :, 0] >= -1e-9))


def _ankle_interval(polygon, lateral):
    """Exact allowed route-forward ankle interval for a fixed lateral position."""
    sole = foot_polygon([0., lateral], 0., margin=.01)
    low, high = -math.inf, math.inf
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b-a
        c = edge[0]*(sole[:, 1]-a[1])-edge[1]*(sole[:, 0]-a[0])
        coefficient = -edge[1]
        if abs(coefficient) < 1e-12:
            if np.min(c) < -1e-9:
                raise ValueError('lateral sole envelope does not fit stone')
        elif coefficient > 0:
            low = max(low, float(np.max(-c/coefficient)))
        else:
            high = min(high, float(np.min(-c/coefficient)))
    if not math.isfinite(low+high) or high-low < .002:
        raise ValueError('no complete-sole placement interval')
    return low+.001, high-.001


def certify_layout(course):
    """Four explicit alternating placement witnesses, not a gait prescription."""
    if course['family'] == 'flat':
        return dict(status='flat_support', routes=[], scope=geometry_parameters()['witness_scope'])
    polygons = [np.asarray(p) for p in course['support_polygons_m']]
    witnesses = []
    for route in course['routes']:
        angle = route['angle_degrees']; r = rotation(angle); yaw = math.radians(angle)
        ids = route['stone_polygon_indices']
        local = [polygons[i] @ r for i in ids]
        for i in range(len(local)-2):
            if polygons_overlap(local[i], local[i+2]):
                raise ValueError('same-column neighboring stones overlap')
        if course['family'] == 'cardinal':
            landing = np.array([[4., -.7], [6., -.7], [6., .7], [4., .7]]) @ r.T
        else:
            landing = np.array([[3.5, 4.], [4., 3.5], [6., 3.5], [6., 6.], [3.5, 6.]]) @ rotation(angle-45).T
        feet = np.array([[-.0260016, .118506455], [-.0260016, -.118506455]])
        initial = (feet @ r.T).tolist(); steps = []

        def move(foot, target, surface, kind, polygon_index=None):
            target = np.asarray(target)
            own = float(np.linalg.norm(target-feet[foot]))
            support = float(np.linalg.norm(target-feet[1-foot]))
            if foot != len(steps) % 2 or own > .600000001 or support > .700000001:
                raise ValueError('placement exceeds geometric reach bound')
            world = target @ r.T
            sole = foot_polygon(world, yaw, margin=.01)
            if not _inside(sole, surface):
                raise ValueError('full margin sole does not fit designated support')
            other = foot_polygon(feet[1-foot] @ r.T, yaw, margin=.01)
            if polygons_overlap(sole, other):
                raise ValueError('sole envelopes overlap')
            feet[foot] = target
            steps.append(dict(foot=foot, ankle_xy=world.tolist(), foot_yaw=yaw,
                              surface_polygon_index=polygon_index, kind=kind,
                              support_polygon_m=surface.tolist(),
                              same_foot_displacement_m=own, support_foot_distance_m=support))

        center_t = float(local[0].mean(0)[1])
        ts = [center_t+.05, center_t-.05]
        central = polygons[0]
        for foot in (0, 1):
            move(foot, [.2, ts[foot]], central, 'central_preparation', 0)
        lo, hi = .2, 1.
        for _ in range(40):
            mid = (lo+hi)/2
            if all(_inside(foot_polygon(np.array([mid, t]) @ r.T, yaw, margin=.01), central) for t in ts):
                lo = mid
            else:
                hi = mid
        for s in np.linspace(.2, lo-.001, max(2, int(math.ceil((lo-.201)/.2))+1))[1:]:
            for foot in (0, 1):
                move(foot, [s, ts[foot]], central, 'central_preparation', 0)
        for index, surface in zip(ids, local):
            t = float(surface.mean(0)[1])
            for foot in (0, 1):
                lateral = t+(.05 if foot == 0 else -.05)
                s, _ = _ankle_interval(surface, lateral)
                move(foot, [s, lateral], polygons[index], 'stone', index)
        # For diagonal exits, the minimum route-forward ring plane is exactly
        # (8-.5)/sqrt(2); the sole lateral strip lies inside the corner infill.
        s = route['goal_start_s_m']+.066
        for foot in (0, 1):
            move(foot, [s, feet[foot, 1]], landing, 'outer_ring')
        witnesses.append(dict(angle_degrees=angle, initial_feet_xy_m=initial,
                              steps=steps, terminal_both_feet_on_ring=True))
    steps = [s for w in witnesses for s in w['steps']]
    return dict(status='path_found', routes=witnesses, sole_margin_m=.01,
                max_same_foot_displacement_m=max(s['same_foot_displacement_m'] for s in steps),
                max_support_foot_distance_m=max(s['support_foot_distance_m'] for s in steps),
                scope=geometry_parameters()['witness_scope'],
                placement_strategy='central preparation and both feet on each stone; not natural-gait proof')


def _validate(seed, level, family, variant):
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError('seed must be an integer')
    if family not in ANGLES:
        raise ValueError('family must be cardinal or diagonal')
    if isinstance(variant, bool) or variant not in VARIANT_PROFILES:
        raise ValueError('variant must be an integer 0..7')
    level = float(level)
    if not math.isfinite(level) or not 0 <= level <= 1:
        raise ValueError('level must be finite in [0,1]')
    return int(seed), level, int(variant)


def extended_layout(seed=0, level=1., family='cardinal', variant=0):
    """JSON-ready 12 m course with deterministic, mirrored seeded variants."""
    seed, level, variant = _validate(seed, level, family, variant)
    center = np.array([[-.7, -.7], [.7, -.7], [.7, .7], [-.7, .7]])
    ring = [[4., -4., 6., 4.], [-6., -4., -4., 4.],
            [-6., 4., 6., 6.], [-6., -6., 6., -4.]]
    fixed = [center] + [np.array([[a, b], [c, b], [c, d], [a, d]]) for a, b, c, d in ring]
    if family == 'diagonal':
        for angle in ANGLES[family]:
            fixed.append(np.array([[3.5, 4.], [4., 3.5], [4., 4.]]) @ rotation(angle-45).T)
        start_s = .7*math.sqrt(2); goal_s = 7.5/math.sqrt(2)
    else:
        start_s = .7; goal_s = 4.
    first, last = start_s+.06, goal_s-.06-.28
    requested_gap = .07+.13*level
    intervals = int(math.floor(2*(last-first)/(.28+requested_gap)+.5))
    phase = (last-first)/intervals
    profile = VARIANT_PROFILES[variant]
    rng = np.random.default_rng(np.random.SeedSequence([seed % (2**64), variant,
                            int(round(level*1000000)), int(family == 'diagonal')]))
    for attempt in range(48):
        scale = 1. if attempt < 24 else .5
        polygons = [p.copy() for p in fixed]
        patches = []
        for number in (0, 1):
            r = rotation(ANGLES[family][number]); route_patches = []
            for i in range(intervals+1):
                t = .2 if i % 2 == 0 else -.2
                center_xy = np.array([first+i*phase+.14, t])
                length = width = .28; angle = 0.
                if variant and 2 <= i < intervals-1:
                    length = rng.uniform(*profile['length'])
                    width = length if profile['square'] else rng.uniform(*profile['width'])
                    center_xy += scale*np.array([rng.uniform(-profile['longitudinal_jitter'], profile['longitudinal_jitter']),
                                                rng.uniform(-profile['lateral_jitter'], profile['lateral_jitter'])])
                    angle = scale*rng.uniform(-profile['rotation_degrees'], profile['rotation_degrees'])
                patch = np.array([[-length/2, -width/2], [length/2, -width/2],
                                  [length/2, width/2], [-length/2, width/2]]) @ rotation(angle).T+center_xy
                route_patches.append(patch @ r.T)
            patches.append(route_patches)
        for number in (0, 1):
            direction = rotation(ANGLES[family][number])[:, 0]
            reflection = np.eye(2)-2*np.outer(direction, direction)
            patches.append([(p @ reflection.T)[::-1] for p in patches[number]])
        routes = []
        for number, (angle, route_patches) in enumerate(zip(ANGLES[family], patches)):
            ids = list(range(len(polygons), len(polygons)+len(route_patches)))
            polygons.extend(route_patches)
            routes.append(dict(angle_degrees=angle, first_column='left' if number < 2 else 'right',
                               stone_polygon_indices=ids, goal_start_s_m=goal_s,
                               stone_count=len(ids)))
        boxes = [np.r_[p.min(0), p.max(0)].tolist() for p in polygons]
        ids = [i for route in routes for i in route['stone_polygon_indices']]
        course = dict(schema='beamdojo_square_extended_geometry_v2', seed=seed, level=level,
                      family=family, variant=variant, rectangles_m=boxes,
                      polygons_m=[p.tolist() for p in polygons], support_polygons_m=[p.tolist() for p in polygons],
                      routes=routes, stone_rectangles_m=[boxes[i] for i in ids],
                      start_platform_m=[-.7, -.7, .7, .7], end_platform_m=ring[0], ring_rectangles_m=ring,
                      metadata=dict(scope=geometry_parameters()['witness_scope'],
                          world_bounds_m=[-6., -6., 6., 6.], body_yaw=0., foot_yaw=0.,
                          branch_count=4, stone_length_m=.28, stone_width_m=.28,
                          lateral_edge_gap_m=.12, longitudinal_phase_m=phase,
                          requested_same_column_gap_m=requested_gap,
                          actual_same_column_gap_m=2*phase-.28,
                          first_entry_gap_m=.06, top_height_m=0., hole_height_m=-1.,
                          central_side_m=1.4, ring_outer_side_m=12., ring_width_m=2.,
                          corner_landing_m=.5 if family == 'diagonal' else 0.,
                          stone_count_per_route=intervals+1, variant_profile=dict(profile),
                          randomization='seeded interior dimensions/spacing/rotation; exact opposite-route reflection',
                          regular_interface_stones_per_end=2, jitter_scale=scale, sample_attempt=attempt+1,
                          terminal='Both feet on selected route outer ring after smooth braking and stable stand',
                          raster_semantics='Cell-center sampling of the exact convex support polygon union'))
        try:
            witness = certify_layout(course)
        except ValueError:
            continue
        course['witness'] = witness
        course['checks'] = dict(full_envelope_single_surface=True, terminal_both_feet_on_ring=True,
                                opposite_first_columns_mirrored=True,
                                max_same_foot_displacement_m=witness['max_same_foot_displacement_m'],
                                max_support_foot_distance_m=witness['max_support_foot_distance_m'])
        return course
    raise ValueError('48 deterministic samples failed continuous geometric connectivity')


def rasterize(course, resolution=RESOLUTION, stage=2):
    """Sample [y,x] cell centers, optimizing polygon tests to each AABB ROI."""
    resolution = float(resolution)
    if not math.isfinite(resolution) or resolution <= 0 or stage not in (1, 2):
        raise ValueError('positive resolution and stage 1 or 2 required')
    n = int(round(12/resolution))
    if n < 1 or not math.isclose(n*resolution, 12., abs_tol=1e-9):
        raise ValueError('resolution must divide the 12m map')
    axis = -6+(np.arange(n)+.5)*resolution
    mask = np.zeros((n, n), dtype=bool)
    polygons = course.get('support_polygons_m', course['polygons_m'])
    for polygon in polygons:
        polygon = np.asarray(polygon, dtype=np.float64)
        lo, hi = polygon.min(0), polygon.max(0)
        x0, y0 = np.searchsorted(axis, lo-1e-10, side='left')
        x1, y1 = np.searchsorted(axis, hi+1e-10, side='right')
        if x1 <= x0 or y1 <= y0:
            continue
        xx = axis[x0:x1][None, :]; yy = axis[y0:y1][:, None]
        inside = np.ones((y1-y0, x1-x0), dtype=bool)
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = b-a
            inside &= edge[0]*(yy-a[1])-edge[1]*(xx-a[0]) >= -1e-10
        mask[y0:y1, x0:x1] |= inside
    height = np.where(mask, 0., -1.).astype(np.float32)
    return dict(x_centers_m=axis, y_centers_m=axis, resolution_m=resolution,
                reward_support_mask=mask, observed_height_m=height,
                collision_height_m=height.copy() if stage == 2 else np.zeros_like(height),
                stage=stage, indexing='[y,x] cell centers', virtual_collision_floor=stage == 1)


@lru_cache(maxsize=1024)
def tile(seed, family, variant, level):
    """Cached profile-compatible (course, support[x,y]) for integer levels 0..10."""
    if isinstance(level, bool) or int(level) != level or not 0 <= level <= 10:
        raise ValueError('tile level must be integer 0..10')
    if family in (2, 'flat'):
        if variant != 0:
            raise ValueError('flat variant must be zero')
        course = dict(schema='beamdojo_square_extended_geometry_v2', seed=int(seed), level=0., family='flat',
                      variant=0, rectangles_m=[[-6., -6., 6., 6.]], routes=[],
                      polygons_m=[[[-6., -6.], [6., -6.], [6., 6.], [-6., 6.]]],
                      metadata=dict(world_bounds_m=[-6., -6., 6., 6.], top_height_m=0.))
        mask = np.ones((600, 600), dtype=bool)
    else:
        name = {0: 'cardinal', 1: 'diagonal', 'cardinal': 'cardinal', 'diagonal': 'diagonal'}.get(family)
        course = extended_layout(seed, level/10, name, variant)
        mask = rasterize(course)['reward_support_mask'].T.copy()
    mask.setflags(write=False)
    return course, mask


cached_tile = tile
mixed_layout = extended_layout


def grid_mesh(grid):
    """Exact cell-union collision boxes; local terrain origin is (6,6,0)."""
    import trimesh
    grid = np.asarray(grid)
    if grid.shape != (600, 600) or not np.isin(grid, [-1., 0.]).all():
        raise ValueError('expected 600x600 grid containing only 0/-1 heights')
    grid = grid.T
    floor = trimesh.creation.box(extents=(12., 12., .1))
    floor.apply_translation((6., 6., -1.05)); meshes = [floor]
    active = {}; spans = []
    for x in range(601):
        runs = []
        if x < 600:
            edges = np.flatnonzero(np.diff(np.r_[False, grid[x] == 0, False].astype(np.int8)))
            runs = list(zip(edges[::2], edges[1::2]))
        for run in list(active):
            if run not in runs:
                spans.append((active.pop(run), x, *run))
        for run in runs:
            active.setdefault(run, x)
    for x0, x1, y0, y1 in spans:
        box = trimesh.creation.box(extents=((x1-x0)*.02, (y1-y0)*.02, 1.))
        box.apply_translation(((x0+x1)*.01, (y0+y1)*.01, -.5)); meshes.append(box)
    return meshes, np.array([6., 6., 0.])


@lru_cache(maxsize=128)
def mesh(seed, family, variant, level):
    _, mask = tile(seed, family, variant, level)
    return grid_mesh(np.where(mask.T, 0., -1.).astype(np.float32))


def terrain_mesh(difficulty, cfg):
    meshes, origin = mesh(cfg.layout_seed, cfg.mixed_family, cfg.mixed_variant, min(10, int(difficulty*11)))
    return meshes, origin.copy()
