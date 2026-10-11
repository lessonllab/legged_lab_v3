"""Seeded square-course perturbations and continuous geometric certificates.

All heights stay at zero. Certificates use nominal G1 sole envelopes with a
1 cm margin, not IK, dynamics, swing clearance, or policy success. Witnesses
permit both feet on each stone, rather than asserting a one-foot-per-stone gait.
Level follows diagram_layout's normalized [0,1] convention.
"""
import math
import numpy as np
from beamdojo_square_diagram_playback import diagram_layout, rotation
from beamdojo_feasibility_geometry import foot_polygon, polygons_overlap


PROFILES = {1: (.26, .30, .01, 0.), 2: (.25, .31, .02, 0.),
            3: (.24, .32, .03, .015)}


def _inside(foot, polygon):
    polygon = np.asarray(polygon)
    edges = np.roll(polygon, -1, axis=0)-polygon
    delta = foot[None, :, :]-polygon[:, None, :]
    return bool(np.all(edges[:, None, 0]*delta[:, :, 1]-
                       edges[:, None, 1]*delta[:, :, 0] >= -1e-9))


def certify_layout(course):
    """Return explicit route-facing, alternating, full-sole placement witnesses.

    Preparatory steps on the center and two feet per stone establish geometric
    connectivity even at diagonal entries. No flight trajectory is certified.
    """
    witnesses = []
    polygons = course['support_polygons_m']
    for route in course['routes']:
        angle = route['angle_degrees']; r = rotation(angle)
        yaw = math.radians(angle)
        ids = route['stone_polygon_indices']
        local = [np.asarray(polygons[i]) @ r for i in ids]
        bounds = [(p.min(0), p.max(0)) for p in local]
        for i in range(len(local)-2):
            if polygons_overlap(local[i], local[i+2]):
                raise ValueError('neighboring same-column stones overlap')
        # Ring/corner landing is a convex subset of the preserved support union.
        if course['family'] == 'cardinal':
            landing = np.array([[3,-.7],[4,-.7],[4,.7],[3,.7]]) @ r.T
        else:
            landing = np.array([[2.5,3],[3,2.5],[4,2.5],[4,4],[2.5,4]]) @ rotation(angle-45).T
        feet = np.array([[-.045,.05],[-.045,-.05]])
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
                              support_polygon_m=np.asarray(surface).tolist(),
                              same_foot_displacement_m=own,
                              support_foot_distance_m=support))

        center_t = float((bounds[0][0][1]+bounds[0][1][1])*.5)
        ts = [center_t+.05, center_t-.05]
        central = polygons[0]
        for foot in (0,1):
            move(foot, [.2, ts[foot]], central, 'central_preparation', 0)
        low, high = .2, 1.
        for _ in range(50):
            mid = (low+high)/2
            fits = all(_inside(foot_polygon(np.array([mid,t]) @ r.T, yaw, margin=.01), central) for t in ts)
            if fits: low = mid
            else: high = mid
        prep = low-.001
        for s in np.linspace(.2, prep, max(2, int(math.ceil((prep-.2)/.2))+1))[1:]:
            for foot in (0,1):
                move(foot, [s, ts[foot]], central, 'central_preparation', 0)
        for index, (lo, hi) in zip(ids, bounds):
            t = (lo[1]+hi[1])*.5
            # Rear-expanded edge is 6.5 cm behind the ankle; 1 mm slack.
            s = lo[0]+.066
            for foot in (0,1):
                move(foot, [s, t+(.05 if foot==0 else -.05)], polygons[index], 'stone', index)
        s = route['goal_start_s_m']+.066
        for foot in (0,1):
            move(foot, [s, feet[foot,1]], landing, 'outer_ring')
        witnesses.append(dict(angle_degrees=angle, initial_feet_xy_m=initial,
                              steps=steps, terminal_both_feet_on_ring=True))
    all_steps = [s for w in witnesses for s in w['steps']]
    return dict(status='path_found', routes=witnesses,
                sole_margin_m=.01, max_same_foot_displacement_m=max(s['same_foot_displacement_m'] for s in all_steps),
                max_support_foot_distance_m=max(s['support_foot_distance_m'] for s in all_steps),
                scope='Continuous sole placements only; no IK/dynamics/swing/policy guarantee',
                placement_strategy='central preparation, both feet on each stone, both feet on ring')


def mixed_layout(seed=0, level=1., family='cardinal', variant=0):
    """Regular variant 0 is exactly diagram_layout, including its metadata.

    Variants 1..3 draw independent lengths/widths per stone in two local routes;
    opposite routes reflect the complete patches including their perturbations.
    Rejected samples deterministically retry, then reduce jitter. Stone sizes
    always stay inside their requested ranges. Failure raises, never silently
    returns uncertified randomized geometry.
    """
    if isinstance(variant, bool) or variant not in (0,1,2,3):
        raise ValueError('variant must be 0, 1, 2, or 3')
    base = diagram_layout(seed, level, family)
    if variant == 0:
        return base
    lo, hi, longitudinal, lateral = PROFILES[variant]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) % (2**64), int(variant), int(round(float(level)*1000000)), int(family=='diagonal')]))
    for attempt in range(96):
        jitter_scale = 1. if attempt < 32 else (.5 if attempt < 64 else 0.)
        polygons = [np.asarray(p).copy() for p in base['polygons_m']]
        for route_number in (0,1):
            route = base['routes'][route_number]
            opposite = base['routes'][route_number+2]
            r = rotation(route['angle_degrees'])
            direction = r[:,0]
            reflection = np.eye(2)-2*np.outer(direction,direction)
            for idx, opp in zip(route['stone_polygon_indices'], opposite['stone_polygon_indices']):
                center = (np.asarray(base['polygons_m'][idx]) @ r).mean(0)
                size = rng.uniform(lo, hi, 2)
                center += jitter_scale*np.array([rng.uniform(-longitudinal,longitudinal), rng.uniform(-lateral,lateral)])
                a, b = center-size/2; c, d = center+size/2
                p = np.array([[a,b],[c,b],[c,d],[a,d]]) @ r.T
                polygons[idx] = p
                # Reflection reverses winding; restore CCW for support_mask.
                polygons[opp] = (p @ reflection.T)[::-1]
        course = dict(base)
        course['polygons_m'] = [p.tolist() for p in polygons]
        course['support_polygons_m'] = course['polygons_m']
        course['rectangles_m'] = [np.r_[p.min(0),p.max(0)].tolist() for p in polygons]
        stone_ids = [idx for route in course['routes'] for idx in route['stone_polygon_indices']]
        course['stone_rectangles_m'] = [course['rectangles_m'][i] for i in stone_ids]
        try:
            certificate = certify_layout(course)
        except ValueError:
            continue
        course['schema'] = 'beamdojo_square_mixed_geometry_v1'
        course['variant'] = int(variant)
        course['witness'] = certificate
        course['checks'] = dict(base['checks'], full_envelope_single_surface=True,
                                terminal_both_feet_on_ring=True,
                                max_same_foot_displacement_m=certificate['max_same_foot_displacement_m'],
                                max_support_foot_distance_m=certificate['max_support_foot_distance_m'])
        course['metadata'] = dict(base['metadata'], randomization='seeded individual stone dimensions and center jitter; opposite routes reflected',
                                  variant=int(variant), stone_size_range_m=[lo,hi],
                                  longitudinal_jitter_limit_m=longitudinal*jitter_scale,
                                  lateral_jitter_limit_m=lateral*jitter_scale,
                                  sample_attempt=attempt+1, jitter_scale=jitter_scale,
                                  scope=certificate['scope'])
        for key in ('stone_length_m','stone_width_m','actual_same_column_gap_m','lateral_edge_gap_m'):
            course['metadata'].pop(key,None)
        return course
    raise ValueError('96 deterministic samples failed the geometric certificate')


def grid_mesh(grid):
    """Merge exact 2 cm [y,x] collision cells into boxes; origin is (4,4,0).

    grid must be the 400x400 collision_height_m raster, with only 0/-1 heights.
    This reproduces cell-center raster support, not continuous diagonal edges.
    """
    import trimesh
    grid = np.asarray(grid)
    if grid.shape != (400,400) or not np.isin(grid, [-1.,0.]).all():
        raise ValueError('expected 400x400 grid with only 0 and -1 meter heights')
    grid = grid.T
    floor = trimesh.creation.box(extents=(8,8,.1))
    floor.apply_translation((4,4,-1.05)); meshes = [floor]
    active = {}; spans = []
    for x in range(grid.shape[0]+1):
        runs = []
        if x < grid.shape[0]:
            edges = np.flatnonzero(np.diff(np.r_[False,grid[x]==0,False].astype(np.int8)))
            runs = list(zip(edges[::2],edges[1::2]))
        for run in list(active):
            if run not in runs:
                spans.append((active.pop(run),x,*run))
        for run in runs:
            active.setdefault(run,x)
    for x0,x1,y0,y1 in spans:
        box = trimesh.creation.box(extents=((x1-x0)*.02,(y1-y0)*.02,1.))
        box.apply_translation(((x0+x1)*.01,(y0+y1)*.01,-.5)); meshes.append(box)
    return meshes, np.array([4.,4.,0.])
