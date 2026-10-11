"""CPU-only +X square course, continuous foot-envelope witness, not dynamics.

All AABBs, polygons, point queries and raster samples come from the same top
rectangles. Stage 1 retains an explicitly virtual collision floor at z=0 while
observations/reward geometry contain holes; stage 2 has physical holes to z=-1.
No simulator is imported. Neither raster center sampling nor this geometric
witness proves IK, dynamic stability, swing clearance or policy success.
"""
import math
import numpy as np

try:
    from .beamdojo_feasibility_geometry import foot_supported, foot_polygon, polygons_overlap
except ImportError:
    from beamdojo_feasibility_geometry import foot_supported, foot_polygon, polygons_overlap


def layout(seed=0, level=1.0):
    """JSON-ready regular course. level 0..1 requests progressively larger gaps.

    Endpoint fitting quantizes the actual gap: a fixed first/last stone and a
    uniform alternating chain avoid a long last step. Seed is recorded but does
    not randomize this first implementation. Coordinates are body/world aligned;
    no global-yaw variant or random initial displacement is applied.
    """
    level = float(level)
    if not math.isfinite(level) or not 0 <= level <= 1:
        raise ValueError('level must be finite in [0,1]')
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError('seed must be an integer')
    seed = int(seed)
    start = [-.7, -.7, .7, .7]
    ring = [[3., -3., 4., 3.], [-4., -3., -3., 3.],
            [-4., 3., 4., 4.], [-4., -4., 4., -3.]]
    first, last, side = .76, 2.66, .28
    requested_gap = .07 + .13 * level
    intervals = int(math.floor(2 * (last-first) / (side+requested_gap) + .5))
    phase = (last-first) / intervals
    pitch = 2 * phase
    stones, feet_of_stones = [], []
    for i in range(intervals+1):
        x = first + i*phase
        foot = i % 2
        ys = (.06, .34) if foot == 0 else (-.34, -.06)
        stones.append([x, ys[0], x+side, ys[1]])
        feet_of_stones.append(foot)
    rects = [start] + ring + stones
    feet = np.array([[-.0260016, .118506455], [-.0260016, -.118506455]])
    initial = feet.copy()
    states = [dict(feet_xy_m=feet.tolist(), next_swing_foot=0)]
    steps = []

    def move(foot, x, surface_index, kind):
        target = np.array([x, initial[foot, 1]])
        own = float(np.linalg.norm(target-feet[foot]))
        support = float(np.linalg.norm(target-feet[1-foot]))
        if foot != len(steps) % 2 or own > .600000001 or support > .700000001:
            raise ValueError('constructed alternating witness violates reach limits')
        if not foot_supported([rects[surface_index]], target, 0., margin=.01):
            raise ValueError('complete foot does not fit designated single surface')
        feet[foot] = target
        if not all(foot_supported(rects, p, 0., margin=.01) for p in feet):
            raise ValueError('witness lost continuous support')
        if polygons_overlap(*(foot_polygon(p, 0., margin=.01) for p in feet)):
            raise ValueError('witness foot envelopes overlap')
        steps.append(dict(foot=foot, ankle_xy=target.tolist(), foot_yaw=0.,
                          surface_rectangle_index=surface_index, kind=kind,
                          same_foot_displacement_m=own, support_foot_distance_m=support))
        states.append(dict(feet_xy_m=feet.tolist(), next_swing_foot=1-foot))

    # Four genuine preparation steps within the central platform, not a leap
    # from root origin straight to the first stone.
    for x in (.25, .50):
        move(0, x, 0, 'central_preparation')
        move(1, x, 0, 'central_preparation')
    for i, rect in enumerate(stones):
        move(i % 2, rect[0]+.095, 5+i, 'stone')
    # The foot trailing on the penultimate stone moves first; 1mm interior
    # margin beyond expanded rear sole edge at x=3.
    next_foot = len(steps) % 2
    move(next_foot, 3.066, 1, 'outer_ring')
    move(1-next_foot, 3.066, 1, 'outer_ring')
    move(next_foot, 3.25, 1, 'terminal_stance')
    move(1-next_foot, 3.25, 1, 'terminal_stance')
    polygons = [[[a,b],[c,b],[c,d],[a,d]] for a,b,c,d in rects]
    gaps = [[stones[j][0]-stones[i][2] for i,j in zip(ids, ids[1:])]
            for ids in ([i for i in range(len(stones)) if i % 2 == foot] for foot in (0,1))]
    witness = dict(status='path_found', initial_feet_xy_m=initial.tolist(),
                   reference_origin=[0.,0.], yaw=0., states=states, steps=steps,
                   limits=dict(max_same_foot_displacement_m=.6,
                               max_support_foot_distance_m=.7, sole_margin_m=.01),
                   total_swing_travel_m=sum(s['same_foot_displacement_m'] for s in steps))
    return dict(schema='beamdojo_square_course_geometry_v1', seed=seed, level=level,
                rectangles_m=rects, stone_rectangles_m=stones, polygons_m=polygons,
                stone_feet=feet_of_stones, start_platform_m=start,
                end_platform_m=ring[0], ring_rectangles_m=ring,
                initial_feet_xy_m=initial.tolist(), first_entry_direction=[1.,0.],
                witness=witness,
                checks=dict(full_envelope_single_surface=True, alternating_feet=True,
                            foot_yaw_rad=0., max_same_foot_displacement_m=max(s['same_foot_displacement_m'] for s in steps),
                            max_support_foot_distance_m=max(s['support_foot_distance_m'] for s in steps),
                            terminal_both_feet_on_ring=True),
                metadata=dict(scope='Continuous sole-envelope geometry only; NOT IK/dynamics/speed guarantee',
                    world_bounds_m=[-4.,-4.,4.,4.], body_yaw=0., foot_yaw=0.,
                    command_body_xy=[1.,0.], branch_count=1, randomization='none; seed reserved',
                    stone_length_m=side, stone_width_m=side, lateral_edge_gap_m=.12,
                    requested_same_column_gap_m=requested_gap,
                    actual_same_column_gap_m=pitch-side, actual_gap_scale=(pitch-side)/.20,
                    actual_same_column_gaps_m=gaps, longitudinal_phase_m=phase,
                    first_entry_gap_m=.06, other_column_entry_gap_m=.06+phase,
                    exit_edge_gaps_m=[3-max(r[2] for i,r in enumerate(stones) if i%2==f) for f in (0,1)],
                    ankle_lateral_separation_m=float(initial[0,1]-initial[1,1]),
                    top_height_m=0., hole_height_m=-1., central_side_m=1.4,
                    ring_outer_side_m=8., ring_width_m=1.,
                    terminal='Both complete foot envelopes on +X ring; root x alone is insufficient',
                    raster_semantics='Cell-center samples; exact continuous geometry remains rectangles_m'))


def support_mask(course, xy):
    """Exact point membership in the top-surface union (not full-foot support)."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[-1:] != (2,) or not np.isfinite(xy).all():
        raise ValueError('xy must be finite [...,2]')
    mask = np.zeros(xy.shape[:-1], dtype=bool)
    if 'support_polygons_m' in course:
        # Playback-only diagram maps include rotated squares and triangular
        # corner landings. Convex CCW polygons retain their actual boundaries.
        for polygon in course['support_polygons_m']:
            polygon = np.asarray(polygon, dtype=np.float64)
            inside = np.ones(xy.shape[:-1], dtype=bool)
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                edge = b-a
                inside &= edge[0]*(xy[...,1]-a[1])-edge[1]*(xy[...,0]-a[0]) >= -1e-10
            mask |= inside
        return mask
    for xmin,ymin,xmax,ymax in course['rectangles_m']:
        mask |= ((xy[...,0]>=xmin)&(xy[...,0]<=xmax)&
                 (xy[...,1]>=ymin)&(xy[...,1]<=ymax))
    return mask


def rasterize(course, resolution=.02, stage=2):
    """Sample [y,x] cell centers inside [-4,4]^2; never extend beyond map.

    observed_height_m and reward_support_mask always encode the real intended
    gaps. collision_height_m alone changes for virtual stage 1. Native consumers
    must explicitly select the correct field, not infer collision from observations.
    """
    resolution = float(resolution)
    if not math.isfinite(resolution) or resolution <= 0 or stage not in (1,2):
        raise ValueError('positive resolution and stage 1 or 2 required')
    n = int(round(8/resolution))
    if n < 1 or not math.isclose(n*resolution, 8., abs_tol=1e-9):
        raise ValueError('resolution must divide the 8m map')
    axis = -4 + (np.arange(n)+.5)*resolution
    xx, yy = np.meshgrid(axis, axis, indexing='xy')
    mask = support_mask(course, np.stack([xx,yy], -1))
    height = np.where(mask, 0., -1.).astype(np.float32)
    return dict(x_centers_m=axis, y_centers_m=axis, resolution_m=resolution,
                observed_height_m=height, reward_support_mask=mask,
                collision_height_m=height.copy() if stage==2 else np.zeros_like(height),
                stage=stage, indexing='[y,x] cell centers', virtual_collision_floor=stage==1)
