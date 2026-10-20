"""Opt-in, process-local two-column geometry for frozen-policy diagnostics.

Call the installer before constructing an environment or its cached geometry.
No source layout, training configuration, checkpoint, or policy is modified.
"""
from functools import wraps

MAP_SIZE = (32.0, 32.0)
SCHEDULE = 'smooth_top_v1'
_PATCH_ATTRIBUTE = '_beamdojo_two_column_corridor_originals'


def _terrain(module=None):
    if module is None:
        from legged_lab.tasks.locomotion.beamdojo import upstream_terrain
        return upstream_terrain
    return module


def _matches(module, stage, size, schedule):
    return (stage == 1 and tuple(size) == MAP_SIZE
            and module.course_schedule.profile_name(schedule) == SCHEDULE)


def _source(module):
    originals = getattr(module, _PATCH_ATTRIBUTE, None)
    return originals[0] if originals else module.rectangles


def _selection(module, level, seed=5):
    source = _source(module)(1, level, seed, MAP_SIZE, SCHEDULE)
    # The upstream stage-1 contract is grid stones, four borders, platform.
    stones, platform = source[:-5], source[-1]
    columns = sorted({(r[1], r[3]) for r in stones})
    if len(columns) < 2:
        raise ValueError('Two-column diagnostic requires at least two source columns')
    midpoint_pixel = MAP_SIZE[1] / module.RESOLUTION / 2
    pair = min(zip(columns, columns[1:]), key=lambda p: (
        abs(sum(sum(column) for column in p) / 4 - midpoint_pixel), p))
    retained = [r for r in stones if (r[1], r[3]) in pair]
    return retained, platform, pair


def corridor_metadata(level, terrain_module=None, seed=5):
    """Describe retained source coordinates, including the centered spawn y."""
    module = _terrain(terrain_module)
    stones, platform, columns = _selection(module, level, seed)
    resolution = module.RESOLUTION
    centers = [(lo + hi) * resolution / 2 - MAP_SIZE[1] / 2
               for lo, hi in columns]
    y_bounds = [columns[0][0] * resolution - MAP_SIZE[1] / 2,
                columns[-1][1] * resolution - MAP_SIZE[1] / 2]
    supported = stones + [platform]
    x_bounds = [min(r[0] for r in supported) * resolution - MAP_SIZE[0] / 2,
                max(r[2] for r in supported) * resolution - MAP_SIZE[0] / 2]
    side = (stones[0][2] - stones[0][0]) * resolution
    pitch = (columns[1][0] - columns[0][0]) * resolution
    return {
        'mode': 'two_column_corridor', 'level': int(level),
        'source_layout_stage': 1, 'schedule': SCHEDULE, 'map_size_m': list(MAP_SIZE),
        'resolution_m': resolution, 'stone_size_m': [side, side],
        'pitch_m': pitch, 'gap_m': pitch - side,
        'source_y_columns_pixels': [list(column) for column in columns],
        'column_centers_y_m': centers, 'recommended_spawn_y_m': sum(centers) / 2,
        'corridor_y_bounds_m': y_bounds, 'corridor_width_m': y_bounds[1] - y_bounds[0],
        'supported_x_bounds_m': x_bounds, 'stone_count': len(stones),
        'platform_source_pixels': list(platform),
        'platform_size_m': [(platform[2] - platform[0]) * resolution,
                            (platform[3] - platform[1]) * resolution],
        'platform_x_bounds_m': [platform[i] * resolution - MAP_SIZE[0] / 2
                                for i in (0, 2)],
        'platform_y_bounds_m': [platform[i] * resolution - MAP_SIZE[1] / 2
                                for i in (1, 3)],
        'stone_top_z_m': 0.0, 'pit_floor_z_m': -1.0,
        'raised_map_borders': False,
        'lateral_void_outside_corridor': True,
        'lateral_void_exception': 'Unmodified central 1 m start platform',
        'source_grid_phase_preserved': True,
    }


def install_two_column_corridor(terrain_module=None):
    """Install once in this process; unrelated layouts retain their originals.

    Heightfields feed both real collision meshes and surface-height queries.
    Display also uses the patched rectangle lookup, but needs a matching adapter
    because its stock implementation assumes four raised border rectangles.
    """
    module = _terrain(terrain_module)
    if hasattr(module, _PATCH_ATTRIBUTE):
        return module
    original_rectangles = module.rectangles
    original_display = module.display_rectangles
    setattr(module, _PATCH_ATTRIBUTE, (original_rectangles, original_display))

    @wraps(original_rectangles)
    def rectangles(stage, level, seed=5, size=(module.SIZE, module.SIZE), schedule='legacy_9'):
        if not _matches(module, stage, size, schedule):
            return original_rectangles(stage, level, seed, size, schedule)
        stones, platform, _ = _selection(module, level, seed)
        return stones + [platform]

    @wraps(original_display)
    def display_rectangles(stage, level, seed=5, size=(module.SIZE, module.SIZE), schedule='legacy_9'):
        if not _matches(module, stage, size, schedule):
            return original_display(stage, level, seed, size, schedule)
        pixels = module.rectangles(stage, level, seed, size, schedule)
        def convert(rect):
            x0, y0, x1, y1 = rect
            return ((x0 + x1) * module.RESOLUTION / 2 - size[0] / 2,
                    (y0 + y1) * module.RESOLUTION / 2 - size[1] / 2,
                    (x1 - x0) * module.RESOLUTION,
                    (y1 - y0) * module.RESOLUTION, 0.0)
        return [convert(pixels[-1])] + [convert(rect) for rect in pixels[:-1]]

    module.rectangles = rectangles
    module.display_rectangles = display_rectangles
    return module
