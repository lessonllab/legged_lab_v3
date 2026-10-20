"""Versioned, simulator-free schedules for the shared virtual/physical raster."""
import math

RESOLUTION_M = .02


def profile_name(cfg_or_name=None):
    name = cfg_or_name if isinstance(cfg_or_name, str) else getattr(
        cfg_or_name, 'beamdojo_course_schedule', 'legacy_9')
    if name not in ('legacy_9', 'smooth_top_v1'):
        raise ValueError(f'Unknown BeamDojo course schedule: {name}')
    return name


def level_count(cfg_or_name=None):
    return 11 if profile_name(cfg_or_name) == 'smooth_top_v1' else 9


def max_level(cfg_or_name=None):
    return level_count(cfg_or_name) - 1


def stone_pixels(level, cfg_or_name=None):
    """Exact raster dimensions; preserve legacy floating-point truncation."""
    name = profile_name(cfg_or_name)
    if isinstance(level, bool) or int(level) != level or not 0 <= level <= max_level(name):
        raise ValueError(f'Invalid level {level} for {name}')
    if name == 'smooth_top_v1' and level >= 8:
        return (19, 17, 14)[int(level) - 8], 10
    return (int(max(.25, 1.5 * (1 - .1 * level)) / RESOLUTION_M),
            int((.05 * math.ceil(level / 2)) / RESOLUTION_M))


def profile_spec(cfg_or_name=None):
    name = profile_name(cfg_or_name)
    dimensions = [stone_pixels(i, name) for i in range(level_count(name))]
    return {'name': name, 'level_count': level_count(name), 'max_level': max_level(name),
            'resolution_m': RESOLUTION_M,
            'stone_side_m': [s * RESOLUTION_M for s, _ in dimensions],
            'gap_m': [g * RESOLUTION_M for _, g in dimensions],
            'layout_source_stage': 1}
