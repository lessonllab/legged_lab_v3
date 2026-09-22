"""AME alternate-column generator, copied unchanged for geometry parity.
Source: /home/ljc/AME_Locomotion/source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/terrains/loco_hf_terrains.py
Function SHA256: 44031545f6bea9b8e7597f51dc5075dd41a78b934f3ab914677dcf08bd4a9434
"""
from __future__ import annotations
import numpy as np
from isaaclab.terrains.height_field.utils import height_field_to_mesh

@height_field_to_mesh
def alternate_column_stakes_terrain(
    difficulty: float, cfg: loco_hf_terrains_cfg.HfDoubleColumnStakesTerrainCfg
) -> np.ndarray:
    """Generate alternating double-column stake terrain along x/y directions."""

    # Interpolate parameters by difficulty
    stake_side = cfg.stake_side_range[1] - difficulty * (
        cfg.stake_side_range[1] - cfg.stake_side_range[0]
    )
    stake_gap = cfg.stake_gap_range[0] + difficulty * (
        cfg.stake_gap_range[1] - cfg.stake_gap_range[0]
    )
    column_gap = cfg.column_gap_range[1] - difficulty * (
        cfg.column_gap_range[1] - cfg.column_gap_range[0]
    )

    # Discretized grid parameters
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    stake_side_px = max(1, int(stake_side / cfg.horizontal_scale))
    stake_gap_px = max(0, int(stake_gap / cfg.horizontal_scale))
    column_gap_px = max(0, int(column_gap / cfg.horizontal_scale))
    column_jitter_px = max(0, int(cfg.column_jitter / cfg.horizontal_scale))

    stake_height_max_px = max(0, int(cfg.stake_height_max / cfg.vertical_scale))
    holes_depth_px = int(cfg.holes_depth / cfg.vertical_scale)

    platform_width_px = max(1, int(cfg.platform_width / cfg.horizontal_scale))

    hf_raw = np.full((width_pixels, length_pixels), holes_depth_px, dtype=float)
    half_lower = stake_side_px // 2
    half_upper = stake_side_px - half_lower

    # Build a deterministic RNG for this sub-terrain when cfg.seed is provided.
    # We mix in quantized difficulty so each tile can still look different while
    # remaining reproducible across runs.
    if getattr(cfg, "seed", None) is not None:
        difficulty_key = int(round(float(difficulty) * 1_000_000.0))
        local_seed = (int(cfg.seed) * 1_000_003 + difficulty_key) % (2**32)
        rng = np.random.default_rng(local_seed)
    else:
        rng = np.random.default_rng()
    stake_height_values = (
        np.arange(-stake_height_max_px, stake_height_max_px + 1)
        if stake_height_max_px > 0
        else np.array([0], dtype=int)
    )

    def paint_square(cx: int, cy: int, value: int) -> None:
        if cx < 0 or cx >= width_pixels or cy < 0 or cy >= length_pixels:
            return
        x1 = max(0, cx - half_lower)
        x2 = min(width_pixels, cx + half_upper)
        y1 = max(0, cy - half_lower)
        y2 = min(length_pixels, cy + half_upper)
        hf_raw[x1:x2, y1:y2] = value

    def place_alternate_columns(start_pos: int, along_x: bool) -> None:
        offset = column_gap_px // 2  # Alternating offset
        step = stake_gap_px + stake_side_px
        while start_pos < (width_pixels if along_x else length_pixels):
            jitter = (
                rng.integers(-column_jitter_px, column_jitter_px + 1)
                if column_jitter_px > 0
                else 0
            )
            height_value = int(rng.choice(stake_height_values))

            if along_x:
                cx = start_pos
                cy = (length_pixels // 2) + offset + jitter
                paint_square(cx, cy, height_value)
            else:
                cy = start_pos
                cx = (width_pixels // 2) + offset + jitter
                paint_square(cx, cy, height_value)

            # Flip offset for alternating pattern
            offset = -offset
            start_pos += step

    # Place alternating columns along x and y
    place_alternate_columns(0, along_x=True)
    place_alternate_columns(0, along_x=False)

    # add the platform in the center
    x1 = (width_pixels - platform_width_px) // 2
    x2 = (width_pixels + platform_width_px) // 2
    y1 = (length_pixels - platform_width_px) // 2
    y2 = (length_pixels + platform_width_px) // 2
    hf_raw[x1:x2, y1:y2] = 0

    return np.rint(hf_raw).astype(np.int16)
