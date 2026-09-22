"""Finite CPU check of island terrain and all 500 perimeter-road goals."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

from isaaclab.terrains import TerrainGenerator
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_target_v6_env_cfg import G1AmpDepthTargetV6EnvCfg
from legged_lab.tasks.locomotion.amp.config.g1.terrain_variants import add_stepping_stones

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rsl_rl'))
from viser_targets import TerrainRayPicker


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    cfg = G1AmpDepthTargetV6EnvCfg()
    add_stepping_stones(cfg)
    generator = cfg.scene.terrain.terrain_generator
    generator.sub_terrains = {'hf_steppingstones': generator.sub_terrains['hf_steppingstones']}
    generator.num_cols, generator.num_rows = 1, 10
    generator.seed, generator.use_cache = 42, False
    terrain = TerrainGenerator(generator, device='cpu')
    goals = terrain.flat_patches['target'].cpu().numpy()
    local = goals - terrain.terrain_origins[:, :, None, :]
    assert goals.shape == (10, 1, 50, 3)
    assert np.isfinite(goals).all()
    np.testing.assert_allclose(local[..., 2], 0., atol=2e-5)
    np.testing.assert_allclose(local[..., 0], 3.85, atol=2e-5)
    np.testing.assert_allclose(local[..., 1], 0., atol=2e-5)
    picker = TerrainRayPicker()
    picker.add_mesh(terrain.terrain_mesh.vertices, terrain.terrain_mesh.faces)
    errors = []
    for level in range(10):
        point = goals[level, 0, 0]
        hit = picker.pick(point + [0., 0., 1.], [0., 0., -1.])
        assert hit is not None
        errors.append(float(np.linalg.norm(hit-point)))
    assert max(errors) < 1e-4, errors
    result = dict(levels=10, goals=500, all_goals_finite=True,
                  goal_local_height_range=[float(local[..., 2].min()), float(local[..., 2].max())],
                  goal_x_range=[float(local[..., 0].min()), float(local[..., 0].max())],
                  layout='AME alternate-column generator with requested 0.4m sides, 0.1m gaps, 2m central platform, 0.25m border',
                  pit_floor_z=-2.,
                  independent_ray_checks=10, max_ray_error=max(errors),
                  scope='Generated geometry and navigation goals; not trained crossing ability.')
    output = Path('docs/analysis/stepping_stones_20260913/terrain.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS', json.dumps(result))


if __name__ == '__main__':
    main()
