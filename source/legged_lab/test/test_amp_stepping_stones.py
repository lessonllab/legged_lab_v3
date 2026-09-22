import copy
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts' / 'rsl_rl'))
from viser_targets import TerrainRayPicker

from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_target_v6_env_cfg import G1AmpDepthTargetV6EnvCfg
from legged_lab.tasks.locomotion.amp.config.g1.terrain_variants import add_stepping_stones, stepping_stones_fall
from legged_lab.tasks.locomotion.amp.mdp.commands.target_velocity import terrain_column_names


class TestSteppingStones(unittest.TestCase):
    def test_opt_in_isolation_and_sampling(self):
        cfg = G1AmpDepthTargetV6EnvCfg()
        before = copy.deepcopy(cfg.to_dict())
        added = copy.deepcopy(cfg)
        add_stepping_stones(added)
        added.validate()
        self.assertEqual(cfg.to_dict(), before)
        generator = added.scene.terrain.terrain_generator
        self.assertEqual(len(generator.sub_terrains), 7)
        self.assertEqual(terrain_column_names(generator).count('hf_steppingstones'), 4)
        stone = generator.sub_terrains['hf_steppingstones']
        self.assertEqual(stone.holes_depth, -2.)
        self.assertEqual(stone.flat_patch_sampling['target'].z_range, (-.01, .01))
        self.assertGreater(stone.flat_patch_sampling['target'].x_range[0], stone.platform_width / 2)
        count = generator.num_cols
        add_stepping_stones(added)
        self.assertEqual(generator.num_cols, count)

    def test_actual_mesh_endpoints_keep_gaps_and_spawn_platform(self):
        cfg = G1AmpDepthTargetV6EnvCfg()
        add_stepping_stones(cfg)
        generator = cfg.scene.terrain.terrain_generator
        stone = generator.sub_terrains['hf_steppingstones']
        stone.size = generator.size
        for difficulty in (0., 1.):
            np.random.seed(42)
            meshes, origin = stone.function(difficulty, stone)
            vertices = np.concatenate([mesh.vertices for mesh in meshes])
            self.assertTrue(np.isfinite(vertices).all())
            self.assertAlmostEqual(vertices[:, 2].min(), -2., places=5)
            np.testing.assert_allclose(origin, [4., 4., 0.])
            picker = TerrainRayPicker()
            for mesh in meshes:
                picker.add_mesh(mesh.vertices, mesh.faces)
            for x, y in ((4., 4.), (7.85, 4.), (.15, 4.), (4., .15), (4., 7.85)):
                self.assertAlmostEqual(picker.pick([x, y, 2.], [0., 0., -1.])[2], 0.)
            for x, y in ((2., 2.), (6., 2.), (2., 6.), (6., 6.)):
                self.assertAlmostEqual(picker.pick([x, y, 2.], [0., 0., -1.])[2], -2.)
    def test_ame_generator_parity_with_requested_size_and_gap(self):
        import importlib.util
        import ast
        source = Path('/home/ljc/AME_Locomotion/source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/terrains/loco_hf_terrains.py')
        if not source.exists():
            self.skipTest('AME checkout is not installed on this machine')
        spec = importlib.util.spec_from_file_location('ame_reference_terrain', source)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        cfg = G1AmpDepthTargetV6EnvCfg()
        add_stepping_stones(cfg)
        stone = cfg.scene.terrain.terrain_generator.sub_terrains['hf_steppingstones']
        play_source = source.parents[1] / '29dof' / 'velocity_env_cfg_29dof.py'
        calls = [node for node in ast.walk(ast.parse(play_source.read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'HfAlternateColumnStakesTerrainCfg']
        self.assertEqual(len(calls), 1)
        requested = {'stake_side_range': (.4, .4), 'stake_gap_range': (.1, .1)}
        for item in calls[0].keywords:
            if item.arg != 'proportion':  # Distribution across terrain types is local.
                self.assertEqual(getattr(stone, item.arg), requested.get(item.arg, ast.literal_eval(item.value)))
        stone.size = (8., 8.)
        for seed in (0, 42):
            for difficulty in (0., .5, 1.):
                stone.seed = seed
                original = copy.deepcopy(stone)
                original.horizontal_scale, original.vertical_scale, original.slope_threshold = .05, .005, .75
                expected, expected_origin = reference.alternate_column_stakes_terrain(difficulty, original)
                actual, actual_origin = stone.function(difficulty, stone)
                np.testing.assert_array_equal(actual_origin, expected_origin)
                np.testing.assert_array_equal(actual[0].vertices, expected[0].vertices)
                np.testing.assert_array_equal(actual[0].faces, expected[0].faces)

    def test_pit_fall_only_applies_to_stone_columns(self):
        scene = NS(terrain=NS(terrain_types=torch.tensor([0, 1, 1])),
                   env_origins=torch.tensor([[0.,0.,0.], [0.,0.,3.], [0.,0.,-2.]]))
        robot = NS(data=NS(root_pos_w=NS(torch=torch.tensor([[0.,0.,-.5], [0.,0.,3.8], [0.,0.,-1.8]]))))
        class Scene:
            terrain, env_origins = scene.terrain, scene.env_origins
            def __getitem__(self, key): return robot
        env = NS(scene=Scene(), device='cpu', command_manager=NS(
            get_term=lambda _: NS(column_names=['boxes', 'hf_steppingstones'])))
        self.assertEqual(stepping_stones_fall(env).tolist(), [False, False, True])


if __name__ == '__main__':
    unittest.main()
