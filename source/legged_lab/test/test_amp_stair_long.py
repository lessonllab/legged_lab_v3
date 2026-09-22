import unittest
from unittest.mock import patch
import numpy as np
import torch
import test_amp_stair_speed as fixtures
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_stairs_long_env_cfg import G1AmpStairsLongEnvCfg
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_stairs_speed_env_cfg import G1AmpStairsSpeedEnvCfg
from legged_lab.rsl_rl.amp.ppo_amp import PPOAMP
from legged_lab.tasks.locomotion.amp.mdp.stair_course import sample_stair_stages, stair_curriculum
from types import SimpleNamespace as NS


def profile(cfg, name, row=1):
    generator = cfg.scene.terrain.terrain_generator
    terrain = generator.sub_terrains[name].copy()
    terrain.size = generator.size
    meshes, origin = terrain.function((row + .5) / 10., terrain)
    x = np.linspace(origin[0], generator.size[0] - .3, 4001)
    z = np.full_like(x, -np.inf)
    # Stair generator consists entirely of axis-aligned boxes. Intersect their
    # top surfaces along the actual center-to-goal route, rather than trusting cfg.
    for mesh in meshes:
        lo, hi = mesh.bounds
        valid = (x >= lo[0]) & (x <= hi[0]) & (origin[1] >= lo[1]) & (origin[1] <= hi[1])
        z[valid] = np.maximum(z[valid], hi[2])
    return x, z, meshes, origin


class TestLongStairs(unittest.TestCase):
    def test_challenge_sampling_changes_one_axis_and_respects_limits(self):
        draw = (torch.arange(100) + .5) / 100
        levels, phases = sample_stair_stages(draw, 3, 1, 5, challenge=True)
        self.assertEqual(((levels == 3) & (phases == 1)).sum(), 60)
        self.assertEqual(((levels == 2) & (phases == 0)).sum(), 20)
        self.assertEqual(((levels == 3) & (phases == 2)).sum(), 10)
        self.assertEqual(((levels == 4) & (phases == 1)).sum(), 10)
        for level, phase in ((1, 0), (9, 4)):
            levels, phases = sample_stair_stages(draw, level, phase, 5, challenge=True)
            self.assertTrue(((levels >= 1) & (levels <= 9)).all())
            self.assertTrue(((phases >= 0) & (phases <= 4)).all())

    def test_late_successes_count_after_unchanged_failed_window(self):
        alg, c, t = fixtures.TestStairSpeed().fixture(200)
        c._ids = lambda ids: torch.as_tensor(ids, dtype=torch.long)
        c.preserve_same_stage_evidence = True
        c.adjacent_stage_challenges = True
        c.frontier[:] = 1
        t.terrain_levels[c.stairs] = 1
        c.stair_manual_episode = torch.zeros(200, dtype=torch.bool)
        c.stair_completed = torch.zeros(200, dtype=torch.bool)
        c.stair_traversed = torch.zeros(200, dtype=torch.bool)
        c.stair_strict_settled = torch.zeros(200, dtype=torch.bool)
        c.exit_geometry = lambda: torch.ones(200, dtype=torch.bool)
        up = torch.where(c.group == 1)[0]
        early, late = up[:50], up[50:]
        failed = torch.zeros(200, dtype=torch.bool)
        failed[early] = True
        alg.course_env.termination_manager = NS(terminated=failed)
        with patch('legged_lab.tasks.locomotion.amp.mdp.stair_course.balanced_curriculum', return_value={}):
            stair_curriculum(alg.course_env, early)
            self.assertEqual(c.epoch[0], 0)
            c.stair_completed[late] = True
            c.stair_traversed[late] = True
            stair_curriculum(alg.course_env, late)
            self.assertEqual(c.attempts[0], 20)
            self.assertEqual(c.wins[0], 20)
            # A real promotion still invalidates in-flight old-stage episodes.
            c.attempts[0], c.wins[0] = 49, 49
            more = up[:1]
            t.terrain_levels[more], c.stair_phase[more], c.stair_epoch[more] = 1, 0, 0
            failed[more] = False
            c.stair_completed[more] = True
            stair_curriculum(alg.course_env, more)
            self.assertEqual(c.phase[0], 1)
            self.assertEqual(c.epoch[0], 1)

    def test_legacy_descent_challenges_disabled_but_normal_promotion_remains(self):
        cfg = G1AmpStairsLongEnvCfg()
        cfg.validate()
        command = cfg.commands.base_velocity.class_type
        self.assertEqual(command.promotion_version, 11)
        self.assertEqual(cfg.rewards.feet_edge.params['kind'], 'edge')
        self.assertIsNone(cfg.rewards.feet_support)
        self.assertEqual(cfg.rewards.feet_toe_riser.params['kind'], 'toe')
        self.assertEqual(cfg.rewards.feet_loaded_overhang.params['kind'], 'support')
        self.assertEqual(cfg.rewards.feet_swing_clearance.params['kind'], 'clearance')
        self.assertEqual(cfg.rewards.feet_slide.weight, -.2)
        self.assertTrue(cfg.rewards.dont_wait.params['use_yaw_frame'])
        from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_stairs_long_ppo_cfg import G1AmpStairsLongPPORunnerCfg
        self.assertEqual(G1AmpStairsLongPPORunnerCfg().algorithm.amp_cfg.amp_discriminator.style_reward_scale, .25)
        alg, c, t = fixtures.TestStairSpeed().fixture(200)
        c.challenge_directions = (True, False)
        c.adjacent_stage_challenges = True
        c.preserve_same_stage_evidence = True
        c._ids = lambda ids: torch.as_tensor(ids, dtype=torch.long)
        c.stair_manual_episode = torch.zeros(200,dtype=torch.bool)
        c.stair_completed = torch.ones(200,dtype=torch.bool)
        c.stair_traversed = torch.ones(200,dtype=torch.bool)
        c.stair_strict_settled = torch.ones(200,dtype=torch.bool)
        c.exit_geometry = lambda: torch.ones(200,dtype=torch.bool)
        alg.course_env.termination_manager = NS(terminated=torch.zeros(200,dtype=torch.bool))
        down = torch.where(c.group == 2)[0]
        t.terrain_levels[down] = 2
        c.attempts[1], c.wins[1] = 10, 10
        with patch('legged_lab.tasks.locomotion.amp.mdp.stair_course.balanced_curriculum',return_value={}):
            stair_curriculum(alg.course_env, down)
        self.assertEqual(c.phase[1], 1)
        self.assertTrue((t.terrain_levels[down] <= c.frontier[1]).all())
        self.assertTrue((c.stair_phase[down] <= c.phase[1]).all())

    def test_version_three_migration_clears_biased_evidence(self):
        alg, c, t = fixtures.TestStairSpeed().fixture()
        c.promotion_version = 2
        c.attempts[:], c.wins[:] = 40, 12
        c.phase[:] = 1
        with patch.object(PPOAMP, 'save', return_value={}):
            saved = alg.save()
        c.promotion_version = 3
        with patch.object(PPOAMP, 'load', return_value=True):
            alg.load(saved)
        self.assertEqual(c.attempts.tolist(), [0, 0])
        self.assertEqual(c.wins.tolist(), [0, 0])
        self.assertEqual(c.phase.tolist(), [1, 1])

    def test_actual_route_has_17_steps_of_same_rise_and_safe_goal(self):
        short, long = G1AmpStairsSpeedEnvCfg(), G1AmpStairsLongEnvCfg()
        long.validate()
        for name, sign in (('pyramid_stairs', -1), ('pyramid_stairs_inv', 1)):
            _, old_z, _, _ = profile(short, name)
            self.assertEqual(np.count_nonzero(abs(np.diff(old_z)) > .001), 7)
            for row, rise in ((1, .08), (4, .14), (9, .23)):
                x, z, meshes, origin = profile(long, name, row)
                # Every face is horizontal or vertical: no rounded/bevel faces.
                for mesh in meshes:
                    normals = np.abs(mesh.face_normals)
                    np.testing.assert_allclose(normals.max(axis=1), 1., atol=1e-6)
                    np.testing.assert_allclose(normals.sum(axis=1), 1., atol=1e-6)
                self.assertTrue(np.isfinite(z).all())
                jumps = np.diff(z)[abs(np.diff(z)) > .001]
                self.assertEqual(len(jumps), 17)
                np.testing.assert_allclose(jumps, sign * rise, atol=1e-6)
                self.assertAlmostEqual(z[0], origin[2])
                self.assertAlmostEqual(z[-1], 0.)
                self.assertAlmostEqual(x[-1] - x[0], 6.7)
                self.assertTrue((abs(z[x > 13.1]) < 1e-6).all())

    def test_flat_entry_and_nonstairs_keep_original_mesh_resolution(self):
        cfg = G1AmpStairsLongEnvCfg()
        for name in ('pyramid_stairs', 'pyramid_stairs_inv'):
            _, z, _, _ = profile(cfg, name, 0)
            np.testing.assert_allclose(z, 0., atol=1e-6)
        old = G1AmpStairsSpeedEnvCfg().scene.terrain.terrain_generator
        new = cfg.scene.terrain.terrain_generator
        for name in set(new.sub_terrains) - {'pyramid_stairs', 'pyramid_stairs_inv'}:
            a, b = old.sub_terrains[name].copy(), new.sub_terrains[name].copy()
            a.size, b.size = old.size, new.size
            np.random.seed(42)
            torch.manual_seed(42)
            old_meshes, old_origin = a.function(.35, a)
            np.random.seed(42)
            torch.manual_seed(42)
            new_meshes, new_origin = b.function(.35, b)
            self.assertEqual(len(new_meshes), len(old_meshes) + 4)
            for original, padded in zip(old_meshes, new_meshes):
                np.testing.assert_allclose(padded.vertices, original.vertices + [3., 3., 0.], atol=1e-6)
            np.testing.assert_allclose(new_origin, old_origin + [3., 3., 0.])
        self.assertGreater(cfg.episode_length_s, 6.7 / .35 + 2.)

    def test_geometry_migration_resets_long_stairs_but_retains_flat_skills(self):
        alg, c, t = fixtures.TestStairSpeed().fixture()
        c.promotion_version = 2
        c.flat_speed_tier[c.flat_pool] = 5
        c.frontier[:] = 4
        c.phase[:] = 2
        c.attempts[:] = 49
        with patch.object(PPOAMP, 'save', return_value={}):
            short = alg.save()
        c.terrain_profile = 'long24_v1'
        c.terrain_tile_size = (24., 24.)
        with patch.object(PPOAMP, 'load', return_value=True):
            alg.load(short)
        self.assertTrue((c.flat_speed_tier[c.flat_pool] == 5).all())
        self.assertEqual(c.frontier.tolist(), [1, 1])
        self.assertEqual(c.phase.tolist(), [0, 0])
        self.assertTrue((t.terrain_levels[c.stairs] == 1).all())
        self.assertEqual(c.attempts.tolist(), [0, 0])
        c.frontier[:] = 3
        c.phase[:] = 4
        c.attempts[:] = 21
        with patch.object(PPOAMP, 'save', return_value={}):
            long = alg.save()
        with patch.object(PPOAMP, 'load', return_value=True):
            alg.load(long)
        self.assertEqual(c.frontier.tolist(), [3, 3])
        self.assertEqual(c.attempts.tolist(), [21, 21])
        c.terrain_profile = 'short8_v1'
        c.terrain_tile_size = (8., 8.)
        with patch.object(PPOAMP, 'load', return_value=True):
            with self.assertRaisesRegex(ValueError, 'matching long/short'):
                alg.load(long)


if __name__ == '__main__':
    unittest.main()
