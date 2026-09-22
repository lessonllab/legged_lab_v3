import unittest
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch
import torch
import test_amp_stair_speed as fixtures
from legged_lab.rsl_rl.amp.ppo_amp import PPOAMP
from legged_lab.tasks.locomotion.amp.mdp.adaptive_stair_course import (
    AdaptiveStairTargetCommand, ADAPTIVE_FIELDS, HEIGHT, SPEED, REVIEW, CHALLENGE,
    adaptive_step, sample_adaptive_stages, adaptive_stair_curriculum, MAX_ADAPTIVE_LEVEL,
    stair_crossing_reached, crossing_success,
)
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_stairs_long_env_cfg import G1AmpStairsLongEnvCfg


class TestAdaptiveStairs(unittest.TestCase):
    def fixture(self, n=1000):
        alg, c, t = fixtures.TestStairSpeed().fixture(n)
        c.device = 'cpu'
        c.promotion_version = 7
        c.height_speed_phase = 1
        c._ids = lambda ids: torch.arange(n) if ids is None else torch.as_tensor(ids, dtype=torch.long)
        c.stair_lane = torch.zeros(n, dtype=torch.long)
        for name in ADAPTIVE_FIELDS:
            setattr(c, name, torch.zeros(2, dtype=torch.float if name.endswith(('rate', 'age')) else torch.long))
        c.speed_last_rate.fill_(-1.)
        for name in ('sample_adaptive', 'save_adaptive_state', 'load_adaptive_state'):
            setattr(c, name, MethodType(getattr(AdaptiveStairTargetCommand, name), c))
        c.stair_manual_episode = torch.zeros(n, dtype=torch.bool)
        c.stair_completed = torch.ones(n, dtype=torch.bool)
        c.stair_traversed = torch.ones(n, dtype=torch.bool)
        c.exit_geometry = lambda: torch.ones(n, dtype=torch.bool)
        c.metrics = {k: torch.full((n,), v) for k, v in (
            ('cruise_seconds', 2.), ('cruise_tracking', .9), ('cruise_actual_mps', .65), ('cruise_command_mps', .65))}
        c.speed_success = lambda: torch.ones(n, dtype=torch.bool)
        c.promotion_diagnostics = lambda: {}
        alg.course_env.cfg = NS(episode_length_s=60.)
        alg.course_env.termination_manager = NS(terminated=torch.zeros(n, dtype=torch.bool))
        return alg, c, t

    def test_height_moves_without_final_speed_and_waits_for_late_success(self):
        self.assertEqual(adaptive_step(2, 80, 100, 60., 9), (3, True))
        self.assertEqual(adaptive_step(2, 49, 100, 60., 9), (1, True))
        self.assertEqual(adaptive_step(2, 50, 100, 60., 9), (2, True))
        self.assertEqual(adaptive_step(2, 0, 100, 59., 9), (2, False))
        self.assertEqual(adaptive_step(2, 99, 99, 90., 9), (2, False))
        self.assertEqual(adaptive_step(1, 0, 100, 60., 9), (1, True))
        self.assertEqual(adaptive_step(9, 100, 100, 60., 9), (9, True))

    def test_sampling_one_axis_limits_and_retains_fast_practice(self):
        draw = (torch.arange(100) + .5) / 100
        level, phase, lane = sample_adaptive_stages(draw, 2, 3)
        self.assertEqual(torch.bincount(lane).tolist(), [50, 20, 20, 10])
        self.assertTrue((level[lane == HEIGHT] == 2).all())
        self.assertTrue((phase[lane == HEIGHT] == 1).all())
        self.assertTrue((level[lane == SPEED] == 1).all())
        self.assertTrue((phase[lane == SPEED] == 3).all())
        self.assertTrue((level[lane == REVIEW] == 1).all())
        self.assertTrue((level[lane == CHALLENGE] == 3).all())
        for height in (1, 9):
            level, _, _ = sample_adaptive_stages(draw, height, 4)
            self.assertTrue(((level >= 1) & (level <= MAX_ADAPTIVE_LEVEL)).all())
        self.assertEqual(MAX_ADAPTIVE_LEVEL, 9)
        self.assertEqual(adaptive_step(7, 100, 100, 60., MAX_ADAPTIVE_LEVEL), (8, True))

    def test_height_and_speed_and_directions_have_separate_evidence(self):
        alg, c, t = self.fixture()
        up = torch.where(c.group == 1)[0]
        down = torch.where(c.group == 2)[0]
        c.phase[:] = 2
        t.terrain_levels[c.stairs] = 2
        c.stair_phase[c.stairs] = 1
        speed = up[100:200]
        c.stair_lane[speed] = SPEED
        t.terrain_levels[speed], c.stair_phase[speed] = 1, 2
        c.height_age[:], c.speed_age[:] = 60., 60.
        alg.course_env.termination_manager.terminated[down[:100]] = True
        ids = torch.cat((up[:200], down[:100]))
        with patch('legged_lab.tasks.locomotion.amp.mdp.adaptive_stair_course.balanced_curriculum', return_value={}):
            adaptive_stair_curriculum(alg.course_env, ids)
        self.assertEqual(c.frontier.tolist(), [3, 1])
        self.assertEqual(c.phase.tolist(), [3, 2])
        self.assertEqual(c.epoch.tolist(), [1, 1])
        self.assertEqual(c.speed_epoch.tolist(), [1, 0])

    def test_review_challenge_stale_and_skip_do_not_promote(self):
        alg, c, t = self.fixture()
        ids = torch.where(c.group == 1)[0]
        t.terrain_levels[ids], c.stair_phase[ids] = 2, 1
        c.stair_lane[ids[:100]] = REVIEW
        c.stair_lane[ids[100:200]] = CHALLENGE
        c.stair_epoch[ids[200:300]] = 7
        c.skip_curriculum_once[ids[300:]] = True
        c.height_age[:] = 60.
        with patch('legged_lab.tasks.locomotion.amp.mdp.adaptive_stair_course.balanced_curriculum', return_value={}):
            adaptive_stair_curriculum(alg.course_env, ids)
        self.assertEqual(c.frontier.tolist(), [2, 2])
        self.assertEqual(c.attempts.tolist(), [0, 0])

    def test_overspeed_blocks_height_even_if_old_promotion_would_pass(self):
        alg, c, t = self.fixture()
        ids = torch.where(c.group == 1)[0][:100]
        t.terrain_levels[ids], c.stair_phase[ids] = 2, 1
        c.metrics['cruise_actual_mps'][ids] = 1.1
        c.height_age[:] = 60.
        with patch('legged_lab.tasks.locomotion.amp.mdp.adaptive_stair_course.balanced_curriculum', return_value={}):
            stats = adaptive_stair_curriculum(alg.course_env, ids)
        self.assertEqual(c.frontier.tolist(), [1, 2])
        self.assertEqual(stats['ascent/height/successes'], 0)

    def test_migration_roundtrip_and_smaller_play_batch(self):
        old, oc, _ = fixtures.TestStairSpeed().fixture(1000)
        oc.promotion_version = 5
        oc.frontier[:] = 1
        oc.phase[:] = 3
        oc.flat_speed_tier[oc.flat_pool] = 4
        with patch.object(PPOAMP, 'save', return_value={}):
            saved = old.save()
        alg, c, t = self.fixture()
        with patch.object(PPOAMP, 'load', return_value=True):
            alg.load(saved)
        self.assertEqual(c.frontier.tolist(), [2, 2])
        self.assertEqual(c.phase.tolist(), [3, 3])
        self.assertTrue((c.flat_speed_tier[c.flat_pool] == 4).all())
        for direction in (1, 2):
            self.assertTrue((t.terrain_levels[c.group == direction] == 3).any())
        c.frontier[:] = torch.tensor([4, 3])
        c.attempts[:], c.wins[:], c.height_age[:] = 21, 17, 35.
        c.speed_attempts[:], c.speed_wins[:], c.speed_age[:] = 27, 24, 42.
        with patch.object(PPOAMP, 'save', return_value={}):
            again = alg.save()
        small, sc, _ = self.fixture(20)
        with patch.object(PPOAMP, 'load', return_value=True):
            small.load(again)
        self.assertEqual(sc.frontier.tolist(), [4, 3])
        self.assertEqual(sc.attempts.tolist(), [21, 21])
        self.assertEqual(sc.speed_attempts.tolist(), [27, 27])
        self.assertEqual(sc.height_age.tolist(), [35., 35.])
        self.assertEqual(sc.speed_age.tolist(), [42., 42.])
        with patch.object(PPOAMP, 'load', return_value=True):
            with self.assertRaisesRegex(ValueError, 'newer promotion'):
                old.load(again)

    def test_20cm_checkpoint_migrates_without_claiming_higher_proficiency(self):
        alg, c, t = self.fixture()
        c.frontier[:] = 7
        c.phase[:] = 4
        c.attempts[:] = 100
        with patch.object(PPOAMP, 'save', return_value={}):
            saved = alg.save()
        state = saved['scratch_course']
        state['heights'] = (0., .08, .10, .12, .14, .16, .18, .20, .215, .23)
        state['adaptive_height']['max_height_cm'] = 20
        with patch.object(PPOAMP, 'load', return_value=True):
            alg.load(saved)
        self.assertEqual(c.frontier.tolist(), [7, 7])
        self.assertEqual(c.phase.tolist(), [4, 4])
        self.assertEqual(c.attempts.tolist(), [0, 0])
        self.assertTrue((t.terrain_levels[c.stairs] == 8).any())
        self.assertEqual(c.save_adaptive_state()['max_height_cm'], 30)

    def test_long_config_enables_both_directions_and_keeps_goals(self):
        cfg = G1AmpStairsLongEnvCfg()
        cfg.validate()
        self.assertIs(cfg.curriculum.terrain_levels.func, adaptive_stair_curriculum)
        self.assertEqual(cfg.commands.base_velocity.class_type.height_speed_phase, 1)
        self.assertEqual(cfg.commands.base_velocity.class_type.stair_speed_schedule[-1][1], 1.5)
        self.assertEqual(cfg.commands.base_velocity.class_type.flat_speed_schedule[-1][1], 3.)

    def test_legacy_higher_frontier_respects_30cm_limit(self):
        old, oc, ot = fixtures.TestStairSpeed().fixture(1000)
        oc.promotion_version = 5
        oc.frontier[:] = 9
        ot.terrain_levels[oc.stairs] = 9
        with patch.object(PPOAMP, 'save', return_value={}):
            saved = old.save()
        alg, c, t = self.fixture()
        with patch.object(PPOAMP, 'load', return_value=True):
            alg.load(saved)
        self.assertEqual(c.frontier.tolist(), [9, 9])
        self.assertTrue((t.terrain_levels[c.stairs] <= 9).all())
        self.assertEqual(c.save_adaptive_state()['max_height_cm'], 30)

    def test_crossing_without_standing_counts_but_simultaneous_fall_does_not(self):
        alg, c, t = self.fixture()
        ids = torch.where(c.group == 1)[0][:100]
        t.terrain_levels[ids], c.stair_phase[ids] = 2, 1
        c.stair_completed.zero_()  # No stop/settle was recorded.
        c.stair_traversed.zero_()  # Metrics have not yet consumed this step.
        c.height_age[:] = 60.
        alg.course_env.termination_manager.terminated[ids[:10]] = True
        with patch('legged_lab.tasks.locomotion.amp.mdp.adaptive_stair_course.balanced_curriculum', return_value={}):
            stats = adaptive_stair_curriculum(alg.course_env, ids)
        self.assertEqual(stats['ascent/height/successes'], 90)
        self.assertEqual(c.frontier.tolist(), [3, 2])
        self.assertEqual(crossing_success(torch.tensor([True, True, False]),
                                        torch.tensor([False, True, False])).tolist(), [True, False, False])

    def test_crossing_timeout_excludes_flat_manual_and_reset_transitions(self):
        c = NS(stairs=torch.tensor([True, False, True, True, True]),
               stair_manual_episode=torch.tensor([False, False, True, False, False]),
               _skip_progress_transition=torch.tensor([False, False, False, True, False]),
               exit_geometry=lambda: torch.ones(5, dtype=torch.bool))
        env = NS(command_manager=NS(get_term=lambda _: c),episode_length_buf=torch.tensor([1,1,1,1,0]))
        self.assertEqual(stair_crossing_reached(env).tolist(), [True, False, False, False, False])
        cfg = G1AmpStairsLongEnvCfg()
        self.assertTrue(cfg.terminations.stair_crossing.time_out)

    def test_halved_geometry_preserves_skills_and_clears_all_old_windows(self):
        alg, c, t = self.fixture()
        c.promotion_version = 6
        c.terrain_profile, c.terrain_tile_size = 'long24_v1', (24.,24.)
        c.frontier[:] = torch.tensor([2,3]); c.phase[:] = torch.tensor([1,2])
        c.attempts[:], c.wins[:], c.height_age[:] = 70, 50, 42.
        c.speed_attempts[:], c.speed_wins[:], c.speed_age[:] = 80, 65, 55.
        c.flat_speed_tier[c.flat_pool] = 4
        with patch.object(PPOAMP, 'save', return_value={}):
            saved = alg.save()
        c.promotion_version = 7
        c.terrain_profile, c.terrain_tile_size = 'stairs14_v2', (14.,14.)
        with patch.object(PPOAMP, 'load', return_value=True):
            alg.load(saved)
        self.assertEqual(c.frontier.tolist(), [2,3])
        self.assertEqual(c.phase.tolist(), [1,2])
        self.assertEqual(c.attempts.sum(), 0)
        self.assertEqual(c.speed_attempts.sum(), 0)
        self.assertEqual(c.height_age.sum(), 0)
        self.assertEqual(c.speed_age.sum(), 0)
        self.assertTrue((c.flat_speed_tier[c.flat_pool] == 4).all())


if __name__ == '__main__':
    unittest.main()
