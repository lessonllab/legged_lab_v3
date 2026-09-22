import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
import test_amp_stair_course as fixtures
from legged_lab.tasks.locomotion.amp.mdp.stair_course import StairTargetCommand, advance_stage, stair_curriculum
from legged_lab.tasks.locomotion.amp.mdp.stair_speed_course import (
    StairSpeedTargetCommand, FAST_FLAT_RANGES, FAST_STAIR_RANGES, cruise_success, cruise_failures,
    cruise_ready_for_promotion,
    overspeed_cost,
)
from legged_lab.tasks.locomotion.amp.mdp.foot_support import support_costs
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_stairs_speed_env_cfg import G1AmpStairsSpeedEnvCfg
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_stairs_env_cfg import G1AmpStairsEnvCfg
from legged_lab.rsl_rl.amp.ppo_amp import PPOAMP


class TestStairSpeed(unittest.TestCase):
    def test_overspeed_deadband_direction_mask_and_bound(self):
        actual = torch.tensor([.45, .59, 1.45, 1.45, 100.])
        desired = torch.full_like(actual, .45)
        enabled = torch.tensor([True, True, True, False, True])
        result = overspeed_cost(actual, desired, enabled)
        torch.testing.assert_close(result, torch.tensor([0., 0., .85**2, 0., 4.]))

    def fixture(self, n=100, fast=True):
        fixture = fixtures.TestStairCourse()
        fixture.setUp()
        alg, c, t = fixture.algorithm_fixture(n)
        if fast:
            c.course_profile = 'speed_v1'
            c.flat_speed_schedule = FAST_FLAT_RANGES
            c.stair_speed_schedule = FAST_STAIR_RANGES
        return alg, c, t

    def test_legacy_migration_and_high_speed_roundtrip(self):
        old, oc, _ = self.fixture(fast=False)
        oc.flat_speed_tier[oc.flat_pool] = 3
        oc.phase[:] = 1
        oc.attempts[:] = 49
        oc.wins[:] = 45
        with patch.object(PPOAMP, 'save', return_value={}):
            legacy = old.save()
        new, c, _ = self.fixture()
        with patch.object(PPOAMP, 'load', return_value=True):
            new.load(legacy)
        self.assertEqual(c.phase.tolist(), [1, 1])
        self.assertTrue((c.flat_speed_tier[c.flat_pool] == 3).all())
        self.assertEqual(c.attempts.sum(), 0)
        self.assertEqual(legacy['scratch_course']['attempts'].tolist(), [49, 49])
        c.flat_speed_tier[c.flat_pool] = 5
        c.phase[:] = 4
        c.stair_phase[c.stairs] = 4
        c.attempts[:] = 21
        with patch.object(PPOAMP, 'save', return_value={}):
            saved = new.save()
        self.assertEqual(saved['scratch_course']['version'], 5)
        small, sc, _ = self.fixture(20)
        with patch.object(PPOAMP, 'load', return_value=True):
            small.load(saved)
            with self.assertRaisesRegex(ValueError, 'matching Stairs-v1'):
                old.load(saved)
        self.assertEqual(sc.phase.tolist(), [4, 4])
        self.assertEqual(sc.attempts.tolist(), [21, 21])
        self.assertTrue((sc.flat_speed_tier[sc.flat_pool] == 5).all())

    def test_more_speed_before_more_height_and_single_step_demotion(self):
        for phase in range(4):
            self.assertEqual(advance_stage(2, phase, 41, 50, 5), (2, phase + 1, True))
        self.assertEqual(advance_stage(2, 4, 41, 50, 5), (3, 0, True))
        self.assertEqual(advance_stage(2, 4, 24, 50, 5), (2, 3, True))
        self.assertEqual(advance_stage(9, 4, 50, 50, 5), (9, 4, True))

    def test_cruise_rejects_crawling_overspeed_and_short_opportunities(self):
        seconds = torch.tensor([1., 1., 1., .1, 1.])
        actual = torch.tensor([1.4, .2, 2., 1.4, float('nan')])
        result = cruise_success(seconds, torch.ones(5), actual, torch.full((5,), 1.5))
        self.assertEqual(result.tolist(), [True, False, False, False, False])

    def test_metrics_exclude_braking_and_reset_transition(self):
        c = object.__new__(StairSpeedTargetCommand)
        c.group = torch.ones(3, dtype=torch.long)
        c._skip_progress_transition = torch.tensor([False, False, True])
        c.stair_manual_episode = torch.zeros(3, dtype=torch.bool)
        c._env = NS(step_dt=.02, episode_length_buf=torch.ones(3))
        c.speed_cap = torch.full((3,), 1.5)
        c.stair_phase = torch.full((3,), 4, dtype=torch.long)
        c.vel_command_b = torch.tensor([[1.5, 0., 0.], [.2, 0., 0.], [1.5, 0., 0.]])
        # Pitch and upward motion must not alter horizontal cruising evidence.
        c.robot = NS(data=NS(root_quat_w=torch.tensor([[.9659258,0.,.258819,0.]] * 3),
                            root_link_lin_vel_w=torch.tensor([[1.4, 0., .8]] * 3)))
        c.metrics = {prefix + k: torch.zeros(3) for prefix in ('', 'warm_')
                     for k in ('cruise_seconds', 'cruise_tracking', 'cruise_actual_mps', 'cruise_command_mps')}
        with patch.object(StairTargetCommand, '_update_metrics'):
            for _ in range(30):
                c._update_metrics()
        self.assertEqual(c.speed_success().tolist(), [True, False, False])
        self.assertEqual(c.metrics['warm_cruise_seconds'].tolist(), [0., 0., 0.])
        c._env.episode_length_buf[:] = 26
        with patch.object(StairTargetCommand, '_update_metrics'):
            c._update_metrics()
        self.assertAlmostEqual(c.metrics['warm_cruise_seconds'][0].item(), .02)

    def test_failure_reasons_cover_all_rejections(self):
        seconds = torch.tensor([.1, 1., 1., 1., 1., 1., 1.])
        tracking = torch.tensor([1., .2, 1., 1., 1., 1., 1.])
        actual = torch.tensor([1., 1., .1, 2., float('nan'), 0., 1.])
        command = torch.tensor([1., 1., 1., 1., 1., 0., 1.])
        failures = cruise_failures(seconds, tracking, actual, command)
        union = torch.stack(list(failures.values())).any(0)
        torch.testing.assert_close(union, ~cruise_success(seconds, tracking, actual, command))

    def test_next_band_readiness_is_not_low_speed_mastery(self):
        actual = torch.tensor([.65, .80, .65, .20, 1.8, 1.4])
        command = torch.tensor([.45, .45, .45, .45, 1.3, 1.3])
        phase = torch.tensor([0, 0, 0, 0, 4, 4])
        tracking = torch.tensor([.64, .64, .3, .9, .9, .9])
        ready = cruise_ready_for_promotion(torch.ones(6), tracking, actual, command, phase)
        self.assertEqual(ready.tolist(), [True, False, False, False, False, True])
        self.assertFalse(cruise_success(torch.ones(6), tracking, actual, command)[0])

    def test_changed_promotion_definition_clears_only_old_evidence(self):
        alg, c, t = self.fixture()
        c.phase[:] = 1
        c.attempts[:] = 49
        c.wins[:] = 45
        with patch.object(PPOAMP, 'save', return_value={}):
            saved = alg.save()
        c.promotion_version = 2
        with patch.object(PPOAMP, 'load', return_value=True):
            alg.load(saved)
        self.assertEqual(c.phase.tolist(), [1, 1])
        self.assertEqual(c.attempts.tolist(), [0, 0])
        self.assertEqual(saved['scratch_course']['attempts'].tolist(), [49, 49])
        c.attempts[:] = 21
        with patch.object(PPOAMP, 'save', return_value={}):
            new = alg.save()
        with patch.object(PPOAMP, 'load', return_value=True):
            alg.load(new)
        self.assertEqual(c.attempts.tolist(), [21, 21])
        c.promotion_version = 1
        with patch.object(PPOAMP, 'load', return_value=True):
            with self.assertRaisesRegex(ValueError, 'newer promotion'):
                alg.load(new)

    def test_successful_but_slow_traversals_do_not_promote(self):
        alg, c, t = self.fixture(200)
        c._ids = lambda ids: torch.as_tensor(ids, dtype=torch.long)
        c.stair_manual_episode = torch.zeros(200, dtype=torch.bool)
        c.stair_completed = c.stair_traversed = c.stair_strict_settled = torch.ones(200, dtype=torch.bool)
        c.exit_geometry = lambda: torch.ones(200, dtype=torch.bool)
        c.speed_success = lambda: torch.zeros(200, dtype=torch.bool)
        t.terrain_levels[c.stairs] = 2
        alg.course_env.termination_manager = NS(terminated=torch.zeros(200, dtype=torch.bool))
        with patch('legged_lab.tasks.locomotion.amp.mdp.stair_course.balanced_curriculum', return_value={}):
            stats = stair_curriculum(alg.course_env, torch.arange(200))
        self.assertEqual(c.phase[0], 0)
        self.assertEqual(stats['ascent/batch_successes'], 0)

    def test_contact_ramp_catches_early_loaded_edge_but_not_swing(self):
        h = torch.full((3, 1, 5, 11), -.035)
        h[:, :, :, 6:] -= .2
        forces = torch.tensor([[100.], [100.], [0.]])
        times = torch.tensor([[.02], [.04], [.04]])
        _, old_edge, _ = support_costs(h, forces, times)
        _, edge, _ = support_costs(h, forces, times, support_target=.9, contact_ramp=.04)
        self.assertEqual(old_edge.tolist(), [0., 0., 0.])
        self.assertEqual(edge.tolist(), [.5, 1., 0.])

    def test_config_keeps_policy_and_motion_inputs_compatible(self):
        old, new = G1AmpStairsEnvCfg(), G1AmpStairsSpeedEnvCfg()
        new.validate()
        for name in ('observations', 'motion_data', 'animation', 'terminations', 'actions'):
            self.assertEqual(getattr(old, name).to_dict(), getattr(new, name).to_dict())
        self.assertEqual(old.rewards.feet_edge.params.get('contact_ramp', 0.), 0.)
        self.assertEqual(new.commands.base_velocity.class_type.flat_speed_schedule[-1][1], 3.)
        self.assertEqual(new.commands.base_velocity.class_type.stair_speed_schedule[-1][1], 1.5)


if __name__ == '__main__':
    unittest.main()
