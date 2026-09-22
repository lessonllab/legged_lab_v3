import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_scratch_env_cfg import G1AmpScratchEnvCfg, entry_terrain
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_target_v7_env_cfg import G1AmpDepthTargetV7EnvCfg
from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_scratch_ppo_cfg import G1AmpScratchPPORunnerCfg
from legged_lab.tasks.locomotion.amp.mdp.scratch_curriculum import stage_speed_ranges, update_streak
from legged_lab.tasks.locomotion.amp.mdp.commands.target_velocity import terrain_column_names
from legged_lab.rsl_rl.amp.scratch_ppo import ScratchPPOAMP, course_restore_indices
from legged_lab.rsl_rl.amp.ppo_amp import PPOAMP


class TestScratchCourse(unittest.TestCase):
    def test_smaller_playback_preserves_terrain_column_assignment(self):
        columns = (torch.arange(2048) / (2048 / 10)).long()
        torch.testing.assert_close(course_restore_indices(2048, columns, 10), torch.arange(2048))
        selected = course_restore_indices(2048, torch.arange(10), 10)
        torch.testing.assert_close(columns[selected], torch.arange(10))

    def test_reference_and_network_inputs_unchanged(self):
        old, new = G1AmpDepthTargetV7EnvCfg(), G1AmpScratchEnvCfg()
        new.validate()
        self.assertEqual(old.motion_data.to_dict(), new.motion_data.to_dict())
        self.assertEqual(old.animation.to_dict(), new.animation.to_dict())
        for name in ('policy', 'critic', 'depth', 'disc', 'disc_demo'):
            self.assertEqual(getattr(old.observations, name).to_dict(), getattr(new.observations, name).to_dict())
        self.assertEqual(new.rewards.is_alive.weight, 3.)
        self.assertEqual(new.scene.terrain.max_init_terrain_level, 0)
        generator = new.scene.terrain.terrain_generator
        self.assertEqual(set(terrain_column_names(generator)), set(generator.sub_terrains))
        runner = G1AmpScratchPPORunnerCfg()
        self.assertEqual(runner.algorithm.amp_cfg.amp_discriminator.style_reward_scale, .5)
        self.assertIsNone(runner.algorithm.amp_cfg.style_reward_gate_group)
        self.assertEqual(runner.algorithm.schedule, 'adaptive')

    def test_first_row_really_flat_and_next_row_rough(self):
        cfg = G1AmpScratchEnvCfg().scene.terrain.terrain_generator
        terrain = cfg.sub_terrains['pyramid_stairs']
        terrain.size = cfg.size
        for d in (0., .05, .099999):
            meshes, origin = entry_terrain(d, terrain)
            self.assertTrue(all(np.allclose(m.vertices[:, 2], 0.) for m in meshes))
            self.assertEqual(origin[2], 0.)
        meshes, origin = entry_terrain(.10001, terrain)
        self.assertGreater(max(m.vertices[:, 2].max() for m in meshes), 0.)

    def test_speed_ranges_restore_fast_reference_commands(self):
        final = torch.tensor([[.45, 2.]] * 5)
        limits = stage_speed_ranges(torch.tensor([0, 1, 2, 3, 9]), final)
        torch.testing.assert_close(limits[:, 0], torch.full((5,), .45))
        self.assertAlmostEqual(limits[0, 1].item(), .6)
        torch.testing.assert_close(limits[3:, 1], torch.tensor([2., 2.]))
        stairs = stage_speed_ranges(torch.tensor([3]), torch.tensor([[.45, .8]]))
        self.assertAlmostEqual(stairs[0, 1].item(), .8)

    def test_standing_falling_and_initial_reset_cannot_promote(self):
        streak = torch.full((5,), 2)
        updated, up, down = update_streak(streak, torch.ones(5), torch.ones(5),
            torch.tensor([0., 2., 2., 2., 2.]), torch.full((5,), 4.),
            torch.ones(5, dtype=torch.bool), torch.tensor([False, True, False, False, False]),
            torch.tensor([True, True, False, True, True]), 10, torch.tensor([0, 0, 0, 0, 9]))
        self.assertEqual(up.tolist(), [False, False, False, True, False])
        self.assertTrue(down[1])
        self.assertEqual(updated[3].item(), 0)

    def test_checkpoint_restores_course_and_skips_reset_promotion(self):
        cmd = NS(success_streak=torch.tensor([2, 1]), skip_curriculum_once=torch.zeros(2, dtype=torch.bool),
                 column_names=['stairs', 'rough'])
        terrain = NS(terrain_levels=torch.tensor([1, 3]), max_terrain_level=10,
                     terrain_types=torch.tensor([0, 1]), terrain_origins=torch.randn(10, 2, 3),
                     env_origins=torch.zeros(2, 3))
        env = NS(command_manager=NS(get_term=lambda name: cmd), scene=NS(terrain=terrain), device='cpu')
        def reset():
            self.assertTrue(cmd.skip_curriculum_once.all())
            cmd.success_streak.zero_()
        env.reset = reset
        alg = object.__new__(ScratchPPOAMP)
        alg.course_env = env
        with patch.object(PPOAMP, 'save', return_value={}):
            state = alg.save()
        terrain.terrain_levels.zero_()
        cmd.success_streak.zero_()
        with patch.object(PPOAMP, 'load', return_value=True):
            alg.load(state)
        torch.testing.assert_close(terrain.terrain_levels, torch.tensor([1, 3]))
        torch.testing.assert_close(cmd.success_streak, torch.tensor([2, 1]))
        torch.testing.assert_close(terrain.env_origins, terrain.terrain_origins[[1, 3], [0, 1]])


if __name__ == '__main__':
    unittest.main()
