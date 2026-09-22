"""Picking, selected-robot ownership, and command-resampling regressions."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts' / 'rsl_rl'))
from viser_targets import TerrainRayPicker, ClickTargetControl
import test_amp_target_command as fixtures


def picker():
    result = TerrainRayPicker()
    # Two overlapping surfaces: the upper one slopes from Z=1 to Z=2.
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    result.add_mesh(np.array([[0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [0., 2., 0.]]), faces)
    result.add_mesh(np.array([[0., 0., 1.], [2., 0., 2.], [2., 2., 2.], [0., 2., 1.]]), faces)
    return result


class TestManualTargets(unittest.TestCase):
    def setUp(self):
        self.env, self.command = fixtures.TestTargetCommand().make_active_command()

    def test_nearest_terrain_surface_and_misses(self):
        mesh = picker()
        np.testing.assert_allclose(mesh.pick([1., 1., 5.], [0., 0., -2.]), [1., 1., 1.5])
        self.assertIsNone(mesh.pick([3., 1., 5.], [0., 0., -1.]))
        self.assertIsNone(mesh.pick([1., 1., 5.], [0., 0., 1.]))
        self.assertIsNone(mesh.pick([1., 1., 5.], [0., 0., 0.]))
        self.assertIsNone(mesh.pick([np.nan, 1., 5.], [0., 0., -1.]))
        # Front and back faces both work, including translated world coordinates.
        mesh = TerrainRayPicker()
        mesh.add_mesh(np.array([[100., 80., 3.], [102., 80., 3.], [100., 82., 3.]]), np.array([[2, 1, 0]]))
        np.testing.assert_allclose(mesh.pick([100.5, 80.5, 5.], [0., 0., -1.]), [100.5, 80.5, 3.])

    def test_only_selected_changes_and_timer_cannot_overwrite(self):
        cmd = self.command
        other = {name: getattr(cmd, name)[1].clone() for name in
                 ('pos_command_w', 'vel_command_b', 'speed_cap', 'target_initial_distance', 'time_left')}
        cmd.is_standing_env[0] = True
        cmd.set_manual_target(0, [1., 1., 1.5])
        for name, before in other.items():
            torch.testing.assert_close(getattr(cmd, name)[1], before)
        self.assertFalse(cmd.is_standing_env[0])
        for _ in range(3):
            cmd.time_left[:] = 0.
            cmd.compute(.02)
            torch.testing.assert_close(cmd.pos_command_w[0], torch.tensor([1., 1., 1.5]))
            self.assertFalse(cmd.manual_target[1])
            self.assertGreater(cmd.time_left[1], 0.)
        self.env.scene['robot'].data.root_pos_w[0] = cmd.pos_command_w[0]
        cmd._update_command()
        self.assertTrue((cmd.command[0] == 0).all())
        self.assertTrue(cmd.manual_target[0])

    def test_release_and_partial_reset(self):
        cmd = self.command
        cmd.set_manual_target(0, [1., 1., 1.5])
        cmd.reset([1])
        self.assertTrue(cmd.manual_target[0])
        cmd.release_manual_target(0)
        self.assertFalse(cmd.manual_target.any())
        torch.testing.assert_close(cmd.pos_command_w[0], torch.tensor([2., 0., 0.]))
        cmd.set_manual_target(0, [1., 1., 1.5])
        cmd.reset([0])
        self.assertFalse(cmd.manual_target[0])

    def test_queue_selection_race_and_switch_release(self):
        control = ClickTargetControl(self.command, picker())
        control.click(0, [1., 1., 5.], [0., 0., -1.])
        self.assertFalse(self.command.manual_target.any())  # callback never writes tensors
        self.assertTrue(control.apply(0))
        self.assertTrue(self.command.manual_target[0])
        control.click(0, [1., 1., 5.], [0., 0., -1.])
        control.apply(1)  # stale click must not control either robot after switching
        self.assertFalse(self.command.manual_target.any())
        control.click(1, [1., 1., 5.], [0., 0., -1.])
        control.apply(1)
        self.assertEqual(self.command.manual_target.tolist(), [False, True])
        control.apply(1, enabled=False)
        self.assertFalse(self.command.manual_target.any())

    def test_invalid_click_preserves_goal_and_random_button(self):
        control = ClickTargetControl(self.command, picker())
        control.click(0, [1., 1., 5.], [0., 0., -1.])
        control.apply(0)
        point = self.command.pos_command_w.clone()
        control.click(0, [3., 1., 5.], [0., 0., -1.])
        control.apply(0)
        torch.testing.assert_close(self.command.pos_command_w, point)
        control.release(0)
        control.apply(0)
        self.assertFalse(self.command.manual_target.any())
        for invalid in ([float('nan'), 0., 0.], [1., 2.]):
            with self.assertRaises(ValueError):
                self.command.set_manual_target(0, invalid)


if __name__ == '__main__':
    unittest.main()
