"""Reference-reset command regression tests; no simulator or robot is started.

Run with an Isaac Lab Python interpreter. Only the simulation-dependent base
class is replaced; the command implementation and quaternion math are real.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import isaaclab.utils.math as math_utils


def _load_command_class():
    base_module = ModuleType("isaaclab.envs.mdp.commands.velocity_command")
    base_module.UniformVelocityCommand = object
    path = (
        Path(__file__).parents[1]
        / "legged_lab/tasks/locomotion/amp/mdp/commands/velocity_command.py"
    )
    spec = importlib.util.spec_from_file_location("_amp_command_bounds_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {base_module.__name__: base_module}):
        spec.loader.exec_module(module)
    return module.AmpVelocityCommand


class TestReferenceCommandBounds(unittest.TestCase):
    def _make_command(self, heading=False):
        zeros = torch.zeros(3)
        quat = math_utils.quat_from_euler_xyz(zeros, zeros, zeros)
        linear = torch.tensor([[8.0, -3.0, 0.0], [-5.0, 4.0, 0.0], [0.7, 0.2, 0.0]])
        angular = torch.tensor([[0.0, 0.0, 9.0], [0.0, 0.0, -8.0], [0.0, 0.0, 0.4]])
        animation = SimpleNamespace(
            get_root_quat=lambda ids: quat[ids, None],
            get_root_vel_w=lambda ids: linear[ids, None],
            get_root_ang_vel_w=lambda ids: angular[ids, None],
        )
        command = _load_command_class()()
        command._env = SimpleNamespace(animation_manager=SimpleNamespace(get_term=lambda name: animation))
        command.cfg = SimpleNamespace(
            animation="animation",
            heading_command=heading,
            reset_heading_lookahead=0.5,
            ranges=SimpleNamespace(lin_vel_x=(0.0, 3.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.5, 1.5)),
        )
        command.vel_command_b = torch.full((3, 3), 42.0)
        command.heading_target = torch.zeros(3)
        command.is_heading_env = torch.zeros(3, dtype=torch.bool)
        command.is_standing_env = torch.ones(3, dtype=torch.bool)
        return command

    def test_direct_reset_bounds_outliers_and_preserves_valid_commands(self):
        command = self._make_command()
        command._align_to_reference(torch.arange(3))
        torch.testing.assert_close(
            command.vel_command_b,
            torch.tensor([[3.0, -0.5, 1.5], [0.0, 0.5, -1.5], [0.7, 0.2, 0.4]]),
        )
        self.assertFalse(command.is_standing_env.any())
        self.assertFalse(command.is_heading_env.any())

    def test_partial_reset_does_not_change_other_environments(self):
        command = self._make_command()
        command._align_to_reference(torch.tensor([1]))
        torch.testing.assert_close(command.vel_command_b[[0, 2]], torch.full((2, 3), 42.0))
        torch.testing.assert_close(command.vel_command_b[1], torch.tensor([0.0, 0.5, -1.5]))
        self.assertTrue(command.is_standing_env[0])

    def test_heading_reset_keeps_bounded_lookahead(self):
        command = self._make_command(heading=True)
        command._align_to_reference(torch.arange(3))
        torch.testing.assert_close(command.heading_target, torch.tensor([0.75, -0.75, 0.2]))
        self.assertTrue(command.is_heading_env.all())


if __name__ == "__main__":
    unittest.main()
