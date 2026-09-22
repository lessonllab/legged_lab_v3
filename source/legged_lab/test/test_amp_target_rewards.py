"""Regression coverage for target-turning reward conflicts."""

from types import SimpleNamespace as NS
import unittest

import torch
from isaaclab.managers import SceneEntityCfg

from legged_lab.tasks.locomotion.amp.mdp.rewards import (
    stand_still_joint_deviation_l1,
    target_heading_error,
)


class TestTargetRewards(unittest.TestCase):
    def make_env(self, commands):
        commands = torch.tensor(commands, dtype=torch.float32)
        count = len(commands)
        # Exercise Isaac Lab's actual joint-deviation reward, including selected
        # joint IDs, using the same .torch accessor as its live asset data.
        data = NS(
            joint_pos=NS(torch=torch.tensor([[.2, -.3, .8]]).repeat(count, 1)),
            default_joint_pos=NS(torch=torch.zeros(count, 3)),
        )
        return NS(
            command_manager=NS(get_command=lambda _: commands),
            scene={"robot": NS(data=data)},
        )

    def test_only_stationary_command_penalizes_joint_offsets(self):
        env = self.make_env([
            [0., 0., 0.], [0., 0., .5], [0., 0., -.5],
            [.2, 0., 0.], [0., -.2, 0.], [.1, .1, .1],
        ])
        penalty = stand_still_joint_deviation_l1(
            env, "base_velocity", command_threshold=.15, angular_threshold=.15,
            asset_cfg=SceneEntityCfg("robot", joint_ids=[0, 1]),
        )
        torch.testing.assert_close(penalty, torch.tensor([.5, 0., 0., 0., 0., .5]))

    def test_thresholds_apply_to_xy_norm_and_absolute_yaw(self):
        env = self.make_env([
            [.12, .12, 0.],  # each axis is small, but total translation is not
            [0., 0., .149], [0., 0., -.149],
            [0., 0., .15], [0., 0., -.15], [.15, 0., 0.],
        ])
        penalty = stand_still_joint_deviation_l1(
            env, "base_velocity", command_threshold=.15, angular_threshold=.15,
        )
        torch.testing.assert_close(penalty, torch.tensor([0., 1.3, 1.3, 0., 0., 0.]))

    def test_existing_positional_asset_argument_is_preserved(self):
        env = self.make_env([[0., 0., 0.], [.061, 0., 0.], [0., 0., -.5]])
        penalty = stand_still_joint_deviation_l1(
            env, "base_velocity", .06, SceneEntityCfg("robot", joint_ids=[0]),
        )
        torch.testing.assert_close(penalty, torch.tensor([.2, 0., 0.]))

    def test_target_heading_cost_is_symmetric_and_zero_when_aligned(self):
        env = self.make_env([[.6, 0., 0.], [0., 0., .4], [0., 0., -.4], [0., 0., 1.]])
        torch.testing.assert_close(target_heading_error(env, "base_velocity"), torch.tensor([0., .4, .4, 1.]))


if __name__ == "__main__":
    unittest.main()
