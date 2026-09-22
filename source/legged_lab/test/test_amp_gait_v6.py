"""v6 gait rewards: turn stepping, whole-body standing, explicit-body tilt."""

from types import SimpleNamespace as NS
import unittest

import torch
from isaaclab.managers import SceneEntityCfg

from legged_lab.tasks.locomotion.amp import mdp
from legged_lab.tasks.locomotion.amp.mdp.rewards import (
    body_orientation_l2,
    instinct_feet_air_time,
    stand_still,
)


class TestGaitV6(unittest.TestCase):
    def feet_env(self, commands, contact_times=None, air_times=None, proxy=True):
        commands = torch.tensor(commands, dtype=torch.float32)
        count = len(commands)
        # Selected feet are IDs 2 and 0. ID 1 is an unrelated body touching the
        # ground, which must not turn valid single support into double support.
        air = torch.tensor([[.0, .0, .3]]).repeat(count, 1) if air_times is None else torch.tensor(air_times)
        contact = torch.tensor([[.2, 2., .0]]).repeat(count, 1) if contact_times is None else torch.tensor(contact_times)
        wrap = (lambda tensor: NS(torch=tensor)) if proxy else (lambda tensor: tensor)
        sensor = NS(data=NS(current_air_time=wrap(air), current_contact_time=wrap(contact)))
        return NS(
            command_manager=NS(get_command=lambda _: commands),
            scene=NS(sensors={"contact_forces": sensor}),
        )

    def feet_reward(self, env):
        return instinct_feet_air_time(
            env, "base_velocity", .15,
            SceneEntityCfg("contact_forces", body_ids=[2, 0]),
        )

    def test_translation_and_both_turn_directions_allow_steps(self):
        commands = [[0., 0., 1.], [0., 0., -1.], [.6, 0., 0.], [0., -.6, 0.], [0., 0., 0.]]
        expected = torch.tensor([.2, .2, .2, .2, 0.])
        torch.testing.assert_close(self.feet_reward(self.feet_env(commands)), expected)
        torch.testing.assert_close(self.feet_reward(self.feet_env(commands, proxy=False)), expected)

    def test_double_support_and_flight_receive_no_air_time_bonus(self):
        env = self.feet_env(
            [[0., 0., 1.]] * 4,
            contact_times=[[.2, 2., .1], [0., 2., 0.], [0., 2., .4], [.6, 2., 0.]],
            air_times=[[0., 0., 0.], [.2, 0., .3], [.3, 0., 0.], [0., 0., .4]],
        )
        torch.testing.assert_close(self.feet_reward(env), torch.tensor([0., 0., .3, .4]))

    def test_step_threshold_uses_translation_norm_or_absolute_yaw(self):
        env = self.feet_env([
            [0., 0., .15], [0., 0., -.15], [.15, 0., 0.],
            [0., 0., -.151], [.11, .11, 0.],
        ])
        torch.testing.assert_close(self.feet_reward(env), torch.tensor([0., 0., 0., .2, .2]))

    def standing_env(self, commands, proxy=True):
        commands = torch.tensor(commands, dtype=torch.float32)
        count = len(commands)
        pos = torch.tensor([[.3, -.2, 1.]]).repeat(count, 1)
        default = torch.tensor([[.1, .1, .2]]).repeat(count, 1)
        wrap = (lambda tensor: NS(torch=tensor)) if proxy else (lambda tensor: tensor)
        return NS(
            command_manager=NS(get_command=lambda _: commands),
            scene={"robot": NS(data=NS(joint_pos=wrap(pos), default_joint_pos=wrap(default)))},
        )

    def test_standing_offset_and_selected_joint_deviation(self):
        for proxy in (True, False):
            env = self.standing_env([[0., 0., 0.]], proxy=proxy)
            all_joints = stand_still(env, "base_velocity")
            selected = stand_still(env, "base_velocity", SceneEntityCfg("robot", joint_ids=[2, 0]))
            torch.testing.assert_close(all_joints, torch.tensor([-2.7]))
            torch.testing.assert_close(selected, torch.tensor([-3.]))
            # A negative configured weight turns the negative result into a
            # standing bonus; the offset must not be clamped to zero.
            self.assertGreater((-selected).item(), 0.)

    def test_whole_body_standing_gate_allows_translation_and_turning(self):
        env = self.standing_env([
            [0., 0., 0.], [0., 0., .5], [0., 0., -.5], [.2, 0., 0.],
            [.12, .12, 0.], [0., 0., .15], [0., 0., -.15], [.15, 0., 0.],
            [.1, .1, .149],
        ])
        torch.testing.assert_close(
            stand_still(env, "base_velocity"),
            torch.tensor([-2.7, 0., 0., 0., 0., 0., 0., 0., -2.7]),
        )

    def test_orientation_reads_selected_body_frames_not_root(self):
        identity = [0., 0., 0., 1.]
        pitch90 = [0., 2**-.5, 0., 2**-.5]
        yaw90 = [0., 0., 2**-.5, 2**-.5]
        quats = torch.tensor([
            [pitch90, identity, yaw90],
            [identity, pitch90, pitch90],
        ])
        for proxy in (True, False):
            wrap = (lambda tensor: NS(torch=tensor)) if proxy else (lambda tensor: tensor)
            data = NS(body_quat_w=wrap(quats), GRAVITY_VEC_W=wrap(torch.tensor([[0., 0., -1.]]).repeat(2, 1)))
            env = NS(scene={"robot": NS(data=data)})
            torch.testing.assert_close(
                body_orientation_l2(env, SceneEntityCfg("robot", body_ids=[1])),
                torch.tensor([0., 1.]),
            )
            torch.testing.assert_close(
                body_orientation_l2(env, SceneEntityCfg("robot", body_ids=[2, 0])),
                torch.tensor([1., 1.]),
            )
            torch.testing.assert_close(body_orientation_l2(env), torch.tensor([1., 2.]))

    def test_instinct_steps_do_not_override_legacy_feet_reward(self):
        self.assertIsNot(mdp.feet_air_time, instinct_feet_air_time)
        self.assertIs(mdp.instinct_feet_air_time, instinct_feet_air_time)


if __name__ == "__main__":
    unittest.main()
