"""Versioned v6 task and root-independent posture termination regressions."""
import copy
import importlib.metadata
from types import SimpleNamespace as NS
import unittest
import gymnasium as gym
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_target_v5_env_cfg import G1AmpDepthTargetV5EnvCfg
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_target_v6_env_cfg import (
    G1AmpDepthTargetV6EnvCfg, G1AmpDepthTargetV6EnvCfg_PLAY,
)
from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_depth_target_v6_ppo_cfg import G1AmpDepthTargetV6PPORunnerCfg
from legged_lab.tasks.locomotion.amp.mdp.terminations import body_tilt_exceeds


class TestTargetV6(unittest.TestCase):
    def test_v5_isolation_and_v6_inputs_gates(self):
        old = G1AmpDepthTargetV5EnvCfg()
        before = copy.deepcopy(old.to_dict())
        new, play = G1AmpDepthTargetV6EnvCfg(), G1AmpDepthTargetV6EnvCfg_PLAY()
        new.validate(); play.validate()
        self.assertEqual(old.to_dict(), before)
        self.assertEqual(new.observations.to_dict(), old.observations.to_dict())
        self.assertEqual(new.motion_data.to_dict(), old.motion_data.to_dict())
        self.assertEqual(new.curriculum.to_dict(), old.curriculum.to_dict())
        self.assertEqual(new.scene.terrain.to_dict(), old.scene.terrain.to_dict())
        self.assertEqual(new.terminations.base_height.params['minimum_height'], .35)
        self.assertEqual(old.terminations.base_height.params['minimum_height'], .2)
        self.assertEqual(new.rewards.feet_air_time.func.__name__, 'instinct_feet_air_time')
        self.assertNotIn('threshold', new.rewards.feet_air_time.params)
        self.assertEqual(new.rewards.upright_body.params['asset_cfg'].body_names, ['pelvis', 'torso_link'])
        self.assertIsNone(new.rewards.flat_orientation_l2)
        self.assertIsNone(new.rewards.joint_deviation_hip)
        self.assertIsNone(new.rewards.joint_deviation_arms)
        self.assertIsNone(new.rewards.joint_deviation_waist)
        self.assertIsNotNone(new.rewards.stand_still)
        self.assertIsNone(play.curriculum.terrain_levels)
        for suffix in ('', '-Play'):
            entry = gym.spec(f'LeggedLab-Isaac-AMP-Depth-Target-G1{suffix}-v2')
            self.assertIn('target_v6_env_cfg', entry.kwargs['env_cfg_entry_point'])
        runner = G1AmpDepthTargetV6PPORunnerCfg()
        runner = handle_deprecated_rsl_rl_cfg(runner, importlib.metadata.version('rsl-rl-lib'))
        # Modern RSL retains MISSING placeholders for deprecated runner fields.
        # Validate the AMP subtree actually consumed by the converted algorithm.
        runner.algorithm.amp_cfg.validate()
        self.assertFalse(runner.resume)
        amp = runner.algorithm.amp_cfg
        self.assertEqual(amp.amp_discriminator.reward_combination, 'additive')
        self.assertEqual(amp.amp_discriminator.style_reward_scale, .25)
        self.assertFalse(amp.amp_discriminator.style_reward_time_scaled)
        self.assertEqual(amp.amp_discriminator.observation_normalization, 'none')
        self.assertEqual(amp.grad_penalty_data, 'agent_demo')
        self.assertEqual(amp.disc_update_interval, 1)
        self.assertEqual(amp.disc_replay_rollouts, 1)

    def test_tilt_uses_selected_bodies_and_catches_inversion(self):
        # Three physical links; link 2 is unselected and may freely rotate.
        pitch = torch.tensor([[0., 0., 3.14], [.2, 1.1, 0.], [3.14, 0., 0.], [.99, .99, 0.]])
        quats = quat_from_euler_xyz(torch.zeros_like(pitch), pitch, torch.zeros_like(pitch))
        env = NS(scene={'robot': NS(data=NS(body_quat_w=NS(torch=quats)))})
        selected = SceneEntityCfg('robot', body_ids=[0, 1])
        self.assertEqual(body_tilt_exceeds(env, 1., selected).tolist(), [False, True, True, False])
        with self.assertRaises(ValueError):
            body_tilt_exceeds(env, 0., selected)


if __name__ == '__main__':
    unittest.main()
