"""v5 configuration and actual PPO exploration-update regression checks."""
import copy
import importlib.metadata
from types import SimpleNamespace as NS
import unittest

import gymnasium as gym
import torch
from tensordict import TensorDict
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from legged_lab.rsl_rl.amp.ppo_amp import PPOAMP
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_target_v5_env_cfg import (
    G1AmpDepthTargetV5EnvCfg, G1AmpDepthTargetV5EnvCfg_PLAY,
)
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_target_env_cfg import G1AmpDepthTargetEnvCfg
from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_depth_target_v5_ppo_cfg import G1AmpDepthTargetV5PPORunnerCfg
from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_depth_target_ppo_cfg import G1AmpDepthTargetPPORunnerCfg


class TestTargetV5(unittest.TestCase):
    def test_registration_and_independent_configuration(self):
        old = G1AmpDepthTargetEnvCfg()
        before = copy.deepcopy(old.to_dict())
        new, play = G1AmpDepthTargetV5EnvCfg(), G1AmpDepthTargetV5EnvCfg_PLAY()
        new.validate(); play.validate()
        self.assertEqual(old.to_dict(), before)
        self.assertEqual(new.observations.to_dict(), old.observations.to_dict())
        self.assertEqual(new.motion_data.to_dict(), old.motion_data.to_dict())
        self.assertIsNotNone(new.scene.depth_camera)
        self.assertEqual(len(new.scene.terrain.terrain_generator.sub_terrains), 6)
        self.assertIsNone(new.observations.policy.height_scan)
        self.assertIsNone(new.observations.critic.height_scan)
        self.assertEqual(new.scene.terrain.max_init_terrain_level, 0)
        self.assertIsNone(new.events.reset_from_ref)
        self.assertEqual(new.events.reset_robot_joints.params['position_range'], (-.15, .15))
        self.assertEqual(play.events.reset_robot_joints.params['position_range'], (0., 0.))
        self.assertIsNone(play.curriculum.terrain_levels)
        self.assertEqual(new.rewards.heading_error.weight, -1.)
        for name in ('joint_deviation_hip', 'joint_deviation_arms', 'joint_deviation_waist'):
            self.assertEqual(getattr(new.rewards, name).params['angular_threshold'], .15)
        for suffix in ('', '-Play'):
            spec = gym.spec(f'LeggedLab-Isaac-AMP-Depth-Target-G1{suffix}-v1')
            self.assertIn('target_v5_env_cfg', spec.kwargs['env_cfg_entry_point'])

    def test_ppo_learns_std_and_roundtrips_checkpoint(self):
        torch.manual_seed(17)
        runner = G1AmpDepthTargetV5PPORunnerCfg()
        self.assertFalse(runner.resume)
        self.assertNotEqual(runner.experiment_name, G1AmpDepthTargetPPORunnerCfg().experiment_name)
        cfg = handle_deprecated_rsl_rl_cfg(runner, importlib.metadata.version('rsl-rl-lib')).to_dict()
        cfg.update(multi_gpu=None, num_steps_per_env=4)
        cfg['algorithm'].update(num_learning_epochs=1, num_mini_batches=1)
        cfg['algorithm']['amp_cfg']['disc_update_interval'] = 1
        obs = TensorDict({
            'policy': torch.randn(4, 495), 'critic': torch.randn(4, 600),
            'depth': torch.rand(4, 8, 18, 32),
            'disc': torch.randn(4, 10, 67), 'disc_demo': torch.randn(4, 10, 67),
        }, batch_size=[4])
        env = NS(num_envs=4, num_actions=29, unwrapped=NS(step_dt=.02))
        alg = PPOAMP.construct_algorithm(obs, env, copy.deepcopy(cfg), 'cpu')
        param = alg.actor.distribution.log_std_param
        self.assertTrue(param.requires_grad)
        initial = param.detach().clone()
        with torch.inference_mode():
            for _ in range(4):
                alg.act(obs)
                alg.process_env_step(obs, torch.tensor([.1, .4, .7, 1.]), torch.zeros(4, dtype=torch.bool), {})
            alg.compute_returns(obs)
        losses = alg.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
        self.assertFalse(torch.equal(initial, param))
        self.assertTrue(torch.isfinite(param).all())
        saved = copy.deepcopy(alg.save())
        restored = PPOAMP.construct_algorithm(obs, env, copy.deepcopy(cfg), 'cpu')
        restored.load(saved, None, True)
        torch.testing.assert_close(restored.actor.distribution.log_std_param, param)


if __name__ == '__main__':
    unittest.main()
