"""Replay validity, fixed exploration and warm-up reward regression tests."""

import copy
import importlib.metadata
from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

from legged_lab.rsl_rl.amp.circular_buffer import CircularBuffer
from legged_lab.rsl_rl.amp.ppo_amp import PPOAMP
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_env_cfg import G1AmpDepthEnvCfg
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_warmup_env_cfg import G1AmpDepthWarmupEnvCfg
from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_depth_warmup_ppo_cfg import G1AmpDepthWarmupPPORunnerCfg
from legged_lab.tasks.locomotion.amp.mdp.rewards import dont_wait


class TestReplay(unittest.TestCase):
    def test_recent_and_history_mix_after_wraparound(self):
        buffer = CircularBuffer(6, 2, "cpu")
        for step in range(10):
            buffer.append(torch.full((2, 1), float(step)))
        samples = torch.cat(list(buffer.mini_batch_generator(2, 2, 1, recent_length=2)))
        self.assertEqual(int((samples >= 8).sum()), 2)
        self.assertEqual(int((samples < 8).sum()), 2)
        self.assertTrue(torch.all((samples >= 4) & (samples <= 9)))

    def test_partial_reset_never_samples_stale_frames(self):
        buffer = CircularBuffer(8, 2, "cpu")
        for step in range(7):
            buffer.append(torch.full((2, 1), float(step)))
        buffer.reset([1])
        for step in (100., 101.):
            buffer.append(torch.full((2, 1), step))
        samples = torch.cat(list(buffer.mini_batch_generator(2, 2, 1, recent_length=2)))
        self.assertEqual(sorted(samples.flatten().tolist()), [100., 100., 101., 101.])

    def test_empty_cold_start_and_fraction_validation(self):
        buffer = CircularBuffer(10, 2, "cpu")
        with self.assertRaises(RuntimeError):
            next(buffer.mini_batch_generator(2, 1))
        for step in (10., 11.):
            buffer.append(torch.full((2, 1), step))
        samples = torch.cat(list(buffer.mini_batch_generator(2, 1, 1, recent_length=2)))
        self.assertEqual(sorted(samples.flatten().tolist()), [10., 10., 11., 11.])
        with self.assertRaises(ValueError):
            next(buffer.mini_batch_generator(2, 1, recent_fraction=1.1))
        buffer.reset()
        with self.assertRaises(RuntimeError):
            next(buffer.mini_batch_generator(2, 1))


class TestWarmup(unittest.TestCase):
    def test_dont_wait_respects_command_grace_and_backward_motion(self):
        commands = torch.tensor([[.5, 0., 0.]] * 6)
        commands[4] = 0
        velocity = torch.tensor([[.5, 0., 0.], [.05, 0., 0.], [-.05, 0., 0.],
                                 [-.2, 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=SimpleNamespace(torch=velocity)))},
            command_manager=SimpleNamespace(get_command=lambda name: commands),
            episode_length_buf=torch.tensor([25, 25, 25, 25, 25, 24]), step_dt=.02,
        )
        torch.testing.assert_close(dont_wait(env, "base_velocity"), torch.tensor([0., 1., 2., 3., 0., 0.]))

    def test_config_keeps_visual_inputs_and_does_not_modify_rough(self):
        warmup = G1AmpDepthWarmupEnvCfg()
        warmup.validate()
        rough = G1AmpDepthEnvCfg()
        self.assertEqual(warmup.scene.terrain.terrain_type, "plane")
        self.assertIsNone(warmup.curriculum.terrain_levels)
        self.assertIsNotNone(warmup.scene.depth_camera)
        self.assertIsNotNone(warmup.observations.critic.height_scan)
        self.assertIsNone(warmup.observations.policy.height_scan)
        self.assertEqual(warmup.commands.base_velocity.ranges.lin_vel_x, (.5, .5))
        self.assertEqual(rough.rewards.feet_air_time.weight, .75)
        self.assertEqual(rough.rewards.dof_torques_l2.weight, -2e-6)
        self.assertFalse(hasattr(rough.rewards, "dont_wait"))
        self.assertEqual(rough.scene.terrain.terrain_type, "generator")

    def test_ppo_updates_fixed_std_online_reward_logs_and_resume(self):
        torch.manual_seed(7)
        config = handle_deprecated_rsl_rl_cfg(
            G1AmpDepthWarmupPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib")
        ).to_dict()
        config["multi_gpu"] = None
        config["num_steps_per_env"] = 2
        config["algorithm"]["num_learning_epochs"] = 1
        config["algorithm"]["num_mini_batches"] = 1
        config["algorithm"]["amp_cfg"]["disc_update_interval"] = 1
        obs = TensorDict({"policy": torch.randn(2, 495), "depth": torch.rand(2, 8, 36, 64),
                          "critic": torch.randn(2, 787), "disc": torch.randn(2, 4, 61),
                          "disc_demo": torch.randn(2, 4, 61)}, batch_size=[2])
        env = SimpleNamespace(num_envs=2, num_actions=29, unwrapped=SimpleNamespace(step_dt=.02))
        alg = PPOAMP.construct_algorithm(obs, env, copy.deepcopy(config), "cpu")
        self.assertEqual(alg.disc_obs_buffer.max_length, 20)
        initial_std = alg.actor.distribution.std_param.clone()
        self.assertFalse(alg.actor.distribution.std_param.requires_grad)
        before = next(alg.actor.cnns.parameters()).clone()
        logs = []
        for _ in range(2):
            with torch.inference_mode():
                for step in range(2):
                    alg.act(obs)
                    extras = {"log": {"existing": torch.tensor(1.)}}
                    alg.process_env_step(obs, torch.tensor([1., 2.]), torch.zeros(2, dtype=torch.bool), extras)
                    logs.append(extras["log"])
                    self.assertIn("existing", extras["log"])
                    torch.testing.assert_close(extras["log"]["AMP/task_reward_per_step"], torch.tensor(1.5))
                    torch.testing.assert_close(extras["log"]["AMP/mixed_reward_per_step"],
                                               .5 * (torch.tensor(1.5) + extras["log"]["AMP/style_reward_per_step"]))
                alg.compute_returns(obs)
            losses = alg.update()
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
        self.assertIsNot(logs[0], logs[1])
        torch.testing.assert_close(initial_std, alg.actor.distribution.std_param)
        self.assertFalse(torch.equal(before, next(alg.actor.cnns.parameters())))
        self.assertEqual(int(alg.disc_obs_buffer.current_length.min()), 4)
        saved = copy.deepcopy(alg.save())
        alg.load(saved, None, True)
        self.assertEqual(int(alg.disc_obs_buffer.current_length.max()), 0)
        with torch.inference_mode():
            for _ in range(2):
                alg.act(obs)
                alg.process_env_step(obs, torch.ones(2), torch.zeros(2, dtype=torch.bool), {})
            alg.compute_returns(obs)
        alg.update()  # Resume uses a fresh, valid rollout until history is refilled.
        torch.testing.assert_close(initial_std, alg.actor.distribution.std_param)


if __name__ == "__main__":
    unittest.main()
