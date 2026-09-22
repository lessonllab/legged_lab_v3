"""CPU regressions for optional additive AMP; no simulator or training run required."""

import copy
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

from legged_lab.rsl_rl.amp.discriminator import AMPDiscriminator
from legged_lab.rsl_rl.amp.ppo_amp import PPOAMP


GROUPS = {"actor": ["policy"], "critic": ["policy"],
          "discriminator": ["disc"], "discriminator_demonstration": ["disc_demo"]}


def discriminator(**kwargs):
    return AMPDiscriminator(2, 3, GROUPS, hidden_dims=[8], **kwargs)


def setup_algorithm(instinct=True):
    torch.manual_seed(21)
    obs = TensorDict({"policy": torch.randn(8, 5), "disc": torch.randn(8, 3, 2),
                      "disc_demo": torch.randn(8, 3, 2)}, batch_size=[8])
    amp = {"amp_discriminator": {"hidden_dims": [8], "style_reward_scale": 2.0,
                                  "task_style_lerp": .5}, "disc_replay_rollouts": 1}
    if instinct:
        amp.update(disc_optimizer="adamw", disc_optimizer_weight_decay=.01,
                   disc_weight_l2_coef=3e-4, disc_logit_l2_coef=.04,
                   grad_penalty_data="agent_demo", grad_penalty_scale=5.,
                   disc_learning_rate=1e-4, disc_max_grad_norm=None,
                   disc_update_interval=1, disc_replay_current_fraction=1.0)
        amp["amp_discriminator"].update(reward_combination="additive",
                                        style_reward_time_scaled=False,
                                        observation_normalization="none", style_reward_scale=.25)
    cfg = {
        "actor": {"class_name": "rsl_rl.models:MLPModel", "hidden_dims": [8],
                  "distribution_cfg": {"class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                                       "init_std": .3}},
        "critic": {"class_name": "rsl_rl.models:MLPModel", "hidden_dims": [8]},
        "algorithm": {"class_name": "legged_lab.rsl_rl.amp.ppo_amp:PPOAMP", "amp_cfg": amp,
                      "num_learning_epochs": 5, "num_mini_batches": 4, "schedule": "fixed"},
        "num_steps_per_env": 4, "obs_groups": copy.deepcopy(GROUPS), "multi_gpu": None,
    }
    env = SimpleNamespace(num_envs=8, num_actions=2, unwrapped=SimpleNamespace(step_dt=.02))
    return PPOAMP.construct_algorithm(obs, env, cfg, "cpu"), obs


def rollout_and_update(alg, obs):
    logs = []
    with torch.inference_mode():
        for _ in range(4):
            alg.act(obs)
            extras = {"log": {"existing": torch.tensor(7.)}}
            alg.process_env_step(obs, torch.ones(8), torch.zeros(8, dtype=torch.bool), extras)
            logs.append(extras["log"])
        alg.compute_returns(obs)
    return alg.update(), logs


class TestInstinctBalance(unittest.TestCase):
    def test_additive_known_scores_and_time_step_invariance(self):
        disc = discriminator(reward_combination="additive", style_reward_time_scaled=False,
                             observation_normalization="none", style_reward_scale=.25)
        scores = torch.tensor([[-2.], [-1.], [0.], [1.], [2.], [3.]])
        expected_raw = torch.tensor([0., 0., .75, 1., .75, 0.])
        obs = torch.zeros(6, 3, 2)
        with patch.object(disc, "forward", return_value=scores):
            for dt in (.01, .02, .05):
                style, actual_scores = disc.predict_style_reward(obs, dt)
                torch.testing.assert_close(actual_scores, scores.flatten())
                torch.testing.assert_close(disc.raw_style_rewards, expected_raw)
                torch.testing.assert_close(style, .25 * expected_raw)
                torch.testing.assert_close(disc.lerp_reward(torch.ones(6), style), 1 + .25 * expected_raw)

    def test_legacy_formula_state_keys_and_normalization_are_unchanged(self):
        disc = discriminator(style_reward_scale=2., task_style_lerp=.5)
        obs = torch.ones(2, 3, 2)
        with patch.object(disc, "forward", return_value=torch.zeros(2, 1)):
            style, _ = disc.predict_style_reward(obs, .02)
        torch.testing.assert_close(style, torch.full((2,), .03))
        torch.testing.assert_close(disc.lerp_reward(torch.ones(2), style), torch.full((2,), .515))
        self.assertEqual(set(disc.disc_obs_normalizer.state_dict()), {"_mean", "_var", "_std", "count"})
        disc.update_normalization(obs)
        self.assertEqual(int(disc.disc_obs_normalizer.count), 6)

    def test_identity_normalization_and_strict_checkpoint_mismatch(self):
        disc = discriminator(observation_normalization="none")
        obs = torch.randn(2, 3, 2)
        for training in (True, False):
            disc.train(training)
            disc.update_normalization(obs)
            torch.testing.assert_close(disc.normalize_disc_obs(obs), obs)
        self.assertEqual(disc.disc_obs_normalizer.state_dict(), {})
        with self.assertRaises(RuntimeError):
            disc.load_state_dict(discriminator().state_dict(), strict=True)

    def test_gradient_penalty_includes_both_populations(self):
        disc = discriminator()
        agent = torch.ones(2, 6)
        demo = torch.full((2, 6), 3.)
        coefficient = torch.nn.Parameter(torch.tensor(1.))
        def quadratic_forward(x):
            return coefficient * x.square().sum(dim=-1, keepdim=True)
        with patch.object(disc, "forward", side_effect=quadratic_forward):
            both = disc.compute_grad_penalty(demo, scale=1., agent_data=agent)
            demo_only = disc.compute_grad_penalty(demo, scale=1.)
            agent_only = disc.compute_grad_penalty(agent, scale=1.)
            changed_agent = disc.compute_grad_penalty(demo, scale=1., agent_data=agent * 2)
            changed_demo = disc.compute_grad_penalty(demo * 2, scale=1., agent_data=agent)
        torch.testing.assert_close(both, .5 * (demo_only + agent_only))
        self.assertGreater(changed_agent.item(), both.item())
        self.assertGreater(changed_demo.item(), both.item())
        both.backward()
        self.assertGreater(coefficient.grad.item(), 0.)
        self.assertIsNone(agent.grad)  # GP trains the discriminator, never the policy through its inputs.

    def test_explicit_regularization_gradient_and_logit_bias_exclusion(self):
        disc = discriminator()
        all_l2, logit_l2 = disc.weight_penalties()
        regularization = 3e-4 * all_l2 + .04 * logit_l2
        self.assertGreater(regularization.item(), 0.)
        regularization.backward()
        for name, param in disc.named_parameters():
            coefficient = 3e-4 + (.04 if name == "disc_linear.weight" else 0.)
            torch.testing.assert_close(param.grad, 2 * coefficient * param)

    def test_ppo_additive_update_logs_save_and_resume(self):
        alg, obs = setup_algorithm()
        self.assertIsInstance(alg.disc_optimizer, torch.optim.AdamW)
        self.assertEqual(alg.disc_optimizer.param_groups[0]["weight_decay"], .01)
        self.assertEqual(alg.disc_optimizer.param_groups[0]["lr"], 1e-4)
        before_actor = next(alg.actor.parameters()).detach().clone()
        before_disc = next(alg.amp_discriminator.parameters()).detach().clone()
        losses, logs = rollout_and_update(alg, obs)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))
        self.assertEqual(losses["amp/disc_updates"], 20.)
        self.assertGreater(losses["amp/disc_weight_l2"], 0.)
        self.assertGreater(losses["amp/disc_logit_l2"], 0.)
        self.assertFalse(torch.equal(before_actor, next(alg.actor.parameters())))
        self.assertFalse(torch.equal(before_disc, next(alg.amp_discriminator.parameters())))
        for log in logs:
            self.assertEqual(log["existing"], 7.)
            torch.testing.assert_close(log["AMP/weighted_task_reward_per_step"], torch.tensor(1.))
            torch.testing.assert_close(log["AMP/weighted_style_reward_per_step"],
                                       .25 * log["AMP/raw_style_reward_per_step"])
            torch.testing.assert_close(log["AMP/mixed_reward_per_step"],
                                       log["AMP/weighted_task_reward_per_step"] + log["AMP/weighted_style_reward_per_step"])
        serialized = io.BytesIO()
        torch.save(alg.save(), serialized)
        serialized.seek(0)
        checkpoint = torch.load(serialized, weights_only=False)
        resumed, _ = setup_algorithm()
        self.assertTrue(resumed.load(checkpoint, None, True))
        self.assertEqual(int(resumed.disc_obs_buffer.current_length.max()), 0)
        for key, value in alg.amp_discriminator.state_dict().items():
            torch.testing.assert_close(resumed.amp_discriminator.state_dict()[key], value)
        resumed.eval_mode()
        with torch.inference_mode():
            torch.testing.assert_close(resumed.actor(obs), alg.actor(obs))
        resumed.train_mode()
        resumed_losses, _ = rollout_and_update(resumed, obs)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in resumed_losses.values()))

    def test_legacy_ppo_optimizer_reward_and_resume(self):
        alg, obs = setup_algorithm(instinct=False)
        self.assertIsInstance(alg.disc_optimizer, torch.optim.Adam)
        self.assertEqual([group["weight_decay"] for group in alg.disc_optimizer.param_groups], [1e-4, .1])
        losses, logs = rollout_and_update(alg, obs)
        self.assertEqual(losses["amp/disc_weight_l2"], 0.)
        self.assertEqual(losses["amp/disc_logit_l2"], 0.)
        for log in logs:
            torch.testing.assert_close(log["AMP/mixed_reward_per_step"],
                                       .5 * (1 + log["AMP/style_reward_per_step"]))
            torch.testing.assert_close(log["AMP/style_reward_per_step"],
                                       .04 * log["AMP/raw_style_reward_per_step"])
        saved = copy.deepcopy(alg.save())
        alg.load(saved, None, True)
        self.assertGreater(int(alg.amp_discriminator.disc_obs_normalizer.count), 0)
        self.assertEqual(int(alg.disc_obs_buffer.current_length.max()), 0)


if __name__ == "__main__":
    unittest.main()
