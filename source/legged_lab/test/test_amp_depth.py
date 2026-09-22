"""CPU checks for visual AMP preprocessing, history, routing, and exported policy.

Run with Isaac Lab + RSL-RL installed and this project's source on PYTHONPATH.
The ObservationManager and neural networks are real; only sensor frames are synthetic.
"""

import importlib.metadata
from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict

from isaaclab.managers import ObservationManager
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from rsl_rl.models import CNNModel

from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_env_cfg import (
    DepthObservationsCfg,
    G1AmpDepthEnvCfg,
    G1AmpDepthEnvCfg_PLAY,
)
from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_depth_ppo_cfg import G1AmpDepthPPORunnerCfg
from legged_lab.tasks.locomotion.amp.mdp.depth import preprocess_depth


class TestDepthAMP(unittest.TestCase):
    def test_invalid_depth_and_units(self):
        raw = torch.tensor([[[0.0, float("nan"), float("inf"), -2.0, 0.1, 1.3, 2.5, 8.0]]])
        result = preprocess_depth(raw[..., None])
        torch.testing.assert_close(result, torch.tensor([[[-1., -1., -1., -1., 0., 0.5, 1., 1.]]]))
        self.assertTrue(torch.isnan(raw[0, 0, 1]))
        self.assertTrue(torch.isinf(raw[0, 0, 2]))
        with self.assertRaises(ValueError):
            preprocess_depth(raw, near=3.0, far=2.0)
        with self.assertRaises(ValueError):
            preprocess_depth(torch.zeros(2, 3))

    def test_camera_history_shape_and_partial_episode_reset(self):
        raw = torch.full((2, 36, 64, 1), 0.5)
        env = SimpleNamespace(
            num_envs=2, device="cpu", sim=SimpleNamespace(is_playing=lambda: True),
            scene={"depth_camera": SimpleNamespace(data=SimpleNamespace(output={"distance_to_image_plane": raw}))},
        )
        manager = ObservationManager({"depth": DepthObservationsCfg()}, env)
        first = manager.compute(update_history=True)["depth"].clone()
        self.assertEqual(tuple(first.shape), (2, 8, 36, 64))
        raw.fill_(1.3)
        manager.compute(update_history=True)
        manager.reset(torch.tensor([0]))
        raw[0].fill_(2.5)
        result = manager.compute(update_history=True)["depth"]
        torch.testing.assert_close(result[0], torch.ones_like(result[0]))
        # Other environments retain their history instead of being reset with env 0.
        self.assertTrue(torch.any(result[1] < 0.5))
        self.assertTrue(torch.any(torch.isclose(result[1], torch.tensor(0.5))))

    def test_config_privileged_inputs_and_cnn_gradient_and_export(self):
        train, play = G1AmpDepthEnvCfg(), G1AmpDepthEnvCfg_PLAY()
        train.validate()
        play.validate()
        self.assertIsNone(train.observations.policy.height_scan)
        self.assertIsNotNone(train.observations.critic.height_scan)
        self.assertIsNotNone(train.scene.height_scanner)
        self.assertEqual(train.scene.depth_camera.ray_alignment, "base")
        self.assertIsNot(train.scene.terrain.terrain_generator, play.scene.terrain.terrain_generator)
        self.assertTrue(train.scene.terrain.terrain_generator.curriculum)
        cfg = handle_deprecated_rsl_rl_cfg(G1AmpDepthPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
        self.assertEqual(cfg.obs_groups["actor"], ["policy", "depth"])
        self.assertEqual(cfg.obs_groups["discriminator"], ["disc"])
        self.assertIsNone(cfg.algorithm.symmetry_cfg)
        obs = TensorDict({"policy": torch.randn(2, 495), "depth": torch.rand(2, 8, 36, 64)}, batch_size=[2])
        kwargs = cfg.actor.to_dict()
        kwargs.pop("class_name")
        model = CNNModel(obs, cfg.obs_groups, "actor", 29, **kwargs)
        output = model(obs)
        self.assertEqual(tuple(output.shape), (2, 29))
        output.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.cnns.parameters()))
        model.eval()
        scripted = torch.jit.script(model.as_jit())
        torch.testing.assert_close(scripted(obs["policy"], [obs["depth"]]), model(obs))
        # Export contains both inputs and the CNN, not just the MLP action head.
        onnx_wrapper = model.as_onnx()
        self.assertEqual(onnx_wrapper.input_names, ["obs", "depth"])
        torch.testing.assert_close(onnx_wrapper(obs["policy"], obs["depth"]), model(obs))


if __name__ == "__main__":
    unittest.main()
