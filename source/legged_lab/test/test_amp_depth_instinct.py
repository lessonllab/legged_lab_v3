"""CPU regression checks for delayed vision and both neural-network inputs."""

from types import SimpleNamespace
import importlib.metadata
import unittest

import torch
from tensordict import TensorDict
from isaaclab.managers import ObservationManager
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from rsl_rl.models import CNNModel

from legged_lab.tasks.locomotion.amp.mdp.instinct_depth import preprocess_instinct_depth
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_instinct_env_cfg import (
    InstinctDepthObservationsCfg, G1AmpDepthInstinctEnvCfg, G1AmpDepthInstinctWarmupEnvCfg,
)
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_env_cfg import G1AmpDepthEnvCfg
from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_depth_instinct_ppo_cfg import (
    G1AmpDepthInstinctPPORunnerCfg,
)


class TestInstinctDepth(unittest.TestCase):
    def test_crop_blur_and_missing_returns(self):
        raw = torch.zeros(2, 36, 64, 1)
        raw[0, 18:, 16:-16] = 1.25
        raw[1] = float("inf")
        out = preprocess_instinct_depth(raw)
        self.assertEqual(tuple(out.shape), (2, 18, 32))
        torch.testing.assert_close(out[0], torch.full((18, 32), 0.5))
        torch.testing.assert_close(out[1], torch.ones(18, 32))
        raw[1] = 0.05
        torch.testing.assert_close(preprocess_instinct_depth(raw)[1], torch.ones(18, 32))
        raw[0] = 0
        raw[0, 25, 30] = 2.5
        out = preprocess_instinct_depth(raw)
        self.assertGreater(out[0, 7, 14], 0)
        self.assertLess(out[0, 7, 14], 1)
        self.assertGreater(out[0, 7, 15], 0)

    def test_history_spacing_delay_and_partial_reset(self):
        raw = torch.zeros(2, 36, 64, 1)
        env = SimpleNamespace(
            num_envs=2, device="cpu", common_step_counter=0,
            sim=SimpleNamespace(is_playing=lambda: True),
            scene={"depth_camera": SimpleNamespace(data=SimpleNamespace(output={"distance_to_image_plane": raw}))},
        )
        manager = ObservationManager({"depth": InstinctDepthObservationsCfg()}, env)
        term = manager._group_obs_term_cfgs["depth"][0].func
        term._delay[:] = torch.tensor([0, 1])
        for step in range(41):
            env.common_step_counter = step
            raw.fill_(1.0 + step / 100)
            out = manager.compute()["depth"].clone()
            torch.testing.assert_close(out, manager.compute()["depth"])
        expected = 0.4 + (40 - torch.arange(35, -1, -5)) / 250
        torch.testing.assert_close(out[0, :, 0, 0], expected)
        torch.testing.assert_close(out[1, :, 0, 0], expected - 1 / 250)
        manager.reset(torch.tensor([0]))
        raw[0] = 2.5
        reset = manager.compute()["depth"]
        torch.testing.assert_close(reset[0], torch.ones_like(reset[0]))
        torch.testing.assert_close(reset[1], out[1])

    def test_routing_network_gradients_and_export(self):
        rough, flat, original = G1AmpDepthInstinctEnvCfg(), G1AmpDepthInstinctWarmupEnvCfg(), G1AmpDepthEnvCfg()
        rough.validate()
        flat.validate()
        self.assertEqual(rough.scene.terrain.terrain_type, "generator")
        self.assertEqual(flat.scene.terrain.terrain_type, "generator")
        self.assertEqual(list(flat.scene.terrain.terrain_generator.sub_terrains), ["flat"])
        self.assertIsNone(flat.curriculum.terrain_levels)
        self.assertIsNotNone(original.observations.critic.height_scan)
        for env_cfg in (rough, flat):
            self.assertIsNone(env_cfg.observations.policy.height_scan)
            self.assertIsNone(env_cfg.observations.critic.height_scan)
            self.assertEqual(env_cfg.scene.depth_camera.update_period, 0.02)
            self.assertTrue(env_cfg.scene.depth_camera.mesh_prim_paths[1].track_mesh_transforms)
            targets = env_cfg.scene.depth_camera.mesh_prim_paths
            self.assertEqual(len(targets), 32)  # terrain + 29 bodies + torso + logo
            bodies = set()
            for target in targets[1:]:
                relative = target.prim_expr.split("/Robot/")[1]
                body = relative.split("/")[0]
                bodies.add(body)
                self.assertNotIn("*", body)
                self.assertNotIn("[", body)
                self.assertTrue(target.track_mesh_transforms)
                if body == "torso_link":
                    self.assertIn(relative, ("torso_link/visuals/torso_link_rev_1_0", "torso_link/visuals/logo_link"))
            self.assertEqual(len(bodies), 30)
        cfg = handle_deprecated_rsl_rl_cfg(G1AmpDepthInstinctPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
        self.assertFalse(cfg.actor.distribution_cfg.learn_std)
        self.assertEqual(cfg.experiment_name, "g1_amp_depth_instinct_v2")
        self.assertEqual(cfg.obs_groups["actor"], ["policy", "depth"])
        self.assertEqual(cfg.obs_groups["critic"], ["critic", "depth"])
        obs = TensorDict({"policy": torch.randn(2, 495), "critic": torch.randn(2, 600),
                          "depth": torch.rand(2, 8, 18, 32)}, batch_size=[2])
        for role, dim in (("actor", 29), ("critic", 1)):
            kwargs = getattr(cfg, role).to_dict()
            kwargs.pop("class_name")
            model = CNNModel(obs, cfg.obs_groups, role, dim, **kwargs)
            out = model(obs)
            self.assertEqual(tuple(out.shape), (2, dim))
            out.square().mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.cnns.parameters()))
            model.eval()
            torch.testing.assert_close(torch.jit.script(model.as_jit())(obs[cfg.obs_groups[role][0]], [obs["depth"]]), model(obs))


if __name__ == "__main__":
    unittest.main()
