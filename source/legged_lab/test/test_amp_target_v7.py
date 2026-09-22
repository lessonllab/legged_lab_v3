import copy
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace as NS
import torch
from isaaclab.managers import SceneEntityCfg
from legged_lab.tasks.locomotion.amp.mdp.locomotion_progress import command_progress, low_pelvis_height, style_tracking_gate
from legged_lab.rsl_rl.amp.reward_gate import get_style_gate
from legged_lab.rsl_rl.actor_warm_start import initialize_actor
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_target_v6_env_cfg import G1AmpDepthTargetV6EnvCfg
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_target_v7_env_cfg import G1AmpDepthTargetV7EnvCfg


class TestProgressV7(unittest.TestCase):
    def test_fast_commands_supported_and_overspeed_discounted_without_erasing_style(self):
        c = torch.tensor([[.5, 0., 0.], [.5, 0., 0.], [2., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., -1.]])
        v = torch.tensor([[.5, 0.], [1., 0.], [2., 0.], [0., 0.], [1., 0.], [0., 0.]])
        g = style_tracking_gate(c, v, torch.tensor([0., 0., 0., 0., 0., -1.]))
        torch.testing.assert_close(g[[0, 2, 3, 5]], torch.ones(4))
        self.assertGreaterEqual(g[1].item(), .5)
        self.assertLess(g[1].item(), .6)
        self.assertLess(g[4].item(), .001)

    def test_tracking_floor_does_not_reward_commanded_standing_or_reverse(self):
        c = torch.tensor([[.5, 0., 0.]] * 4 + [[0., 0., 1.]])
        v = torch.tensor([[0., 0.], [-.5, 0.], [.25, 0.], [.5, 1.], [0., 0.]])
        w = torch.zeros(5)
        gate = style_tracking_gate(c, v, w)
        torch.testing.assert_close(gate[[0, 1, 4]], torch.zeros(3))
        self.assertGreaterEqual(gate[2].item(), .25)
        self.assertLessEqual(gate[2].item(), .5)
        self.assertGreaterEqual((.5 * gate[3]).item(), .25)
        strict = style_tracking_gate(c, v, w, moving_tracking_floor=0.)
        self.assertTrue(torch.all(gate >= strict))
        with self.assertRaises(ValueError):
            style_tracking_gate(c, v, w, moving_tracking_floor=1.1)

    def test_posture_penalty_tracks_clearance_not_world_height(self):
        data = NS(root_pos_w=torch.tensor([[0., 0., .3], [0., 0., 2.3], [0., 0., .7]]),
                  body_pos_w=torch.tensor([[[0., 0., 0.], [0., 0., .2]],
                                          [[0., 0., 2.], [0., 0., 2.2]],
                                          [[0., 0., 0.], [0., 0., .1]]]))
        env = NS(scene={'robot': NS(data=data)})
        cost = low_pelvis_height(env, .55, SceneEntityCfg('robot', body_ids=[0, 1]))
        torch.testing.assert_close(cost, torch.tensor([.0625, .0625, 0.]))

    def test_still_wrong_direction_and_correct_direction(self):
        c = torch.tensor([[.5, 0., 0.]] * 5 + [[0., 0., 1.], [0., 0., -1.], [0., 0., 0.], [2., 0., 0.]])
        v = torch.tensor([[0., 0.], [-.5, 0.], [.25, 0.], [.5, 0.], [1., 0.], [0., 0.], [0., 0.], [0., 0.], [1., 0.]])
        w = torch.tensor([0., 0., 0., 0., 0., 0., -.5, 0., 0.])
        torch.testing.assert_close(command_progress(c, v, w), torch.tensor([0., 0., .5, 1., 1., 0., .5, 1., .5]))
        # Turning in place cannot collect moving-forward style rewards.
        self.assertEqual(command_progress(c[:1], v[:1], torch.ones(1)).item(), 0.)

    def test_gate_uses_terminal_state_and_does_not_modify_obs(self):
        obs = {'style_gate': torch.tensor([[1.], [.5]])}
        extras = {'terminal_obs': {'style_gate': torch.tensor([[0.], [1.]])}}
        gate = get_style_gate(obs, torch.tensor([True, False]), extras, 'style_gate')
        torch.testing.assert_close(gate, torch.tensor([0., .5]))
        self.assertEqual(obs['style_gate'][0].item(), 1.)
        with self.assertRaises(ValueError):
            get_style_gate(obs, torch.tensor([True, False]), {}, 'style_gate')
        with self.assertRaises(ValueError):
            get_style_gate({'style_gate': torch.tensor([[float('nan')], [.5]])}, torch.tensor([False, False]), {}, 'style_gate')

    def test_reference_set_and_actor_inputs_preserved(self):
        old = G1AmpDepthTargetV6EnvCfg()
        before = copy.deepcopy(old.to_dict())
        new = G1AmpDepthTargetV7EnvCfg()
        new.validate()
        self.assertEqual(old.to_dict(), before)
        self.assertEqual(old.motion_data.to_dict(), new.motion_data.to_dict())
        self.assertEqual(old.animation.to_dict(), new.animation.to_dict())
        for key in ('policy', 'critic', 'depth', 'disc', 'disc_demo'):
            self.assertEqual(getattr(old.observations, key).to_dict(), getattr(new.observations, key).to_dict())
        self.assertIn('style_gate', new.terminal_obs_groups)
        self.assertEqual(new.commands.base_velocity.speed_ranges['random_rough'], (.45, 2.))
        self.assertEqual(new.commands.base_velocity.speed_ranges['pyramid_stairs'], (.45, .8))

    def test_actor_transfer_preserves_fresh_noise(self):
        actor = torch.nn.Module()
        actor.add_module('mlp', torch.nn.Linear(2, 2))
        actor.add_module('distribution', torch.nn.Linear(2, 2))
        noise = {k: v.clone() for k, v in actor.distribution.state_dict().items()}
        source = {k: torch.full_like(v, .8) for k, v in actor.state_dict().items()}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'model.pt'
            torch.save({'actor_state_dict': source}, p)
            initialize_actor(actor, p)
        torch.testing.assert_close(actor.mlp.weight, torch.full_like(actor.mlp.weight, .8))
        for k, v in actor.distribution.state_dict().items():
            torch.testing.assert_close(v, noise[k])


if __name__ == '__main__':
    unittest.main()
