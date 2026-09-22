import unittest
import torch
from legged_lab.tasks.locomotion.amp.mdp.foot_support import support_costs, sole_ray_pattern
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_instinct_env_cfg import G1AmpDepthInstinctEnvCfg


class TestFootSupport(unittest.TestCase):
    def test_graded_edge_separates_safe_lip_overhang_load_and_unloading(self):
        h = torch.full((5, 1, 5, 11), -.035)
        h[0, :, :, -1] -= .08  # full support, edge only in outer ring
        h[1:, :, :, 5:] -= .08  # missing sole support
        force = torch.tensor([[150.], [150.], [280.], [10.], [150.]])
        age = torch.tensor([[.2], [.2], [.2], [.2], [0.]])
        _, edge, _ = support_costs(h, force, age, support_target=.9, graded_edge=True)
        self.assertGreater(edge[1], edge[0])
        self.assertGreater(edge[2], edge[1])
        self.assertEqual(edge[3], 0.)
        self.assertGreater(edge[4], 0.)
        self.assertLess(edge[4], edge[1])
        _, legacy, _ = support_costs(h, force, age, immediate_edge=True)
        self.assertEqual(legacy[0], legacy[1])

    def test_immediate_edge_penalizes_loaded_landing_but_not_swing(self):
        h = torch.full((3, 1, 5, 11), -.035)
        h[:2, :, :, -1] -= .08
        forces = torch.tensor([[100.], [0.], [100.]])
        age = torch.zeros(3, 1)
        deficit, edge, _ = support_costs(h, forces, age, contact_ramp=.04,
                                         immediate_edge=True)
        torch.testing.assert_close(edge, torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(deficit, torch.zeros(3))
        _, legacy, _ = support_costs(h, forces, age, contact_ramp=.04)
        torch.testing.assert_close(legacy, torch.zeros(3))

    def test_safe_partial_edge_and_swing(self):
        h = torch.full((5, 2, 5, 11), -.035)
        # One loaded foot: partial overhang, outside-perimeter edge, missing ray.
        h[1, 0, :, 6:] -= .2
        h[2, 0, :, -1] -= .2
        h[3, 0, :, :] = float('inf')
        h[4] = h[1]
        force = torch.full((5, 2), 100.)
        force[4] = 0.
        deficit, edge, fraction = support_costs(h, force, torch.ones(5, 2))
        self.assertEqual(deficit[0], 0.)
        self.assertEqual(edge[0], 0.)
        self.assertGreater(deficit[1], 0.)
        self.assertEqual(edge[1], 1.)
        self.assertEqual(deficit[2], 0.)
        self.assertEqual(edge[2], 1.)
        self.assertEqual(deficit[3], 1.)
        self.assertEqual(edge[3], 1.)
        self.assertEqual(deficit[4], 0.)
        self.assertEqual(edge[4], 0.)
        self.assertTrue(torch.isfinite(deficit).all())
        # Landing grace and weak contacts must not punish a swing/impact.
        d, e, _ = support_costs(h, force, torch.zeros(5, 2))
        self.assertEqual(d.sum() + e.sum(), 0.)
        d, e, _ = support_costs(h, torch.full_like(force, 10.), torch.ones(5, 2))
        self.assertEqual(d.sum() + e.sum(), 0.)

    def test_slope_pattern_and_config(self):
        starts, directions = sole_ray_pattern(None, 'cpu')
        self.assertEqual(starts.shape, (55, 3))
        grid = starts.reshape(5, 11, 3)
        torch.testing.assert_close(grid[1, 1, :2], torch.tensor([-.05, -.025]))
        torch.testing.assert_close(grid[-2, -2, :2], torch.tensor([.12, .025]))
        # Even a foot misaligned by 10 degrees with a smooth slope has no
        # step discontinuity and remains within the sole tolerance here.
        h = -.035 + grid[..., 0] * .1763
        d, e, _ = support_costs(h[None, None], torch.ones(1, 1)*100, torch.ones(1, 1))
        self.assertEqual(d.item(), 0.)
        self.assertEqual(e.item(), 0.)
        cfg = G1AmpDepthInstinctEnvCfg()
        cfg.validate()
        self.assertLess(cfg.rewards.feet_support.weight, 0)
        self.assertLess(cfg.rewards.feet_edge.weight, 0)
        for group in vars(cfg.observations).values():
            self.assertNotIn('sole_scanner', str(group))


if __name__ == '__main__':
    unittest.main()
