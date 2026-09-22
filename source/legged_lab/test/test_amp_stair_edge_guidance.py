import unittest
import numpy as np
import torch
from legged_lab.tasks.locomotion.amp.mdp.stair_edge_guidance import exposed_stair_rings,edge_approach_cost
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_stairs_long_env_cfg import G1AmpStairsLongEnvCfg


class TestEdgeGuidance(unittest.TestCase):
    def test_exposed_upper_lips_match_both_stair_profiles_without_seams(self):
        generator = G1AmpStairsLongEnvCfg().scene.terrain.terrain_generator
        for name in ('pyramid_stairs','pyramid_stairs_inv'):
            cfg = generator.sub_terrains[name].copy()
            cfg.size = generator.size
            for row in (1,4,9):
                meshes,origin = cfg.function((row+.5)/10.,cfg)
                rings = exposed_stair_rings(meshes,origin,cfg.size)
                self.assertEqual(rings.shape,(17,2))
                np.testing.assert_allclose(np.diff(rings[:,0]),.3,atol=1e-5)
                # Check upper-lip height against the actual visible mesh at all
                # four sides; buried box tops must never produce ghost edges.
                for r,z in rings:
                    for axis in (0,1):
                        for sign in (-1,1):
                            tops=[]
                            for delta in (-.0001,.0001):
                                p=origin[:2].copy();p[axis]+=sign*(float(r)+delta)
                                tops.append(max(m.bounds[1,2] for m in meshes if
                                    np.all(p>=m.bounds[0,:2]) and np.all(p<=m.bounds[1,:2])))
                            self.assertGreater(abs(tops[0]-tops[1]),.05)
                            self.assertAlmostEqual(max(tops)-origin[2],float(z),places=5)

    def test_contact_approach_and_toeoff_are_distinguished(self):
        p = torch.tensor([[[[1.01,0.,.01]]], [[[1.01,0.,.01]]],
                          [[[.85,0.,.01]]], [[[1.01,0.,.2]]], [[[1.01,0.,.01]]]])
        v = torch.tensor([[[[-.3,0.,-.3]]], [[[.3,0.,.3]]],
                          [[[.3,0.,-.3]]], [[[-.3,0.,-.3]]], [[[-.3,0.,-.3]]]])
        rings = torch.tensor([[[1.,0.],[2.,.2]]]).expand(5,-1,-1)
        unloaded = torch.tensor([[1.],[1.],[1.],[1.],[0.]])
        result = edge_approach_cost(p,v,rings,unloaded)
        self.assertGreater(result[0],0.)
        torch.testing.assert_close(result[1:],torch.zeros(4))

    def test_corners_rotated_sides_padding_and_higher_edge(self):
        p = torch.tensor([[[[1.01,1.01,.01],[-1.01,0.,.01],[0.,-1.01,.01]]]])
        v = -p.clone(); v[...,2]=-.3
        rings=torch.tensor([[[1.,0.],[2.,.2],[1e6,0.]]])
        for point in range(3):
            self.assertGreater(edge_approach_cost(p[:,:,point:point+1],v[:,:,point:point+1],rings,torch.ones(1,1)),0.)
        empty=rings.clone();empty[...,0]=1e6
        self.assertEqual(edge_approach_cost(p,v,empty,torch.ones(1,1)),0.)
        self.assertTrue(torch.isfinite(edge_approach_cost(p,v,rings,torch.ones(1,1))).all())


if __name__=='__main__':unittest.main()
