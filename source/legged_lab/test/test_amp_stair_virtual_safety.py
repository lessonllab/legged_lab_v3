import unittest
import torch
from legged_lab.tasks.locomotion.amp.mdp.stair_virtual_safety import (
    cylinder_depth, instinct_penetration_cost, toe_riser_cost, stair_surface_height,
    loaded_overhang_cost, swing_clearance_cost,
)


class TestVirtualSafety(unittest.TestCase):
    def test_surface_height_matches_actual_meshes_both_directions_all_heights(self):
        import numpy as np
        from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_stairs_long_env_cfg import G1AmpStairsLongEnvCfg
        from legged_lab.tasks.locomotion.amp.mdp.stair_edge_guidance import exposed_stair_rings
        gen = G1AmpStairsLongEnvCfg().scene.terrain.terrain_generator
        rng = np.random.default_rng(42)
        for name in ('pyramid_stairs_inv', 'pyramid_stairs'):
            cfg = gen.sub_terrains[name].copy(); cfg.size = gen.size
            for row in range(1, 10):
                meshes, origin = cfg.function((row+.5)/10., cfg)
                bounds = np.array([m.bounds for m in meshes])
                rings = exposed_stair_rings(meshes, origin, gen.size)
                def height(xy):
                    hit = ((xy[:,None] >= bounds[None,:,0,:2]) & (xy[:,None] <= bounds[None,:,1,:2])).all(-1)
                    return np.where(hit, bounds[None,:,1,2], -np.inf).max(-1)-origin[2]
                r = np.asarray(rings)[:,0]
                inside = height(np.column_stack((r+origin[0]-1e-4, np.full_like(r,origin[1]))))
                outside = height(np.column_stack((r+origin[0]+1e-4, np.full_like(r,origin[1]))))
                xy = rng.uniform(-6.8, 6.8, (500,2))
                p = torch.tensor(np.column_stack((xy, np.zeros(500))),dtype=torch.float32)[None]
                result = stair_surface_height(p,torch.tensor(np.asarray(rings),dtype=torch.float32)[None],
                    torch.tensor(np.minimum(inside,outside),dtype=torch.float32)[None],
                    torch.tensor(outside>inside)[None])
                torch.testing.assert_close(result[0],torch.tensor(height(xy+origin[:2]),dtype=torch.float32),atol=1e-5,rtol=1e-5)

    def test_loaded_overhang_excludes_swing_unloading_flat_tilt_and_missing_data(self):
        heights = torch.tensor([[[.15,.15,0.,0.]]]).expand(6,-1,-1).clone()
        sole = torch.zeros(6,1,4,3); sole[...,2] = .15
        forces = torch.zeros(6,1,3); forces[...,2] = 250.
        forces[1,0,2] = 0. # swing
        forces[2,0,2] = 30. # unloading
        heights[3] = .15; sole[3,0,:,2] += torch.arange(4)*.02 # tilted on one plane
        heights[4] = float('nan') # unavailable != edge
        heights[5] = .15 # safe tread
        cost,_ = loaded_overhang_cost(sole,heights,forces)
        torch.testing.assert_close(cost,torch.tensor([.5,0.,0.,0.,0.,0.]))

    def test_swing_clearance_anticipates_lip_without_penalizing_stance_descent_or_retreat(self):
        p = torch.tensor([[[[.96,0.,.15]]]]).expand(7,-1,-1,-1).clone()
        rings = torch.tensor([[[1.,.15]]]).expand(7,-1,-1)
        forces = torch.zeros(7,1,3); forces[1,0,2] = 250.
        air = torch.full((7,1),.1); air[2] = 0.
        desired = torch.tensor([[.6,0.,0.]]).expand(7,-1).clone()
        desired[3,0] = -.6 # heading away / descending
        p[4,0,0,2] = .18 # sufficient clearance
        p[5,0,0,0] = .7 # far away
        desired[6] = 0. # stop request
        cost = swing_clearance_cost(p,rings,torch.zeros(7,1),torch.ones(7,1,dtype=torch.bool),forces,air,desired)
        self.assertGreater(cost[0],0.) # at tread height still lacks 2.5 cm clearance
        torch.testing.assert_close(cost[1:],torch.zeros(6))
        # Mirror geometry and direction: all four sides and inward ascent agree.
        p = torch.tensor([[[[1.04,0.,.15]]],[[[-1.04,0.,.15]]],[[[0.,1.04,.15]]],[[[0.,-1.04,.15]]]])
        desired = torch.tensor([[-.6,0.,0.],[.6,0.,0.],[0.,-.6,0.],[0.,.6,0.]])
        mirror = swing_clearance_cost(p,rings[:4],torch.zeros(4,1),torch.zeros(4,1,dtype=torch.bool),
                                      torch.zeros(4,1,3),torch.full((4,1),.1),desired)
        torch.testing.assert_close(mirror,cost[0].expand(4))

    def test_finite_cylinders_match_independent_segment_reference(self):
        torch.manual_seed(1)
        points=torch.rand(2,3000,3)*2.6-1.3
        points[:,:,2]*=.15
        rings=torch.tensor([[[.7,0.],[1.,.08]]]).expand(2,-1,-1)
        expected=torch.zeros(2,3000)
        for r,h in rings[0].tolist():
            for axis in (0,1):
                for sign in (-1,1):
                    a=torch.tensor([-r,-r,h]);b=a.clone()
                    a[axis]=b[axis]=sign*r;b[1-axis]=r
                    length=(b-a).norm();direction=(b-a)/length
                    t=((points-a)*direction).sum(-1)
                    distance=(points-a-t[...,None]*direction).norm(dim=-1)
                    d=torch.where((t>=0)&(t<=length),(.05-distance).clamp_min(0),0.)
                    expected=torch.maximum(expected,d)
        torch.testing.assert_close(cylinder_depth(points,rings),expected,atol=1e-6,rtol=1e-5)

    def test_cylinder_axis_is_finite_and_no_capsule_beyond_corner(self):
        p=torch.tensor([[[1.,0.,0.],[1.01,1.01,0.],[1.06,0.,0.],[1.,0.,.06]]])
        depth=cylinder_depth(p,torch.tensor([[[1.,0.]]]))
        torch.testing.assert_close(depth,torch.tensor([[.05,0.,0.,0.]]))
        self.assertTrue(torch.isfinite(instinct_penetration_cost(depth,torch.zeros_like(p))).all())
        v=torch.ones_like(p)
        self.assertAlmostEqual(instinct_penetration_cost(depth,v).item(),.05*(3**.5+1e-6),places=6)

    def test_toe_approach_contact_clearance_retreat_and_floor(self):
        p=torch.tensor([[[[.98,0.,.05]]]]).expand(6,-1,-1,-1).clone()
        v=torch.zeros_like(p);v[:,0,0,0]=.5
        p[1,0,0,2]=.16 # above upper lip
        v[2,0,0,0]=-.5 # withdrawing without contact
        p[3,0,0,0]=.8 # away from riser
        v[4]=0. # blocked, horizontal normal impact
        forces=torch.zeros(6,1,3);forces[:,0,2]=200.;forces[4,0,0]=-120.
        p[5,0,0,2]=.15 # on tread height, not the vertical face
        rings=torch.tensor([[[1.,.15]]]).expand(6,-1,-1)
        cost,near=toe_riser_cost(p,v,rings,torch.zeros(6,1),torch.ones(6,1,dtype=torch.bool),forces)
        self.assertGreater(cost[0],0);self.assertGreater(cost[4],0)
        torch.testing.assert_close(cost[[1,2,3,5]],torch.zeros(4))
        self.assertFalse(near[1].any())

    def test_climbing_inward_and_other_sides(self):
        p=torch.tensor([[[[1.02,0.,.05]]],[[[-1.02,0.,.05]]],[[[0.,1.02,.05]]]])
        v=torch.tensor([[[[-.5,0.,0.]]],[[[.5,0.,0.]]],[[[0.,-.5,0.]]]])
        cost,_=toe_riser_cost(p,v,torch.tensor([[[1.,.15]]]).expand(3,-1,-1),
                             torch.zeros(3,1),torch.zeros(3,1,dtype=torch.bool),torch.zeros(3,1,3))
        self.assertTrue((cost>0).all())

if __name__=='__main__':unittest.main()
