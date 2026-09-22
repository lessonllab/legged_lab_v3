import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_stairs_long_env_cfg import G1AmpStairsLongEnvCfg_PLAY
from legged_lab.tasks.locomotion.amp.mdp.stair_play_matrix import configure_stair_matrix, matrix_assignment, StairMatrixCommand
from legged_lab.tasks.locomotion.amp.mdp.stair_speed_course import StairSpeedTargetCommand
from legged_lab.tasks.locomotion.amp.mdp.commands.target_velocity import terrain_column_names
from legged_lab.tasks.locomotion.amp.mdp.stair_edge_guidance import exposed_stair_rings


class TestStairMatrix(unittest.TestCase):
    def test_every_tile_has_exactly_one_robot_and_all_heights_both_directions(self):
        cfg=G1AmpStairsLongEnvCfg_PLAY();configure_stair_matrix(cfg);cfg.validate()
        gen=cfg.scene.terrain.terrain_generator;names=terrain_column_names(gen)
        rows,cols,groups=matrix_assignment(cfg.scene.num_envs,names,'cpu')
        self.assertEqual(cfg.scene.num_envs,100)
        self.assertEqual(len(set(zip(rows.tolist(),cols.tolist()))),100)
        self.assertEqual(torch.bincount(groups).tolist(),[10,18,18,54])
        self.assertEqual(int(rows.max()),9)
        for row in range(1,10):
            for g in (1,2):self.assertEqual(int(((rows==row)&(groups==g)).sum()),2)
        self.assertIsNone(cfg.curriculum.terrain_levels)
        self.assertEqual(G1AmpStairsLongEnvCfg_PLAY().scene.terrain.terrain_generator.num_rows,10)

    def test_actual_stair_meshes_have_17_steps_at_each_displayed_height(self):
        cfg=G1AmpStairsLongEnvCfg_PLAY();configure_stair_matrix(cfg)
        gen=cfg.scene.terrain.terrain_generator
        for name in ('pyramid_stairs','pyramid_stairs_inv'):
            t=gen.sub_terrains[name].copy();t.size=gen.size
            for row,height in enumerate((.08,.10,.12,.14,.16,.18,.20,.25,.30),1):
                difficulty=(row+.5)/gen.num_rows*(gen.difficulty_range[1]-gen.difficulty_range[0])
                meshes,origin=t.function(difficulty,t)
                rings=exposed_stair_rings(meshes,origin,t.size)
                self.assertEqual(rings.shape,(17,2))
                np.testing.assert_allclose(abs(np.diff(rings[:,1])),height,atol=1e-5)

    def test_fixed_speed_survives_resets_without_changing_manual_targets(self):
        c=object.__new__(StairMatrixCommand);c._ids=lambda ids:torch.tensor(ids)
        c.manual_target=torch.tensor([False,True,False]);c.speed_cap=torch.zeros(3)
        c.matrix_speed=.8;c._update_selected=lambda ids:None
        with patch.object(StairSpeedTargetCommand,'_resample_command'):
            c._resample_command([0,1,2])
        torch.testing.assert_close(c.speed_cap,torch.tensor([.8,0.,.8]))

    def test_manual_control_switches_origins_groups_and_height_limit(self):
        from legged_lab.tasks.locomotion.amp.mdp.stair_play_matrix import configure_stair_control, StairControlCommand
        cfg=G1AmpStairsLongEnvCfg_PLAY();configure_stair_control(cfg);cfg.validate()
        self.assertEqual(cfg.scene.num_envs,1)
        c=object.__new__(StairControlCommand)
        c.column_names=terrain_column_names(cfg.scene.terrain.terrain_generator)
        c.group=torch.ones(1,dtype=torch.long);c.flat_pool=torch.zeros(1,dtype=torch.bool)
        c.manual_target=torch.ones(1,dtype=torch.bool)
        origins=torch.arange(10*10*3).reshape(10,10,3).float()
        c.terrain=NS(terrain_levels=torch.zeros(1,dtype=torch.long),terrain_types=torch.zeros(1,dtype=torch.long),
                     terrain_origins=origins,env_origins=torch.zeros(1,3))
        for kind,level,group,column in [('下楼',9,2,0),('上楼',2,1,2),('平地',7,0,0),('块状地形',4,3,4)]:
            c.select_terrain(kind,level,.8)
            row=0 if kind=='平地' else level
            self.assertEqual(c.group.item(),group)
            self.assertEqual(c.flat_pool.item(),group==0)
            self.assertFalse(c.manual_target.item())
            torch.testing.assert_close(c.terrain.env_origins[0],origins[row,column])
        for level in (0,10):
            with self.assertRaises(ValueError):c.select_terrain('上楼',level,.8)

    def test_invalid_speed_or_incomplete_layout_is_rejected(self):
        for speed in (0.,-1.,float('nan'),2.):
            with self.assertRaises(ValueError):configure_stair_matrix(G1AmpStairsLongEnvCfg_PLAY(),speed)
        with self.assertRaises(ValueError):matrix_assignment(19,['pyramid_stairs','pyramid_stairs_inv'],'cpu')


if __name__=='__main__':unittest.main()
