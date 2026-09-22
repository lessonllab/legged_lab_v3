import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import torch
from legged_lab.tasks.locomotion.amp.mdp.agility_course import agility_layout, rough_outcomes, AgilityCommandMixin, flat_reverse_command
from legged_lab.tasks.locomotion.amp.mdp.commands.target_velocity import target_velocity
from legged_lab.tasks.locomotion.amp.mdp.balanced_course import update_evidence
from legged_lab.rsl_rl.amp.stairs_ppo import restore_indices
from legged_lab.rsl_rl.amp.ppo_amp import PPOAMP
import test_amp_stair_speed as fixtures


class TestAgility(unittest.TestCase):
    def test_reverse_goals_keep_heading_and_use_negative_velocity(self):
        position=torch.zeros(4,3)
        quat=torch.tensor([[0.,0.,0.,1.]]).expand(4,-1).clone()
        quat[2]=torch.tensor([0.,0.,1.,0.]) # must align before translating
        target=torch.tensor([[-2.5,0.,0.],[2.5,0.,0.],[-2.5,0.,0.],[.1,0.,0.]])
        velocity,heading,reached=flat_reverse_command(position,quat,target,torch.full((4,),.4))
        torch.testing.assert_close(velocity[:,0],torch.tensor([-.4,.4,0.,0.]))
        self.assertEqual(velocity[0,2],0.) # no turn-around for the rear goal
        self.assertEqual(heading[0],0.)
        self.assertAlmostEqual(abs(velocity[2,2].item()),1.)
        self.assertEqual(reached.tolist(),[False,False,False,True])

    def test_reverse_resampling_stays_flat_and_does_not_steal_height_lanes(self):
        from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_stairs_long_env_cfg import G1AmpStairsLongEnvCfg
        from legged_lab.tasks.locomotion.amp.mdp.adaptive_stair_course import adaptive_stair_curriculum, sample_adaptive_stages, HEIGHT, CHALLENGE, adaptive_step
        cfg=G1AmpStairsLongEnvCfg()
        self.assertIs(cfg.curriculum.terrain_levels.func,adaptive_stair_curriculum)
        self.assertEqual(cfg.commands.base_velocity.class_type.promotion_version,11)
        levels,_,lanes=sample_adaptive_stages((torch.arange(100)+.5)/100,3,4)
        self.assertEqual((lanes==HEIGHT).sum(),50)
        self.assertEqual((lanes==CHALLENGE).sum(),10)
        self.assertEqual(levels[lanes==CHALLENGE].unique().tolist(),[4])
        self.assertEqual(adaptive_step(3,80,100,60.,7),(4,True))
        self.assertEqual(adaptive_step(7,100,100,60.,7),(7,True))

    def test_reverse_manual_override_and_forward_return(self):
        class Base:
            def _update_selected(self,ids): pass
        class Command(AgilityCommandMixin,Base): pass
        c=object.__new__(Command);c._ids=lambda ids:torch.arange(3)[ids]
        c.turn_pool=torch.zeros(3,dtype=torch.bool);c.reverse_pool=torch.ones(3,dtype=torch.bool)
        c.manual_target=torch.tensor([False,False,True]);c.turn_rate=torch.zeros(3)
        c.reverse_speed=torch.full((3,),.4);c.vel_command_b=torch.ones(3,3)
        c.heading_command_w=torch.ones(3);c.reached_target=torch.zeros(3,dtype=torch.bool)
        c.robot=NS(data=NS(root_pos_w=torch.zeros(3,3),root_quat_w=torch.tensor([[0.,0.,0.,1.]]).expand(3,-1)))
        c.pos_command_w=torch.tensor([[-2.5,0.,0.],[2.5,0.,0.],[-2.5,0.,0.]])
        c._update_selected(slice(None))
        torch.testing.assert_close(c.vel_command_b[:2],torch.tensor([[-.4,0.,0.],[.4,0.,0.]]))
        torch.testing.assert_close(c.vel_command_b[2],torch.ones(3))

    def test_heading_gate_and_rear_turn(self):
        angles = torch.deg2rad(torch.tensor([0.,20.,45.,70.,90.,179.,-179.]))
        goals = 3*torch.stack((angles.cos(),angles.sin(),torch.zeros_like(angles)),-1)
        command,_,_,_=target_velocity(torch.zeros_like(goals),torch.tensor([[0.,0.,0.,1.]]).expand(7,-1),
            goals,torch.full((7,),.8),torch.zeros(7,dtype=torch.bool),heading_slowdown=True)
        torch.testing.assert_close(command[:,0],torch.tensor([.8,.8,.4,0.,0.,0.,0.]),atol=1e-6,rtol=1e-5)
        self.assertEqual(command[-2:,2].tolist(),[1.,-1.])
        self.assertTrue((command[:,1]==0).all())

    def test_layout_remap_preserves_terrain_and_direction(self):
        old,c,t=fixtures.TestStairSpeed().fixture(2048)
        with patch.object(PPOAMP,'save',return_value={}): state=old.save()['scratch_course']
        groups,columns=agility_layout(2048,c.column_names,'cpu')
        with self.assertRaisesRegex(ValueError,'different environment assignment'):
            restore_indices(state,groups,columns)
        index=restore_indices(state,groups,columns,allow_remap=True)
        torch.testing.assert_close(state['group'][index],groups)
        torch.testing.assert_close(state['types'][index][groups!=0],columns[groups!=0])
        rough=torch.tensor([c.column_names[i]=='random_rough' for i in columns]) & (groups==3)
        self.assertLess(abs(rough.float().mean().item()-.15),.002)
        self.assertLess(abs((groups==1).float().mean().item()-.25),.002)

    def test_full_v9_to_v10_migration_and_roundtrip(self):
        old,c,t=fixtures.TestStairSpeed().fixture(200)
        c.promotion_version=9;c.phase[:]=4;c.frontier[:]=torch.tensor([3,4])
        with patch.object(PPOAMP,'save',return_value={}): saved=old.save()
        new,nc,nt=fixtures.TestStairSpeed().fixture(200)
        groups,columns=agility_layout(200,nc.column_names,'cpu')
        nc.group[:]=groups;nc.flat_pool[:]=groups==0;nc.stairs=(groups==1)|(groups==2)
        nt.terrain_types[:]=columns;nc.layout_profile='agility_v10';nc.promotion_version=10
        with patch.object(PPOAMP,'load',return_value=True):new.load(saved)
        torch.testing.assert_close(nc.group,groups)
        self.assertEqual(nc.frontier.tolist(),[3,4]);self.assertEqual(nc.phase.tolist(),[4,4])
        with patch.object(PPOAMP,'save',return_value={}): v10=new.save()
        self.assertEqual(v10['scratch_course']['layout_profile'],'agility_v10')
        with patch.object(PPOAMP,'load',return_value=True):new.load(v10)

    def test_rough_requires_arrival_tracking_and_no_fall(self):
        ones=torch.ones(5);arrivals=torch.tensor([1.,0.,1.,1.,1.])
        yaw=torch.tensor([1.,1.,.2,1.,1.]);failed=torch.tensor([False,False,False,True,False])
        progress=torch.tensor([2.,2.,2.,2.,.1])
        passed,bad=rough_outcomes(ones,yaw,progress,ones*3,arrivals,failed)
        self.assertEqual(passed.tolist(),[True,False,False,False,False])
        good,fail,up,down=update_evidence(torch.zeros(5,dtype=torch.long),torch.zeros(5,dtype=torch.long),passed,bad,ones.bool())
        self.assertFalse(up.any());self.assertFalse(down.any())
        _,_,up,down=update_evidence(good,fail,passed,bad,ones.bool())
        self.assertEqual(up.tolist(),[True,False,False,False,False])
        self.assertTrue(down[3])

    def test_pure_turn_override_respects_manual_control(self):
        class Base:
            def _update_selected(self,ids): pass
        class Command(AgilityCommandMixin,Base): pass
        c=object.__new__(Command)
        c._ids=lambda ids:torch.arange(3)[ids]
        c.turn_pool=torch.tensor([True,True,False]);c.manual_target=torch.tensor([False,True,False])
        c.turn_rate=torch.tensor([.7,-.8,0.]);c.vel_command_b=torch.ones(3,3)
        c.heading_command_w=torch.ones(3);c.reached_target=torch.ones(3,dtype=torch.bool)
        c._update_selected(slice(None))
        torch.testing.assert_close(c.vel_command_b[0],torch.tensor([0.,0.,.7]))
        torch.testing.assert_close(c.vel_command_b[1:],torch.ones(2,3))
        self.assertFalse(c.reached_target[0])


if __name__=='__main__':unittest.main()
