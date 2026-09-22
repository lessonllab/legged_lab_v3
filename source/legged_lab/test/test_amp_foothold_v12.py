import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from legged_lab.tasks.locomotion.amp.mdp.foothold_rewards import progress_tracking, loaded_point_speed, lookahead_height, step_credit, navigation_heading_error
from legged_lab.tasks.locomotion.amp.mdp.foothold_course import foothold_layout,FootholdCourseMixin,SLOW
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_foothold_env_cfg import G1AmpFootholdEnvCfg
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_stairs_long_env_cfg import G1AmpStairsLongEnvCfg
from legged_lab.rsl_rl.amp.ppo_amp import PPOAMP
import test_amp_stair_speed as fixtures


class TestFootholdV12(unittest.TestCase):
    def test_standing_cannot_collect_moving_tracking_at_low_speed_or_reverse(self):
        command=torch.tensor([[.2,0.,0.],[-.2,0.,0.],[0.,0.,.6],[0.,0.,0.]])
        stopped=torch.zeros(4,2);yaw=torch.zeros(4)
        torch.testing.assert_close(progress_tracking(command,stopped,yaw),torch.tensor([0.,0.,1.,1.]))
        torch.testing.assert_close(progress_tracking(command,stopped,yaw,angular=True),torch.tensor([0.,0.,0.,1.]))
        torch.testing.assert_close(progress_tracking(command,command[:,:2],command[:,2]),torch.ones(4))
        torch.testing.assert_close(progress_tracking(command,command[:,:2],command[:,2],angular=True),torch.ones(4))
        self.assertEqual(progress_tracking(command[:1],-command[:1,:2],yaw[:1]).item(),0.)

    def test_contact_spin_without_center_translation_is_detected(self):
        v=torch.zeros(3,2,3);w=v.clone();w[:,:,2]=1.
        offset=torch.tensor([[[[.1,0.,0.],[-.1,0.,0.]]]]).expand(3,2,2,3)
        load=torch.tensor([[1.,1.],[0.,0.],[1.,0.]])
        torch.testing.assert_close(loaded_point_speed(v,w,offset,load),torch.tensor([.2,0.,.1]))

    def test_path_uses_nearest_lip_and_ignores_downhill_smooth_slope_and_misses(self):
        heights=torch.zeros(5,2,8,3)
        heights[0,:,4:]=.2;heights[0,:,6:]=.4
        heights[1]=.2;heights[1,:,4:]=0.
        heights[2]=torch.arange(8)[None,:,None]*.02
        heights[3,:,4:]=float('inf')
        heights[4,:,:3]=.15
        forward=torch.ones(5,2,dtype=torch.bool);forward[4]=False
        target,active=lookahead_height(heights,forward)
        self.assertEqual(active[:,0].tolist(),[True,False,False,False,True])
        torch.testing.assert_close(target[:,0],torch.tensor([.2,0.,0.,0.,.15]))
        self.assertTrue(torch.isfinite(target).all())

    def test_step_credit_requires_turn_clearance_motion_and_alternation(self):
        touchdown=torch.tensor([[True,False]]).expand(7,-1).clone()
        air=torch.full((7,2),.2);yaw=torch.full((7,2),.1);clear=torch.full((7,2),.05);slip=torch.zeros(7,2)
        last=torch.full((7,),1);turn=torch.ones(7,dtype=torch.bool)
        air[1]=.02;clear[2]=0.;yaw[3]=0.;last[4]=0;slip[5]=.5;turn[6]=False
        good=step_credit(touchdown,air,yaw,clear,slip,last,turn)
        self.assertEqual(good.any(-1).tolist(),[True,False,False,False,False,False,False])


    def test_bilateral_landing_never_earns_turn_step_credit(self):
        both=torch.ones(1,2,dtype=torch.bool)
        credit=step_credit(both,torch.full((1,2),.2),torch.full((1,2),.1),
                           torch.full((1,2),.05),torch.zeros(1,2),torch.tensor([1]),torch.tensor([True]))
        self.assertFalse(credit.any())


    def test_rear_target_keeps_heading_cost_but_pure_rate_does_not(self):
        c=NS(command=torch.tensor([[0.,0.,.6],[0.,0.,.6],[0.,0.,.6]]),
             turn_pool=torch.tensor([True,False,True]),manual_target=torch.tensor([False,False,True]))
        env=NS(command_manager=NS(get_term=lambda name:c))
        torch.testing.assert_close(navigation_heading_error(env),torch.tensor([0.,.6,.6]))

    def test_layout_keeps_stairs_and_doubles_boxes(self):
        names=['pyramid_stairs']*2+['pyramid_stairs_inv']*2+['boxes']*2+['random_rough']*2+['hf_pyramid_slope','hf_pyramid_slope_inv']
        groups,cols=foothold_layout(1200,names,'cpu')
        self.assertEqual(torch.bincount(groups).tolist(),[300,300,240,360])
        for name,count in [('boxes',180),('random_rough',120)]:
            selected=torch.tensor([n==name for n in names])[cols] & (groups==3)
            self.assertEqual(int(selected.sum()),count)

    def test_slow_speed_survives_timer_resampling_and_reset_with_old_episode_buffer(self):
        class Base:
            def _resample_command(self,ids):
                ids=torch.as_tensor(ids)
                fresh=ids[~self.stair_started[ids]]
                self.speed_cap[fresh]=.65;self.stair_started[fresh]=True
        class Command(FootholdCourseMixin,Base):pass
        c=object.__new__(Command);c.device="cpu";c._ids=lambda ids:torch.tensor(ids)
        c.stair_started=torch.tensor([False,True,False]);c.stair_lane=torch.tensor([SLOW,SLOW,0]);c.stairs=torch.ones(3,dtype=torch.bool)
        c.manual_target=torch.zeros(3,dtype=torch.bool);c.speed_cap=torch.tensor([.65,.25,.65]);c._update_selected=lambda ids:None
        c._env=NS(episode_length_buf=torch.tensor([2300,500,2300]))
        c._resample_command([0,1,2])
        self.assertTrue(.2<=c.speed_cap[0]<=.5);self.assertEqual(float(c.speed_cap[1]),.25)
        self.assertAlmostEqual(float(c.speed_cap[2]),.65)

    def test_new_task_is_opt_in_and_preserves_observation_contract(self):
        old=G1AmpStairsLongEnvCfg();new=G1AmpFootholdEnvCfg()
        self.assertEqual(old.commands.base_velocity.class_type.promotion_version,11)
        self.assertEqual(new.commands.base_velocity.class_type.promotion_version,12)
        self.assertEqual(new.observations.to_dict(),old.observations.to_dict())
        self.assertEqual(new.rewards.is_alive.weight,.6)
        self.assertIsNone(new.rewards.dont_wait)
        self.assertTrue(hasattr(new.scene,'left_path_scanner'))

    def test_migration_preserves_height_frontier_and_speed_but_clears_old_evidence(self):
        old,oc,ot=fixtures.TestStairSpeed().fixture(120)
        oc.layout_profile='agility_v10';oc.promotion_version=11;oc.frontier[:]=7;oc.phase[:]=4
        with patch.object(PPOAMP,'save',return_value={}):saved=old.save()
        new,c,t=fixtures.TestStairSpeed().fixture(120)
        c.layout_profile='foothold_v12';c.promotion_version=12
        with patch.object(PPOAMP,'load',return_value=True):new.load(saved)
        self.assertEqual(c.frontier.tolist(),[7,7]);self.assertEqual(c.phase.tolist(),[4,4])
        self.assertEqual(c.attempts.sum(),0)

if __name__=='__main__':unittest.main()
