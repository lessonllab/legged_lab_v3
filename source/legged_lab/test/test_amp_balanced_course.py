import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from legged_lab.tasks.locomotion.amp.mdp.balanced_course import update_evidence, flat_speed_ranges, BalancedTargetCommand, instinct_terrain_decisions, balanced_curriculum
from legged_lab.tasks.locomotion.amp.mdp.scratch_curriculum import ScratchTargetCommand
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_scratch_env_cfg import G1AmpScratchEnvCfg
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_balanced_env_cfg import G1AmpBalancedEnvCfg
from legged_lab.rsl_rl.amp.balanced_ppo import BalancedPPOAMP, clamp_loaded_exploration
from legged_lab.rsl_rl.amp.ppo_amp import PPOAMP


class TestBalancedCourse(unittest.TestCase):
    def test_instinct_threshold_boundaries(self):
        xy = torch.tensor([.7,.6,.3,.299,.7,.7,float('nan')])
        yaw = torch.tensor([.1,1.,1.,1.,0.,-.1,1.])
        up, down = instinct_terrain_decisions(xy,yaw)
        self.assertEqual(up.tolist(),[True,False,False,False,False,False,False])
        self.assertEqual(down.tolist(),[False,False,False,True,False,False,False])

    def test_terrain_uses_full_horizon_without_extra_gates_or_top_level_clamp(self):
        n=5
        cmd=NS(_ids=lambda ids:torch.as_tensor(ids),flat_pool=torch.tensor([False,False,False,True,False]),
               skip_curriculum_once=torch.tensor([False,False,False,False,True]))
        for name in ('success_streak','failure_streak','flat_speed_tier','speed_good','speed_bad'):
            setattr(cmd,name,torch.zeros(n,dtype=torch.long))
        cmd.metrics={key:torch.zeros(n) for key in ('moving_tracking_exp_vel_xy','moving_tracking_exp_vel_yaw',
                                                    'target_progress_m','moving_opportunity_s')}
        cmd.metrics.update(tracking_exp_vel_xy=torch.tensor([.7,.2,.7,.7,.7]),tracking_exp_vel_yaw=torch.ones(n))
        changes=[]
        terrain=NS(terrain_levels=torch.tensor([1,1,9,0,2]),max_terrain_level=10,
                   update_env_origins=lambda ids,up,down:changes.append((up.clone(),down.clone())))
        env=NS(scene=NS(terrain=terrain),command_manager=NS(get_term=lambda _:cmd),
               termination_manager=NS(terminated=torch.ones(n,dtype=torch.bool),time_outs=torch.zeros(n,dtype=torch.bool)))
        balanced_curriculum(env,list(range(n)))
        self.assertEqual(changes[0][0].tolist(),[True,False,True,False,False])
        self.assertEqual(changes[0][1].tolist(),[False,True,False,False,False])

    def test_exploration_clamp_preserves_other_optimizer_state(self):
        parameter = torch.nn.Parameter(torch.tensor([.8, .3]))
        other = torch.nn.Parameter(torch.ones(1))
        actor = NS(distribution=NS(std_type='scalar', std_param=parameter, std_range=(.05,.5)))
        optimizer = NS(state={parameter: {'old': 1}, other: {'keep': 2}})
        self.assertTrue(clamp_loaded_exploration(actor, optimizer))
        torch.testing.assert_close(parameter, torch.tensor([.5,.3]))
        self.assertNotIn(parameter,optimizer.state)
        self.assertEqual(optimizer.state[other], {'keep': 2})

    def test_neutral_does_not_erase_success_and_one_fall_does_not_demote(self):
        good = bad = torch.zeros(1, dtype=torch.long)
        t, f = torch.tensor([True]), torch.tensor([False])
        good, bad, up, down = update_evidence(good, bad, t, f, t)
        self.assertFalse(up.item())
        good, bad, up, down = update_evidence(good, bad, f, f, t)
        self.assertEqual(good.item(), 1)
        good, bad, up, down = update_evidence(good, bad, t, f, t)
        self.assertTrue(up.item())
        good, bad, up, down = update_evidence(good, bad, f, t, t)
        self.assertFalse(down.item())
        good, bad, up, down = update_evidence(good, bad, f, t, t)
        self.assertTrue(down.item())
        _, _, up, down = update_evidence(torch.tensor([1]), torch.tensor([1]), t, t, f)
        self.assertFalse(up.item() or down.item())

    def test_flat_fast_commands_and_reference_rewards_preserved(self):
        bounds = flat_speed_ranges(torch.arange(4))
        torch.testing.assert_close(bounds[:, 1], torch.tensor([.6, 1., 1.5, 2.]))
        self.assertGreater(bounds[3, 0].item(), 1.)
        old, new = G1AmpScratchEnvCfg(), G1AmpBalancedEnvCfg()
        new.validate()
        for name in ('motion_data', 'animation', 'observations', 'rewards', 'terminations'):
            self.assertEqual(getattr(old, name).to_dict(), getattr(new, name).to_dict())
        self.assertEqual(old.scene.terrain.to_dict(), new.scene.terrain.to_dict())
        # The command changes, but task rewards, references and sensors do not.

    def test_flat_arrival_retargets_but_manual_and_standing_are_kept(self):
        cmd = object.__new__(BalancedTargetCommand)
        cmd.terrain = NS(terrain_levels=torch.tensor([0, 0, 0, 1]))
        cmd.reached_target = torch.ones(4, dtype=torch.bool)
        cmd.is_standing_env = torch.tensor([False, True, False, False])
        cmd.manual_target = torch.tensor([False, False, True, False])
        selected = []
        cmd._resample = lambda ids: selected.extend(ids.tolist())
        with patch.object(ScratchTargetCommand, '_update_command'):
            cmd._update_command()
        self.assertEqual(selected, [0])

    def test_flat_sampling_alternates_endpoints_and_applies_independent_speed(self):
        cmd = object.__new__(BalancedTargetCommand)
        cmd._env = NS(device='cpu',num_envs=3,scene=NS(env_origins=torch.zeros(3,3)))
        cmd.terrain = NS(terrain_levels=torch.zeros(3,dtype=torch.long))
        cmd.robot = NS(data=NS(root_pos_w=torch.tensor([[0.,0.,0.],[2.7,0.,0.],[-2.7,0.,0.]])))
        cmd.manual_target = torch.zeros(3,dtype=torch.bool)
        cmd.flat_speed_tier = torch.tensor([0,1,3])
        cmd.speed_cap = torch.zeros(3)
        cmd.pos_command_w = torch.zeros(3,3)
        cmd._begin_target = lambda ids: None
        with patch.object(ScratchTargetCommand, '_resample_command'):
            cmd._resample_command(torch.arange(3))
        torch.testing.assert_close(cmd.pos_command_w[:,0],torch.tensor([3.,-3.,3.]))
        self.assertTrue(.45 <= cmd.speed_cap[0] <= .6)
        self.assertTrue(.65 <= cmd.speed_cap[1] <= 1.)
        self.assertTrue(1.2 <= cmd.speed_cap[2] <= 2.)

    def test_migration_and_roundtrip_preserve_policy_load_and_speed_tiers(self):
        n=8
        cmd = NS(column_names=['a', 'b'], flat_pool=torch.arange(n)%4==0,
                 skip_curriculum_once=torch.zeros(n,dtype=torch.bool))
        for name in ('success_streak','failure_streak','flat_speed_tier','speed_good','speed_bad'):
            setattr(cmd,name,torch.zeros(n,dtype=torch.long))
        terrain=NS(terrain_levels=torch.ones(n,dtype=torch.long),terrain_types=torch.arange(n)//4,
                   terrain_origins=torch.randn(10,2,3),env_origins=torch.zeros(n,3),max_terrain_level=10)
        env=NS(device='cpu',scene=NS(terrain=terrain),command_manager=NS(get_term=lambda _:cmd),reset=lambda:None)
        alg=object.__new__(BalancedPPOAMP);alg.course_env=env
        alg.actor=NS(distribution=NS(std_type='scalar',std_param=torch.nn.Parameter(torch.full((2,),.3)),std_range=(.05,.5)))
        alg.optimizer=NS(state={})
        state={'scratch_course':{'version':1,'levels':torch.ones(n,dtype=torch.long),'streak':torch.ones(n,dtype=torch.long),
                                'rows':10,'columns':['a','b']}}
        with patch.object(PPOAMP,'load',return_value=True) as load:
            alg.load(state);self.assertEqual(load.call_count,1)
        torch.testing.assert_close(terrain.terrain_levels,torch.tensor([0,1,1,1,0,1,1,1]))
        self.assertEqual(cmd.flat_speed_tier.sum().item(),0)
        self.assertEqual(state['scratch_course']['levels'].sum().item(),8)
        cmd.flat_speed_tier[0]=3;cmd.speed_good[0]=1
        with patch.object(PPOAMP,'save',return_value={}): saved=alg.save()
        self.assertEqual(saved['scratch_course']['version'],2)
        cmd.flat_speed_tier.zero_();cmd.speed_good.zero_()
        with patch.object(PPOAMP,'load',return_value=True):alg.load(saved)
        self.assertEqual(cmd.flat_speed_tier[0].item(),3)
        self.assertEqual(cmd.speed_good[0].item(),1)


if __name__ == '__main__':
    unittest.main()
