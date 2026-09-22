import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch
from legged_lab.tasks.locomotion.amp.mdp.stair_course import (
    specialist_layout, stair_speed_ranges, advance_stage, settled_at_exit,
    ENV_STATE, GLOBAL_STATE, STAIR_HEIGHTS, StairTargetCommand, stair_curriculum,
    ExitWindow, completion_success,
)
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_balanced_env_cfg import G1AmpBalancedEnvCfg
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_stairs_env_cfg import G1AmpStairsEnvCfg, specialist_stairs
from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_stairs_ppo_cfg import G1AmpStairsPPORunnerCfg
from legged_lab.tasks.locomotion.amp.mdp.commands.target_velocity import terrain_column_names
from legged_lab.rsl_rl.amp.stairs_ppo import StairsPPOAMP, restore_indices
from legged_lab.rsl_rl.amp.ppo_amp import PPOAMP


class TestStairCourse(unittest.TestCase):
    def setUp(self):
        self.cfg = G1AmpStairsEnvCfg()
        self.names = terrain_column_names(self.cfg.scene.terrain.terrain_generator)

    def test_distribution_direction_and_full_flat_speed(self):
        g, col = specialist_layout(2000, self.names, 'cpu')
        self.assertEqual(torch.bincount(g).tolist(), [500,700,400,400])
        self.assertEqual({self.names[i] for i in col[g == 1]}, {'pyramid_stairs_inv'})
        self.assertEqual({self.names[i] for i in col[g == 2]}, {'pyramid_stairs'})
        self.assertEqual(len({self.names[i] for i in col[g == 3]}), 4)
        torch.testing.assert_close(stair_speed_ranges(torch.tensor([0,1])), torch.tensor([[.35,.55],[.55,.75]]))

    def test_rewards_references_and_camera_unchanged(self):
        self.cfg.validate()
        self.assertFalse(self.cfg.randomize_initial_episode_length)
        old = G1AmpBalancedEnvCfg()
        for name in ('motion_data', 'animation', 'observations', 'rewards', 'terminations'):
            self.assertEqual(getattr(old,name).to_dict(),getattr(self.cfg,name).to_dict())
        a, b = old.scene.to_dict(), self.cfg.scene.to_dict()
        a.pop('terrain'); b.pop('terrain')
        self.assertEqual(a,b)
        runner = G1AmpStairsPPORunnerCfg()
        self.assertEqual(runner.algorithm.amp_cfg.amp_discriminator.style_reward_scale,.5)

    def test_actual_stair_geometry_and_flat_row(self):
        gen = self.cfg.scene.terrain.terrain_generator
        for name, sign in (('pyramid_stairs',1),('pyramid_stairs_inv',-1)):
            cfg = gen.sub_terrains[name].copy(); cfg.size = gen.size
            flat, origin = specialist_stairs(.05,cfg)
            self.assertTrue(all(np.allclose(m.vertices[:,2],0) for m in flat))
            for row in (1,2,6,9):
                meshes, origin = specialist_stairs((row+.5)/10,cfg)
                # Check the actual mesh rather than only the configuration.
                self.assertEqual(np.sign(origin[2]), sign)
                heights = np.unique(np.round(np.concatenate([m.vertices[:,2] for m in meshes]),6))
                self.assertTrue(np.any(np.isclose(np.diff(heights),STAIR_HEIGHTS[row])))

    def test_height_and_speed_do_not_increase_together(self):
        self.assertEqual(advance_stage(2,0,49,49),(2,0,False))
        self.assertEqual(advance_stage(2,0,40,50),(2,0,True))
        self.assertEqual(advance_stage(2,0,41,50),(2,1,True))
        self.assertEqual(advance_stage(2,1,41,50),(3,0,True))
        self.assertEqual(advance_stage(2,1,24,50),(2,0,True))
        self.assertEqual(advance_stage(2,0,24,50),(1,0,True))
        self.assertEqual(advance_stage(1,0,0,50),(1,0,True))
        self.assertEqual(advance_stage(9,1,50,50),(9,1,True))

    def test_arrival_rejects_under_stair_speed_tilt_and_edge(self):
        n=6
        root=torch.tensor([[3.7,0.,.75]]*n)
        feet=torch.tensor([[[3.5,-.1,.05],[3.8,.1,.05]]]*n)
        goal=torch.tensor([[3.7,0.,0.]]*n)
        v=torch.zeros(n,3); w=torch.zeros(n,3); gravity=torch.tensor([[0.,0.,-1.]]*n)
        feet[1,:,2] = -.2
        v[2,0]=1.8
        gravity[3,2]=-.5
        feet[4,1,0]=4.1
        feet[5,0,0]=2.9
        self.assertEqual(settled_at_exit(root,feet,goal,v,w,gravity,4.).tolist(),[True,False,False,False,False,False])

    def algorithm_fixture(self,n=100):
        group, columns = specialist_layout(n,self.names,'cpu')
        c=NS(group=group, column_names=self.names, flat_pool=group==0, stairs=(group==1)|(group==2),
             skip_curriculum_once=torch.zeros(n,dtype=torch.bool))
        for name in ENV_STATE:
            if not hasattr(c,name): setattr(c,name,torch.zeros(n,dtype=torch.long))
        for name in GLOBAL_STATE:
            setattr(c,name,torch.zeros(2,dtype=torch.float32 if name=='last_rate' else torch.long))
        c.frontier[:]=2
        t=NS(terrain_types=columns,terrain_levels=torch.zeros(n,dtype=torch.long),max_terrain_level=10,
             terrain_origins=torch.randn(10,10,3),env_origins=torch.zeros(n,3))
        c.terrain=t
        c.sync_origins=lambda:t.env_origins.copy_(t.terrain_origins[t.terrain_levels,t.terrain_types])
        env=NS(device='cpu',scene=NS(terrain=t),command_manager=NS(get_term=lambda _:c),reset=lambda:None)
        alg=object.__new__(StairsPPOAMP);alg.course_env=env
        alg.actor=NS(distribution=NS(std_type='scalar',std_param=torch.nn.Parameter(torch.tensor([.3])),std_range=(.05,.5)))
        alg.optimizer=NS(state={})
        return alg,c,t

    def test_v2_migration_and_v4_roundtrip(self):
        alg,c,t=self.algorithm_fixture()
        n=200
        state={'version':2,'rows':10,'columns':self.names,'levels':torch.full((n,),7),
               'flat_pool':torch.arange(n)%4==0,'flat_speed_tier':torch.full((n,),3),
               'speed_good':torch.ones(n,dtype=torch.long),'speed_bad':torch.zeros(n,dtype=torch.long)}
        original=state['levels'].clone()
        with patch.object(PPOAMP,'load',return_value=True) as load:
            alg.load({'scratch_course':state}); self.assertEqual(load.call_count,1)
        self.assertTrue((t.terrain_levels[c.stairs]==2).all())
        self.assertTrue((t.terrain_levels[c.group==3]==7).all())
        self.assertTrue((t.terrain_levels[c.flat_pool]==0).all())
        self.assertTrue((c.flat_speed_tier[c.flat_pool]==3).all())
        torch.testing.assert_close(state['levels'],original)
        c.frontier[:]=torch.tensor([5,3]); c.attempts[:]=torch.tensor([31,29]); c.wins[:]=torch.tensor([28,17])
        with patch.object(PPOAMP,'save',return_value={}): saved=alg.save()
        self.assertEqual(saved['scratch_course']['version'],4)
        c.frontier.zero_();c.wins.zero_()
        with patch.object(PPOAMP,'load',return_value=True): alg.load(saved)
        self.assertEqual(c.frontier.tolist(),[5,3]);self.assertEqual(c.wins.tolist(),[28,17])
        # A smaller Viser batch samples within explicit groups/columns.
        small,sc,st=self.algorithm_fixture(20)
        with patch.object(PPOAMP,'load',return_value=True):small.load(saved)
        self.assertTrue((sc.flat_speed_tier[sc.flat_pool]==3).all())
        self.assertEqual(sc.frontier.tolist(),[5,3])

    def test_v3_migration_preserves_stages_but_clears_old_evidence(self):
        alg,c,t=self.algorithm_fixture()
        c.frontier[:]=torch.tensor([4,2]);c.phase[:]=torch.tensor([1,0])
        c.flat_speed_tier[c.flat_pool]=3
        c.attempts[:]=49;c.wins[:]=3;c.last_rate[:]=.06
        c.epoch[:]=8;c.stair_epoch[:]=8
        t.terrain_levels[c.stairs]=4
        with patch.object(PPOAMP,'save',return_value={}):saved=alg.save()
        saved['scratch_course']['version']=3
        with patch.object(PPOAMP,'load',return_value=True):alg.load(saved)
        self.assertEqual(c.frontier.tolist(),[4,2]);self.assertEqual(c.phase.tolist(),[1,0])
        self.assertTrue((t.terrain_levels[c.stairs]==4).all())
        self.assertTrue((c.flat_speed_tier[c.flat_pool]==3).all())
        self.assertEqual(c.attempts.sum().item(),0);self.assertEqual(c.wins.sum().item(),0)
        self.assertTrue((c.last_rate==-1).all());self.assertEqual(c.stair_epoch.sum().item(),0)
        self.assertEqual(saved['scratch_course']['attempts'].tolist(),[49,49])

    def test_window_requires_full_duration_and_tolerates_brief_perturbations(self):
        window=ExitWindow(3,.02,'cpu');valid=torch.ones(3,dtype=torch.bool)
        for i in range(50):
            geometry=torch.tensor([i%10!=0,True,True]) # exactly 90% in first lane
            speed=torch.tensor([.1, .6 if i%10==0 else .1, .5])
            passed=window.update(geometry,speed,valid)
            if i<49:self.assertFalse(passed.any())
        self.assertEqual(passed.tolist(),[True,True,False])
        # An invalid current pose cannot pass based solely on previous history.
        self.assertFalse(window.update(torch.tensor([False,True,True]),torch.zeros(3),valid)[0])
        window.reset(torch.tensor([1]))
        for _ in range(49):
            passed=window.update(torch.ones(3,dtype=torch.bool),torch.zeros(3),valid)
            self.assertFalse(passed[1])
        self.assertTrue(window.update(torch.ones(3,dtype=torch.bool),torch.zeros(3),valid)[1])

    def test_window_rejects_nan_and_missing_geometry_then_recovers(self):
        w=ExitWindow(2,.02,'cpu');yes=torch.ones(2,dtype=torch.bool)
        for _ in range(50):
            passed=w.update(torch.tensor([False,True]),torch.tensor([0.,float('nan')]),yes)
        self.assertFalse(passed.any())
        w.reset(torch.arange(2))
        for _ in range(50):passed=w.update(yes,torch.zeros(2),yes)
        self.assertTrue(passed.all())

    def test_completion_rejects_fall_or_leaving_exit_after_earlier_settling(self):
        self.assertEqual(completion_success(torch.tensor([True,True,True,False]),
            torch.tensor([False,True,False,False]),torch.tensor([True,True,False,True])).tolist(),
            [True,False,False,False])

    def test_flat_migration_samples_whole_pool_without_column_bias(self):
        n=2000
        state={'version':2,'columns':self.names,'levels':torch.zeros(n,dtype=torch.long),
               'flat_pool':torch.arange(n)%4==0}
        group,columns=specialist_layout(n,self.names,'cpu')
        indices=restore_indices(state,group,columns)
        torch.testing.assert_close(indices[group==0],torch.where(state['flat_pool'])[0])

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires GPU checkpoint mapping')
    def test_gpu_checkpoint_layout_supports_smaller_playback(self):
        alg,c,t=self.algorithm_fixture(100)
        with patch.object(PPOAMP,'save',return_value={}): state=alg.save()['scratch_course']
        g,col=specialist_layout(20,self.names,'cpu')
        expected=restore_indices(state,g,col)
        gpu={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in state.items()}
        torch.testing.assert_close(restore_indices(gpu,g.cuda(),col.cuda()),expected)
        torch.testing.assert_close(restore_indices(gpu,c.group.cuda(),t.terrain_types.cuda()),torch.arange(100))

    def test_curriculum_ignores_stale_manual_initial_and_failed_success(self):
        alg,c,t=self.algorithm_fixture(200)
        c._ids=lambda ids:torch.as_tensor(ids,dtype=torch.long)
        c.stair_manual_episode=torch.zeros(200,dtype=torch.bool)
        c.stair_hold=torch.full((200,),1.2)
        c.stair_completed=torch.ones(200,dtype=torch.bool)
        c.stair_traversed=torch.ones(200,dtype=torch.bool)
        c.stair_strict_settled=torch.zeros(200,dtype=torch.bool)
        c.exit_geometry=lambda:torch.ones(200,dtype=torch.bool)
        t.terrain_levels[c.stairs]=2
        stairs=torch.where(c.group==1)[0] # 70 ascents, 40 descents
        c.stair_epoch[stairs[:10]]=9
        c.stair_manual_episode[stairs[10:15]]=True
        c.skip_curriculum_once[stairs[15:20]]=True
        # 50 eligible attempts, 41 successes -> faster at same height.
        failed=torch.zeros(200,dtype=torch.bool);failed[stairs[20:29]]=True
        alg.course_env.termination_manager=NS(terminated=failed)
        with patch('legged_lab.tasks.locomotion.amp.mdp.stair_course.balanced_curriculum',return_value={}):
            result=stair_curriculum(alg.course_env,torch.arange(200))
        self.assertEqual(c.frontier.tolist(),[2,2])
        self.assertEqual(c.phase.tolist(),[1,0])
        self.assertEqual(c.attempts.tolist(),[0,40])
        self.assertAlmostEqual(result['ascent/window_pass_rate'].item(),.82,places=5)
        self.assertEqual(result['ascent/batch_failures'].item(),9)


if __name__=='__main__': unittest.main()
