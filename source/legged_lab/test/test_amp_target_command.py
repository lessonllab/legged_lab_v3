import unittest
from types import SimpleNamespace as NS
import torch
from legged_lab.tasks.locomotion.amp.mdp.commands.target_velocity import (
    target_velocity, terrain_column_names, TargetVelocityCommand, target_tracking_curriculum,
    target_progress_curriculum)
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_target_env_cfg import (
    TargetVelocityCommandCfg, G1AmpDepthTargetEnvCfg, G1AmpDepthTargetEnvCfg_PLAY)
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_style_env_cfg import G1AmpDepthStyleEnvCfg


class TestTargetCommand(unittest.TestCase):
    def test_target_geometry_and_stop(self):
        pos = torch.zeros(6, 3)
        quat = torch.tensor([[0., 0., 0., 1.]]).repeat(6, 1)
        quat[1] = torch.tensor([0., 0., 2**-.5, 2**-.5])
        goals = torch.tensor([[2.,0.,0.], [0.,2.,0.], [-2.,0.,0.],
                              [0.,2.,0.], [.2,0.,0.], [2.,0.,0.]])
        cap = torch.full((6,), .6)
        stand = torch.tensor([False]*5+[True])
        v, h, dist, reached = target_velocity(pos, quat, goals, cap, stand)
        torch.testing.assert_close(v[:2], torch.tensor([[.6,0.,0.], [.6,0.,0.]]), atol=1e-6, rtol=1e-6)
        self.assertEqual(v[2,0], 0.)  # target behind: turn, do not walk backwards
        self.assertEqual(abs(v[2,2]), 1.)
        torch.testing.assert_close(v[3], torch.tensor([0.,0.,1.]))
        self.assertTrue((v[4:] == 0).all())
        self.assertTrue(reached[4]); self.assertFalse(reached[5])
        # All geometry is invariant to global world translation.
        offset = torch.tensor([100., -80., 3.])
        shifted = target_velocity(pos+offset, quat, goals+offset, cap, stand)
        torch.testing.assert_close(shifted[0], v)

    def test_actual_ranges_and_v3_preserved(self):
        new, old, play = G1AmpDepthTargetEnvCfg(), G1AmpDepthStyleEnvCfg(), G1AmpDepthTargetEnvCfg_PLAY()
        new.validate(); play.validate()
        self.assertEqual(old.commands.base_velocity.ranges.lin_vel_x, (.2, 1.))
        self.assertEqual(new.commands.base_velocity.resampling_time_range, (8.,12.))
        self.assertEqual(new.commands.base_velocity.speed_ranges['random_rough'], (.45,1.))
        self.assertEqual(new.rewards.to_dict(), old.rewards.to_dict())
        self.assertEqual(new.observations.to_dict(), old.observations.to_dict())
        self.assertIsNone(play.curriculum.terrain_levels)
        self.assertTrue(play.scene.terrain.terrain_generator.curriculum)
        columns = terrain_column_names(new.scene.terrain.terrain_generator)
        self.assertEqual([columns.count(n) for n in dict.fromkeys(columns)], [4,4,4,4,2,2])
        for terrain in new.scene.terrain.terrain_generator.sub_terrains.values():
            self.assertEqual(terrain.flat_patch_sampling['target'].num_patches,50)
        new.scene.terrain.terrain_generator.curriculum = False
        with self.assertRaises(ValueError): terrain_column_names(new.scene.terrain.terrain_generator)

    def make_env(self):
        cfg = G1AmpDepthTargetEnvCfg()
        generator = cfg.scene.terrain.terrain_generator
        targets = torch.zeros(10,20,50,3)
        targets[...,0] = torch.arange(10)[:,None,None]*8 + 2
        targets[...,1] = torch.arange(20)[None,:,None]*8
        terrain = NS(cfg=NS(terrain_generator=generator), flat_patches={'target':targets},
                     terrain_types=torch.tensor([0,13]), terrain_levels=torch.tensor([0,3]))
        data = NS(root_pos_w=torch.zeros(2,3),root_quat_w=torch.tensor([[0.,0.,0.,1.]]).repeat(2,1),
                  root_lin_vel_b=torch.zeros(2,3), root_ang_vel_b=torch.zeros(2,3))
        class Scene(dict): pass
        scene = Scene(robot=NS(data=data));scene.terrain=terrain
        env = NS(scene=scene,device='cpu',num_envs=2,max_episode_length=1000,step_dt=.02,
                 episode_length_buf=torch.zeros(2,dtype=torch.long))
        return env

    def make_active_command(self):
        env = self.make_env()
        command = TargetVelocityCommand(TargetVelocityCommandCfg(rel_standing_envs=0.), env)
        command.reset()
        command._update_metrics()  # post-reset observation: no physical transition
        return env, command

    def step_command(self, env, command, x):
        env.episode_length_buf += 1
        env.scene['robot'].data.root_pos_w[0, 0] = x
        command._update_metrics()
        command._update_command()

    def test_partial_resample_world_coordinates_and_reset(self):
        env = self.make_env()
        cfg = TargetVelocityCommandCfg(rel_standing_envs=0.)
        command = TargetVelocityCommand(cfg,env)
        command.reset()
        torch.testing.assert_close(command.pos_command_w,torch.tensor([[2.,0.,0.],[26.,104.,0.]]))
        first_target = command.pos_command_w[0].clone()
        first_velocity = command.command[0].clone()
        env.scene.terrain.terrain_levels[1] = 5
        command.reset(torch.tensor([1]))
        self.assertEqual(command.pos_command_w[1,0],42.)
        torch.testing.assert_close(command.pos_command_w[0],first_target)
        torch.testing.assert_close(command.command[0],first_velocity)
        self.assertTrue(((command.time_left>=8)&(command.time_left<=12)).all())
        self.assertTrue((command.command[:,0] > 0).all())
        for _ in range(100):
            command._resample_command(torch.tensor([0,1]))
            self.assertTrue(.45 <= command.speed_cap[0] <= .8)
            self.assertTrue(.45 <= command.speed_cap[1] <= 1.)
        env.scene['robot'].data.root_pos_w[:] = command.pos_command_w
        command._update_command();command._update_command()
        self.assertTrue((command.command == 0).all())
        self.assertTrue((command.metrics['targets_reached'] == 1).all())

    def test_curriculum_scores_and_partial_update(self):
        captured = []
        terrain = NS(terrain_levels=torch.zeros(4),update_env_origins=lambda ids,up,down:captured.append((ids,up,down)))
        command = NS(metrics={'tracking_exp_vel_xy':torch.tensor([.7,.2,.4,.9]),
                              'tracking_exp_vel_yaw':torch.tensor([.8,.2,.4,0.])})
        env = NS(scene=NS(terrain=terrain),command_manager=NS(get_term=lambda _:command))
        target_tracking_curriculum(env,torch.tensor([0,1,3]))
        self.assertEqual(captured[0][1].tolist(),[True,False,False])
        self.assertEqual(captured[0][2].tolist(),[False,True,False])
        env = self.make_env();cmd=TargetVelocityCommand(TargetVelocityCommandCfg(),env)
        cmd.vel_command_b.zero_()
        cmd._update_metrics()
        torch.testing.assert_close(cmd.metrics['tracking_exp_vel_xy'],torch.full((2,),.001))

    def test_valid_arrival_progress_and_arrival_dwell(self):
        env, command = self.make_active_command()
        for step in range(1, 43):
            self.step_command(env, command, step * .04)
        self.assertEqual(command.metrics['valid_targets_reached'][0], 1.)
        self.assertGreaterEqual(command.metrics['target_progress_m'][0], 1.6)
        opportunity = command.metrics['moving_opportunity_s'][0].clone()
        for _ in range(30):
            self.step_command(env, command, 1.8)
        torch.testing.assert_close(command.metrics['moving_opportunity_s'][0], opportunity)
        self.assertEqual(command.metrics['valid_targets_reached'][0], 1.)
        # Leaving/re-entering an already completed target cannot add arrivals.
        self.step_command(env, command, .2)
        self.step_command(env, command, 1.8)
        self.assertEqual(command.metrics['valid_targets_reached'][0], 1.)

    def test_spawn_in_radius_and_standing_are_not_movement(self):
        env = self.make_env()
        env.scene['robot'].data.root_pos_w[0, 0] = 2.
        command = TargetVelocityCommand(TargetVelocityCommandCfg(rel_standing_envs=0.), env)
        command.reset()
        command.is_standing_env[1] = True
        command._update_command()
        command._update_metrics()
        for _ in range(5):
            self.step_command(env, command, 1.8)
        self.assertEqual(command.metrics['targets_reached'][0], 1.)  # legacy metric preserved
        self.assertTrue((command.metrics['valid_targets_reached'] == 0).all())
        self.assertTrue((command.metrics['target_progress_m'] == 0).all())
        self.assertTrue((command.metrics['moving_opportunity_s'] == 0).all())

    def test_progress_is_net_approach_and_resampling_does_not_add_it(self):
        env, command = self.make_active_command()
        for x in [.2, .6, .1, .6, .2, .6]:
            self.step_command(env, command, x)
        torch.testing.assert_close(command.metrics['target_progress_m'][0], torch.tensor(.6))
        command._resample_command(torch.tensor([0]))
        torch.testing.assert_close(command.target_initial_distance[0], torch.tensor(1.4))
        self.assertEqual(command.target_best_progress[0], 0.)
        torch.testing.assert_close(command.metrics['target_progress_m'][0], torch.tensor(.6))
        self.step_command(env, command, 1.2)
        torch.testing.assert_close(command.metrics['target_progress_m'][0], torch.tensor(.6))

    def test_partial_reset_clears_only_selected_and_ignores_teleport(self):
        env, command = self.make_active_command()
        for x in [.2, .6, 1.1]:
            self.step_command(env, command, x)
        before_metrics = {key: value[0].clone() for key, value in command.metrics.items()}
        before_initial = command.target_initial_distance[0].clone()
        before_best = command.target_best_progress[0].clone()
        # Simulate IsaacLab order: scene/event reset first, command.reset later.
        env.scene['robot'].data.root_pos_w[1] = torch.tensor([42., 104., 0.])
        env.scene.terrain.terrain_levels[1] = 5
        command.reset(slice(1, 2))
        env.episode_length_buf[1] = 0
        for key, value in command.metrics.items():
            torch.testing.assert_close(value[0], before_metrics[key])
        torch.testing.assert_close(command.target_initial_distance[0], before_initial)
        torch.testing.assert_close(command.target_best_progress[0], before_best)
        command._update_metrics()
        self.assertEqual(command.metrics['target_progress_m'][1], 0.)
        self.assertEqual(command.metrics['valid_targets_reached'][1], 0.)
        self.assertEqual(command.metrics['moving_opportunity_s'][1], 0.)
        self.assertEqual(command.target_initial_distance[1], 0.)

    def test_nonfinite_state_does_not_create_progress_or_poison_new_metrics(self):
        env, command = self.make_active_command()
        self.step_command(env, command, float('nan'))
        for name in ('target_progress_m', 'valid_targets_reached', 'moving_opportunity_s',
                     'moving_tracking_exp_vel_xy', 'moving_tracking_exp_vel_yaw'):
            self.assertTrue(torch.isfinite(command.metrics[name]).all())
            self.assertEqual(command.metrics[name][0], 0.)

    def test_progress_curriculum_requires_tracking_progress_and_true_timeout(self):
        # behind/idle, good progress, valid arrival, born-near, standing-only,
        # late fall, simultaneous fall+timeout, poor yaw, early fall, no timeout,
        # and high tracking scores with no physical progress.
        captured = []
        count = 11
        metrics = {
            'moving_tracking_exp_vel_xy': torch.tensor([1., .8, .8, 1., 0., .9, .9, 1., .1, .9, 1.]),
            'moving_tracking_exp_vel_yaw': torch.tensor([.0183, .8, .8, 1., 0., .9, .9, .1, .1, .9, 1.]),
            'moving_opportunity_s': torch.tensor([20., 5., 5., 0., 0., 15., 20., 10., .2, 10., 20.]),
            'target_progress_m': torch.tensor([0., 1.2, .3, 0., 0., 1.5, 1.5, .1, .1, 2., 0.]),
            'valid_targets_reached': torch.tensor([0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.]),
        }
        terrain = NS(terrain_levels=torch.zeros(count),
                     update_env_origins=lambda ids, up, down: captured.append((ids, up, down)))
        command = NS(metrics=metrics, _ids=lambda ids: torch.arange(count)[ids] if isinstance(ids, slice) else ids)
        env = NS(scene=NS(terrain=terrain), command_manager=NS(get_term=lambda _: command),
                 termination_manager=NS(
                     terminated=torch.tensor([False]*5 + [True, True, False, True, False, False]),
                     time_outs=torch.tensor([True]*5 + [False, True, True, False, False, True])))
        target_progress_curriculum(env, slice(None))
        self.assertEqual(captured[0][1].tolist(), [False, True, True, False, False, False, False, False, False, False, False])
        self.assertEqual(captured[0][2].tolist(), [True, False, False, False, False, True, True, True, True, False, False])
        # Even excellent tracking and net motion need >= 2 s opportunity.
        metrics['moving_opportunity_s'][1] = 1.
        target_progress_curriculum(env, torch.tensor([1]))
        self.assertFalse(captured[-1][1].item())


if __name__ == '__main__':unittest.main()
