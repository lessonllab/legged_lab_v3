import unittest
from types import SimpleNamespace
import torch
from isaaclab.managers import ObservationManager
from legged_lab.tasks.locomotion.amp.mdp.style_state import encode_style, mirror_style, joint_mirror_map
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_style_env_cfg import (
    StyleAgentObs, StyleReferenceObs, G1AmpDepthStyleEnvCfg)
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_depth_instinct_env_cfg import G1AmpDepthInstinctEnvCfg


class TestStyleState(unittest.TestCase):
    def test_body_frame(self):
        quat = torch.tensor([[0., 0., 2**-.5, 2**-.5]])
        q = torch.ones(1, 29)
        state = encode_style(quat, q, q*2, torch.tensor([[1., 0., 0.]]), torch.tensor([[0., 0., 2.]]), q*.5)
        torch.testing.assert_close(state[:, :3], torch.tensor([[0., 0., -1.]]))
        torch.testing.assert_close(state[:, 3:32], q*.5)
        torch.testing.assert_close(state[:, 32:61], q*.1)
        torch.testing.assert_close(state[:, 61:64], torch.tensor([[0., -1., 0.]]), atol=1e-6, rtol=1e-6)
        self.assertEqual(state.shape, (1, 67))

    def test_physical_reflection_and_involution(self):
        names = ['left_hip_roll_joint', 'right_hip_roll_joint', 'waist_pitch_joint',
                 'left_knee_joint', 'right_knee_joint', 'waist_yaw_joint']
        indices, signs = joint_mirror_map(names, 'cpu')
        quat = torch.randn(3, 10, 4); quat /= quat.norm(dim=-1, keepdim=True)
        q, qd = torch.randn(3, 10, 6), torch.randn(3, 10, 6)
        lin, ang = torch.randn(3, 10, 3), torch.randn(3, 10, 3)
        state = encode_style(quat, q, qd, lin, ang, torch.zeros_like(q))
        expected = encode_style(quat * torch.tensor([-1., 1., -1., 1.]),
            q[..., indices]*signs, qd[..., indices]*signs,
            lin*torch.tensor([1., -1., 1.]), ang*torch.tensor([-1., 1., -1.]), torch.zeros_like(q))
        torch.testing.assert_close(mirror_style(state, indices, signs), expected)
        torch.testing.assert_close(mirror_style(mirror_style(state, indices, signs), indices, signs), state)

    def test_live_and_reference_encoding_history_and_reset(self):
        names = [f'{side}_{joint}_joint' for joint in ('hip_pitch', 'hip_roll', 'knee') for side in ('left', 'right')]
        pos = torch.zeros(2, len(names))
        data = SimpleNamespace(root_quat_w=torch.tensor([[0.,0.,0.,1.]]).repeat(2,1),
            joint_pos=pos, joint_vel=pos.clone(), default_joint_pos=pos.clone(),
            root_link_lin_vel_w=torch.zeros(2,3), root_ang_vel_w=torch.zeros(2,3))
        ref = SimpleNamespace(get_root_quat=lambda:data.root_quat_w[:,None].expand(-1,10,-1),
            get_dof_pos=lambda:seq, get_dof_vel=lambda:data.joint_vel[:,None].expand(-1,10,-1),
            get_root_vel_w=lambda:data.root_link_lin_vel_w[:,None].expand(-1,10,-1),
            get_root_ang_vel_w=lambda:data.root_ang_vel_w[:,None].expand(-1,10,-1))
        seq = torch.zeros(2,10,len(names))
        env = SimpleNamespace(num_envs=2,device='cpu',sim=SimpleNamespace(is_playing=lambda:True),
            scene={'robot':SimpleNamespace(data=data,joint_names=names)},
            animation_manager=SimpleNamespace(get_term=lambda _:ref))
        obs = ObservationManager({'disc':StyleAgentObs(),'disc_demo':StyleReferenceObs()},env)
        term = obs._group_obs_term_cfgs['disc_demo'][0].func
        term.mirrored[:] = False
        for t in range(10):
            data.joint_pos.fill_(float(t))
            seq = torch.arange(max(t-9,0),t+1,dtype=torch.float32)
            seq = torch.cat([seq[:1].repeat(10-len(seq)), seq])[None,:,None].expand(2,-1,len(names))
            result = obs.compute(update_history=True)
            torch.testing.assert_close(result['disc'],result['disc_demo'])
        torch.testing.assert_close(result['disc'][0,:,3],torch.arange(10,dtype=torch.float32))
        term.mirrored[:] = torch.tensor([True,False])
        term.reset(torch.tensor([0]))
        self.assertFalse(term.mirrored[1])
        obs.reset(torch.tensor([0])); result=obs.compute(update_history=True)
        torch.testing.assert_close(result['disc'][0],result['disc'][0,-1:].expand_as(result['disc'][0]))

    def test_v2_unchanged_and_v3_compatible_policy_inputs(self):
        old, new = G1AmpDepthInstinctEnvCfg(), G1AmpDepthStyleEnvCfg()
        old.validate();new.validate()
        self.assertEqual(old.observations.disc.history_length,4)
        self.assertEqual(new.observations.disc.history_length,10)
        self.assertEqual(new.animation.animation.num_steps_to_use,10)
        for name in ('policy','critic','depth'):
            self.assertEqual(getattr(old.observations,name).to_dict(),getattr(new.observations,name).to_dict())
        self.assertEqual(old.rewards.to_dict(),new.rewards.to_dict())


if __name__ == '__main__':
    unittest.main()
