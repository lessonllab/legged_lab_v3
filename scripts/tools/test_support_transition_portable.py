"""CPU regression tests requiring neither checkpoints nor an Isaac scene."""
import pytest
import torch
from tensordict import TensorDict

from beamdojo_support_transition_events import landing_cell
from beamdojo_support_transition_loader import configure_actor
from beamdojo_support_transition_v1 import EventReplay
from beamdojo_support_transition_v2 import SupportTransitionActor


JOINT_NAMES = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
    'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint',
    'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint',
    'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint',
    'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint',
    'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
    'left_wrist_yaw_joint', 'right_wrist_yaw_joint',
]


def inputs(n=8):
    generator = torch.Generator().manual_seed(42)
    obs = TensorDict(dict(policy=torch.randn(n, 480, generator=generator)*.1,
                          terrain=torch.ones(n, 225),
                          course_tracking=torch.zeros(n, 3)), batch_size=[n])
    obs['policy'][:, 42:45] = torch.tensor([.75, 0., 0.])
    obs['terrain'].reshape(n, 15, 15)[:, 9, 7] = 5.
    return obs


def actor(obs):
    return SupportTransitionActor(
        obs, {'actor': ['policy', 'terrain', 'course_tracking']}, 'actor', 29,
        hidden_dims=[32, 32], obs_normalization=False,
        distribution_cfg={'class_name': 'GaussianDistribution',
                          'init_std': 1., 'std_type': 'scalar'},
        transition_joint_names=JOINT_NAMES)


def test_zero_residual_preserves_base_and_dynamic_arms():
    obs=inputs();model=actor(obs)
    raw=torch.cat([obs[k] for k in model.obs_groups],-1)
    with torch.no_grad():
        expected=model.distribution.deterministic_output(model.mlp(raw))
        torch.testing.assert_close(model(obs),expected,rtol=0,atol=0)
        arms=~model.transition_control_mask.bool()
        assert expected[:,arms].std(0).max()>.001


def test_gate_excludes_flat_reverse_turn_stand_and_unsupported_speed():
    obs=inputs(6)
    obs['terrain'][1]=1.
    obs['policy'][2,42:45]=torch.tensor([-.5,0.,0.])
    obs['policy'][3,42:45]=torch.tensor([0.,0.,.2])
    obs['policy'][4,42:45]=0.
    obs['policy'][5,42:45]=torch.tensor([1.5,0.,0.])
    assert SupportTransitionActor.gate(obs).tolist()==[True,False,False,False,False,False]


def test_joint_projection_retains_waist_and_bounds_added_motion():
    obs=inputs();model=actor(obs)
    with torch.no_grad():
        model.transition_residual[-1].bias.fill_(100.)
        model(obs)
        dq=model.last_transition_delta*.25
    arms=~model.transition_control_mask.bool()
    assert int(model.transition_control_mask.sum())==15
    assert not dq[:,arms].any()
    for index,name in enumerate(JOINT_NAMES):
        if 'waist_' in name:
            torch.testing.assert_close(dq[:,index],torch.full((len(obs),),.04))
        elif not arms[index]:
            torch.testing.assert_close(dq[:,index],torch.full((len(obs),),.08))


def test_policy_and_event_supervision_have_separate_gradient_paths():
    obs=inputs();model=actor(obs)
    model(obs).square().mean().backward()
    predictor=list(model.transition_encoder.parameters())+list(model.transition_region.parameters())+list(model.transition_handoff.parameters())
    assert all(p.grad is None for p in predictor)
    assert all(p.grad is None for p in model.mlp.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.transition_residual.parameters())
    model.zero_grad(set_to_none=True)
    raw=torch.cat([obs[k] for k in model.obs_groups],-1).repeat(32,1)
    replay=EventReplay('cpu',capacity=256)
    replay.add(raw,torch.arange(256)%2,torch.arange(256)%225,
               torch.zeros(256,7),torch.ones(256,dtype=torch.bool))
    loss,_=replay.loss(model,batch=64);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in predictor)
    assert all(p.grad is None for p in model.transition_residual.parameters())


def test_touchdown_cells_rotate_with_the_observation_anchor():
    root=torch.tensor([[10.,20.,0.]])
    cell,visible=landing_cell(torch.tensor([[10.,20.2,0.]]),root,torch.tensor([torch.pi/2]))
    assert visible.item() and cell.item()==9*15+7
    _,visible=landing_cell(torch.tensor([[10.,21.,0.]]),root,torch.tensor([torch.pi/2]))
    assert not visible.item()


def test_actor_loader_rejects_changed_archived_dependency(tmp_path):
    archive=tmp_path/'params';archive.mkdir()
    (archive/'beamdojo_support_transition_v1.py').write_text('# changed archived source\n')
    settings={'actor':{}}
    contract={'support_transition':{'version':2,'base_actor_source_sha256':'0'*64}}
    with pytest.raises(ValueError,match='base actor source digest mismatch'):
        configure_actor(settings,contract,tmp_path/'model_16700.pt')
    assert settings=={'actor':{}}
