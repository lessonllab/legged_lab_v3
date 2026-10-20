"""Projection, pure teacher, gradient separation and native PPO resume checks."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import yaml
import pytest

from test_support_transition_stage1 import inputs,ROOT,fake_native
from beamdojo_support_transition_events import NativeEventCollector
from beamdojo_support_transition_v1 import EventReplay
from beamdojo_support_transition_v2 import attach_auxiliary
from beamdojo_support_transition_diagnostics import intervene
from beamdojo_gait_retention import frozen_actor_copy,attach_algorithm,GaitRetentionConfig
from legged_lab.rsl_rl.beamdojo import DualCriticPPO

PARENT=Path(os.environ.get('SUPPORT_TRANSITION_15700',str(ROOT/'logs/rsl_rl/g1_beamdojo_fullbody_sim6/2026-10-16_14-20-57-510728_support_transition14700_candidate1000/model_15700.pt')))
TEACHER=Path(os.environ.get('SUPPORT_TRANSITION_11700',str(ROOT/'logs/rsl_rl/g1_beamdojo_fullbody_sim6/2026-10-11_11-43-11-667266_square_flat_rehearsal_3000_4096/model_11700.pt')))
if not PARENT.is_file() or not TEACHER.is_file():
    pytest.skip('Set SUPPORT_TRANSITION_15700 and SUPPORT_TRANSITION_11700 to archived checkpoints',allow_module_level=True)
CONTRACT=json.loads((PARENT.parent/'contract.json').read_text())


def algorithm(obs,version=2):
    cfg=yaml.safe_load((PARENT.parent/'params/agent.yaml').read_text())
    cfg['num_steps_per_env']=4
    cfg['algorithm']['num_learning_epochs']=1;cfg['algorithm']['num_mini_batches']=2
    cfg['actor']['class_name']=f'beamdojo_support_transition_v{version}:SupportTransitionActor'
    cfg['actor']['transition_joint_names']=CONTRACT['joint_names']
    alg=DualCriticPPO.construct_algorithm(obs,NS(num_envs=len(obs),num_actions=29),cfg,'cpu')
    saved=torch.load(PARENT,map_location='cpu',weights_only=False)
    alg.load(saved,dict(actor=True,critic=True),strict=version==1)
    alg.optimizer.load_state_dict(saved['optimizer_state_dict'])
    return alg


def test_initial_v2_matches_frozen_joint_projection_and_preserves_base_arm_motion():
    obs=inputs();old=algorithm(obs,1);new=algorithm(obs)
    hook=intervene(old._raw_actor,CONTRACT['joint_names'],'no_arms')
    with torch.no_grad():
        torch.testing.assert_close(old.actor(obs),new.actor(obs),rtol=0,atol=0)
        delta=new.actor.last_transition_delta
        protected=~new.actor.transition_control_mask.bool()
        assert not delta[:,protected].any()
        raw=torch.cat([obs[k] for k in new.actor.obs_groups],-1)
        base=new.actor.distribution.deterministic_output(new.actor.mlp(new.actor.obs_normalizer(raw)))
        torch.testing.assert_close(new.actor(obs)[:,protected],base[:,protected],rtol=0,atol=0)
        # Protected actions are state-dependent, not held to a fixed posture.
        assert base[:,protected].std(0).max()>.001
    hook.remove()


def test_predictor_receives_supervision_but_not_policy_gradients():
    obs=inputs();alg=algorithm(obs);actor=alg._raw_actor
    actor(obs).square().mean().backward()
    assert all(p.grad is None for p in actor.transition_encoder.parameters())
    assert all(p.grad is None for p in actor.transition_region.parameters())
    assert all(p.grad is None for p in actor.transition_handoff.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in actor.transition_residual.parameters())
    alg.optimizer.zero_grad(set_to_none=True)
    replay=EventReplay('cpu',capacity=1024)
    raw=torch.cat([obs[k] for k in actor.obs_groups],-1).repeat(32,1)
    replay.add(raw,torch.arange(512)%2,torch.arange(512)%225,
               torch.zeros(512,7),torch.ones(512,dtype=torch.bool))
    loss,_=replay.loss(actor);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.transition_encoder.parameters())
    assert all(p.grad is None for p in actor.transition_residual.parameters())


def test_resumed11700_teacher_excludes_inherited15700_residual():
    obs=inputs();alg=algorithm(obs);teacher=frozen_actor_copy(alg._raw_actor)
    saved=torch.load(TEACHER,map_location='cpu',weights_only=False)
    teacher.load_state_dict(saved['actor_state_dict'],strict=False)
    with torch.no_grad():
        contaminated=teacher(obs)
        assert teacher.last_transition_delta.abs().max()>0
        teacher.diagnostic_disable_residual=True
        clean=teacher(obs)
        expected=teacher.distribution.deterministic_output(teacher.mlp(teacher.get_latent(obs)))
        torch.testing.assert_close(clean,expected,rtol=0,atol=0)
        assert (contaminated-clean).abs().max()>0


def test_native_ppo_resumes_all_adam_states_and_updates_corrective_branch():
    obs=inputs();alg=algorithm(obs);actor=alg._raw_actor
    assert len(alg.optimizer.state)>0
    teacher=frozen_actor_copy(actor)
    teacher.load_state_dict(torch.load(TEACHER,map_location='cpu',weights_only=False)['actor_state_dict'],strict=False)
    teacher.diagnostic_disable_residual=True
    attach_algorithm(alg,joint_names=CONTRACT['joint_names'],action_scales=torch.full((29,),.8),
        teacher_actor=teacher,teacher_sha256='a'*64,config=GaitRetentionConfig(coefficient=.1))
    replay=EventReplay('cpu',capacity=1024)
    raw=torch.cat([obs[k] for k in actor.obs_groups],-1).repeat(32,1)
    replay.add(raw,torch.arange(512)%2,torch.arange(512)%225,torch.zeros(512,7),torch.ones(512,dtype=torch.bool))
    attach_auxiliary(alg,replay)
    before={k:v.detach().clone() for k,v in actor.state_dict().items()}
    for _ in range(4):
        with torch.inference_mode():
            alg.act(obs)
            dense=torch.ones(16)*.01;sparse=torch.linspace(-.01,0.,16)
            alg.process_env_step(obs,dense+sparse,torch.zeros(16,dtype=torch.bool),
                dict(rewards_dense=dense,rewards_sparse=sparse))
    alg.compute_returns(obs);metrics=alg.update()
    assert metrics['transition_bias_penalty']>=0
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    after=actor.state_dict()
    for k in before:
        if k.startswith(('mlp.','obs_normalizer.')):assert torch.equal(before[k],after[k])
    assert not torch.equal(before['transition_region.weight'],after['transition_region.weight'])
    assert not torch.equal(before['transition_residual.4.weight'],after['transition_residual.4.weight'])
    restored=algorithm(obs);restored.load(alg.save(),None,True)
    with torch.no_grad():torch.testing.assert_close(restored.actor(obs),alg.actor(obs),rtol=0,atol=0)


def test_collector_reset_after_native_inference_probe_preserves_normal_buffers():
    native,obs=fake_native();collector=NativeEventCollector(native)
    with torch.inference_mode():
        for _ in range(8):
            collector.record_prefix(obs);collector.post_physics()
    assert not collector.air.is_inference()
    assert not collector.touch_tick.is_inference()
    collector.reset_ids(torch.tensor([0,1]))
    assert (collector.episode==1).all() and not collector.air.any()
