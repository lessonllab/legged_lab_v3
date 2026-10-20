"""Critical stage-one contracts: zero identity, label causality, PPO gradients."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from tensordict import TensorDict
import yaml
import pytest

from beamdojo_support_transition_v1 import SupportTransitionActor,EventReplay,attach_auxiliary,migrate_optimizer
from beamdojo_support_transition_events import NativeEventCollector,landing_cell
from beamdojo_gait_retention import frozen_actor_copy,attach_algorithm,GaitRetentionConfig
from legged_lab.rsl_rl.beamdojo import DualCriticPPO

ROOT=Path(__file__).resolve().parents[2]
_status=ROOT/'.runtime/gait_preservation_11700_training.json'
_source=os.environ.get('SUPPORT_TRANSITION_14700')
if _source is None and _status.is_file():
    _source=json.loads(_status.read_text())['last_checkpoint']
if _source is None or not Path(_source).is_file():
    pytest.skip('Set SUPPORT_TRANSITION_14700 to the archived14700 checkpoint for integration tests',allow_module_level=True)
SOURCE=Path(_source)
PARENT=json.loads((SOURCE.parent/'contract.json').read_text())


def inputs(n=16):
    obs=TensorDict(dict(policy=torch.randn(n,480)*.1,critic=torch.randn(n,495)*.1,
                        terrain=torch.ones(n,225),course_tracking=torch.zeros(n,3)),batch_size=[n])
    obs['policy'][:,42:45]=torch.tensor([.75,0.,0.])
    obs['terrain'].reshape(n,15,15)[:,10:12,5:10]=5.
    return obs


def algorithm(obs,candidate=True):
    cfg=yaml.safe_load((SOURCE.parent/'params/agent.yaml').read_text())
    cfg['num_steps_per_env']=4;cfg['algorithm']['num_learning_epochs']=1;cfg['algorithm']['num_mini_batches']=2
    if candidate:
        cfg['actor']['class_name']='beamdojo_support_transition_v1:SupportTransitionActor'
        cfg['actor']['transition_joint_names']=PARENT['joint_names']
    alg=DualCriticPPO.construct_algorithm(obs,NS(num_envs=len(obs),num_actions=29),cfg,'cpu')
    saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
    alg.load(saved,dict(actor=True,critic=True),strict=not candidate)
    if candidate:migrate_optimizer(alg,saved,alg._raw_actor)
    return alg,saved


def test_zero_residual_reproduces_source_with_native_tails():
    obs=inputs();child,_=algorithm(obs);parent,_=algorithm(obs,False)
    with torch.no_grad():
        assert torch.equal(parent.actor(obs),child.actor(obs))
        assert torch.count_nonzero(child.actor.last_transition_delta)==0
    assert not any(p.requires_grad for p in child.actor.mlp.parameters())


def test_action_bounds_and_flat_reverse_turn_stop_exclusion():
    obs=inputs(5);alg,_=algorithm(obs)
    obs['terrain'][1]=1.
    obs['policy'][2,42:45]=torch.tensor([-.5,0.,0.])
    obs['policy'][3,42:45]=torch.tensor([0.,0.,.2])
    obs['policy'][4,42:45]=0.
    with torch.no_grad():
        alg.actor.transition_residual[-1].bias.fill_(100)
        alg.actor(obs)
        dq=alg.actor.last_transition_delta*.25
        assert torch.count_nonzero(dq[1:])==0
        assert dq[0].abs().max()<=.080001
        for i,name in enumerate(PARENT['joint_names']):
            if 'shoulder' in name:assert dq[0,i].abs()<=.030001


def test_real_native_ppo_plus_existing_retention_and_auxiliary_backward():
    obs=inputs();alg,saved=algorithm(obs)
    replay=EventReplay('cpu',capacity=1024)
    replay.add(torch.cat([obs[k] for k in ('policy','terrain','course_tracking')],-1).repeat(32,1),
               torch.arange(512)%2,torch.arange(512)%225,torch.zeros(512,7),torch.ones(512,dtype=torch.bool))
    teacher=frozen_actor_copy(alg._raw_actor)
    attach_algorithm(alg,joint_names=PARENT['joint_names'],action_scales=torch.full((29,),.8),
                     teacher_actor=teacher,teacher_sha256='a'*64,config=GaitRetentionConfig(coefficient=.1))
    attach_auxiliary(alg,replay)
    before={k:v.detach().clone() for k,v in alg._raw_actor.state_dict().items()}
    for _ in range(4):
        with torch.inference_mode():
            alg.act(obs)
            dense=torch.ones(16)*.01;sparse=torch.linspace(-.01,0.,16)
            alg.process_env_step(obs,dense+sparse,torch.zeros(16,dtype=torch.bool),
                                 dict(rewards_dense=dense,rewards_sparse=sparse))
    alg.compute_returns(obs)
    result=alg.update()
    assert result['transition_region_ce']>0 and np.isfinite(list(result.values())).all()
    after=alg._raw_actor.state_dict()
    for k in before:
        if k.startswith('mlp.'):assert torch.equal(before[k],after[k])
    assert not torch.equal(before['transition_region.weight'],after['transition_region.weight'])
    assert not torch.equal(before['transition_residual.4.weight'],after['transition_residual.4.weight'])
    assert all(p.grad is None for p in teacher.parameters())
    restored,_=algorithm(obs)
    restored.load(alg.save(),None,True)
    with torch.no_grad():torch.testing.assert_close(restored.actor(obs),alg.actor(obs),rtol=0,atol=0)


def test_local_map_cell_uses_anchor_heading_not_world_axes():
    root=torch.tensor([[10.,20.,0.]])
    cell,valid=landing_cell(torch.tensor([[10.,20.2,0.]]),root,torch.tensor([torch.pi/2]))
    assert valid.item() and cell.item()==9*15+7
    _,outside=landing_cell(torch.tensor([[10.,21.,0.]]),root,torch.tensor([torch.pi/2]))
    assert not outside.item()


def fake_native(n=2):
    d=NS(root_pos_w=torch.zeros(n,3),root_quat_w=torch.tensor([[0.,0.,0.,1.]]).repeat(n,1),
         root_lin_vel_b=torch.zeros(n,3),projected_gravity_b=torch.tensor([[0.,0.,-1.]]).repeat(n,1),
         root_ang_vel_b=torch.zeros(n,3),body_pos_w=torch.zeros(n,2,3))
    robot=NS(data=d,find_bodies=lambda *a,**kw:([0,1],['left','right']))
    sensor=NS(data=NS(net_forces_w=torch.zeros(n,2,3)),find_bodies=robot.find_bodies)
    class Scene(dict):pass
    scene=Scene(robot=robot);scene.sensors={'contact_forces':sensor}
    env=NS(scene=scene,step_dt=.02,reset_buf=torch.zeros(n,dtype=torch.bool))
    obs=inputs(n);obs['policy'][0,0]=10.;obs['policy'][1,0]=20.
    return NS(num_envs=n,device='cpu',unwrapped=env,get_observations=lambda:obs),obs


def test_delayed_contact_labels_never_cross_reset_or_label_initial_support():
    native,obs=fake_native();c=NativeEventCollector(native)
    force=c.sensor.data.net_forces_w
    for t in range(16):
        c.record_prefix(obs)
        force[:]=0
        force[:,1,2]=100
        if t>=7:force[:,0,2]=100
        c.post_physics()
        if t==9:c.reset_ids(torch.tensor([1]))
    assert c.replay.size>0
    assert (c.replay.raw[:c.replay.size,0]==10.).all()
    assert (c.replay.foot[:c.replay.size]==0).all()
    assert (c.replay.target[:c.replay.size,0]>0).all()
    assert c.incomplete>=1
    assert c.event_total==2


def test_offline_forecast_audit_aligns_prephysics_prefix_and_delayed_outcome():
    from audit_support_transition_stage1 import analyze
    t=32;n=1
    trace=dict(time=(np.arange(1,t+1)*.02)[:,None],active=np.ones((t,n),bool),
        contact=np.ones((t,n,2),bool),contact_fz=np.ones((t,n,2))*100.,
        full_foot_support=np.ones((t,n,2),bool),root=np.zeros((t,n,3)),
        joint_names=np.array(['left_knee_joint','right_knee_joint']),joint_pos=np.zeros((t,n,2)),
        body_velocity=np.zeros((t,n,3)),gravity=np.zeros((t,n,3)),body_omega=np.zeros((t,n,3)),
        link_local=np.zeros((t,n,2,3)),transition_prefix_root=np.zeros((t,n,3)),
        transition_prefix_quat=np.zeros((t,n,4)),transition_handoff=np.zeros((t,n,2,7)),
        transition_region_cell=np.zeros((t,n,2),int),transition_active=np.ones((t,n),bool))
    trace['root'][:,:,0]=2.;trace['body_velocity'][:,:,0]=.7;trace['gravity'][:,:,2]=-1.
    trace['transition_prefix_quat'][:,:,3]=1.
    trace['link_local'][:,:,0,:2]=[.2,.1]
    trace['contact'][2:8,0,0]=False
    # Contact observed at post-physics tick9, future state at tick15. Prefix
    # tick5 is stored with post-physics trace index5, not post-physics index4.
    for lag in (4,8):
        anchor=9-lag
        trace['transition_handoff'][anchor,0,0]=[lag*.02/.5,.7,0,0,0,0,.5]
        trace['transition_region_cell'][anchor,0,0]=9*15+8
    rows=[dict(case_id=0,kind='stone',mode='course',family=0,level=10,variant=7)]
    result=analyze(trace,rows,['left_ankle_roll_link','right_ankle_roll_link'])
    assert result['prediction_label_count']==2
    assert result['prediction']['all']['cell_correct']==1.
    assert result['prediction']['all']['time_absolute_error_s']==0.
    assert result['prediction']['all']['handoff_velocity_error_mps']==0.
    trace['active'][12:]=False
    assert analyze(trace,rows,['left_ankle_roll_link','right_ankle_roll_link'])['prediction_label_count']==0
