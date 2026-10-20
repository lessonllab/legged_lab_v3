"""Correct stage-one action interference while keeping the frozen14700 gait.

The original actor still controls every joint. Only the additional terrain
correction protects the arms. Waist/hip yaw authority is retained because
frozen interventions showed removing it doubles yaw error.
"""
import torch
from beamdojo_support_transition_v1 import SupportTransitionActor as PreviousActor
from beamdojo_support_transition_v1 import declaration as previous_declaration

VERSION=2
ACTOR_CLASS='beamdojo_support_transition_v2:SupportTransitionActor'


def declaration():
    result=previous_declaration()
    result.update(version=VERSION,
        stage='corrective continuation of15700; separated supervised prediction and locomotion correction',
        actor_base='same14700MLP/normalizer frozen; original dynamic full-body actions preserved',
        residual_joints='legs and waist; no additional arm targets; waist/hip yaw authority retained',
        predictor_gradients='supervision only; actor loss does not update predictor or its features',
        initialization='load all15700network parameters; explicitly remove direct arm residual',
        auxiliary=dict(region_ce=.02,handoff_huber=.02,residual_l2=.005,
                       residual_dc_l2=.02,batch=1024,capacity=65536),
        task_reward_changes=[],pose_or_arm_reward_changes=[],
        causal_scope='joint projection, gradient separation and soft batch bias regularization bundled; no isolated mechanism claim')
    return result


class SupportTransitionActor(PreviousActor):
    def __init__(self,*args,transition_joint_names=None,**kwargs):
        super().__init__(*args,transition_joint_names=transition_joint_names,**kwargs)
        keep=[not any(part in name for part in ('shoulder_','elbow_','wrist_'))
              for name in transition_joint_names]
        if sum(keep)!=15:raise ValueError('Expected twelve leg and three waist joints')
        self.register_buffer('transition_control_mask',torch.tensor(keep,dtype=torch.float32))

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        if masks is not None or hidden_state is not None:
            raise ValueError('Feedforward unpadded support-transition actor only')
        raw=torch.cat([obs[k] for k in self.obs_groups],-1)
        # Predictor changes only through the observed-event supervised loss.
        # It cannot sacrifice calibration to serve as an unconstrained actor.
        with torch.no_grad():
            base=self.mlp(self.obs_normalizer(raw))
            latent,regions,handoff=self.prediction(raw)
            condition=torch.cat((latent,regions.softmax(-1).flatten(1),handoff.tanh().flatten(1)),-1)
        self.last_transition_region=regions.detach()
        self.last_transition_handoff=handoff.detach()
        delta=self.transition_residual(condition).tanh()*self.transition_action_caps*self.transition_control_mask
        gate=self.gate(obs)
        if self.diagnostic_disable_residual:gate=torch.zeros_like(gate)
        delta=delta*gate[:,None]
        self.last_transition_delta=delta
        self.last_transition_gate=gate
        mean=base+delta
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mean)


def attach_auxiliary(algorithm,replay):
    """Native PPO plus supervision and soft correction-bias regularization.

    The DC loss acts on additional targets across diverse gait phases. It has
    no fixed posture or fixed arm trajectory and does not penalize arm motion.
    """
    original_update=algorithm.update
    def update():
        optimizer=algorithm.optimizer;original_zero=optimizer.zero_grad
        had='zero_grad' in vars(optimizer);previous=vars(optimizer).get('zero_grad')
        metrics=[];pending=[]
        def capture(module,args,kwargs,output):
            if kwargs.get('stochastic_output',False) and torch.is_grad_enabled():
                if pending:raise RuntimeError('Unexpected extra actor forward before PPO update')
                delta=module.last_transition_delta
                selected=module.transition_control_mask.bool()
                normalized=delta[:,selected]/module.transition_action_caps[selected]
                active=module.last_transition_gate.float()
                bias=(normalized*active[:,None]).sum(0)/active.sum().clamp_min(1)
                l2=.005*normalized.square().mean()
                dc=.02*bias.square().mean()
                penalty=l2+dc
                pending.append((penalty,dict(
                    transition_delta_rms_rad=float(delta.detach().square().mean().sqrt()*.25),
                    transition_delta_max_rad=float(delta.detach().abs().max()*.25),
                    transition_active_fraction=float(active.mean()),
                    transition_residual_penalty=float(l2.detach()),
                    transition_bias_penalty=float(dc.detach()),
                    transition_bias_normalized_rms=float(bias.detach().square().mean().sqrt()))))
        def zero(*args,**kwargs):
            result=original_zero(*args,**kwargs)
            if len(pending)!=1:raise RuntimeError('Expected one correction minibatch')
            penalty,entry=pending.pop();penalty.backward(retain_graph=True)
            aux=replay.loss(algorithm._raw_actor)
            if aux is not None:
                loss,extra=aux;loss.backward();entry.update(extra)
            metrics.append(entry)
            return result
        hook=algorithm.actor.register_forward_hook(capture,with_kwargs=True)
        optimizer.zero_grad=zero
        try:result=original_update()
        finally:
            hook.remove()
            if had:optimizer.zero_grad=previous
            else:del optimizer.zero_grad
        if pending or len(metrics)!=algorithm.num_learning_epochs*algorithm.num_mini_batches:
            raise RuntimeError('Auxiliary PPO minibatch count mismatch')
        result.update({key:sum(m.get(key,0.) for m in metrics)/len(metrics) for key in metrics[0]})
        result['transition_label_count']=float(replay.total)
        return result
    algorithm.update=update
