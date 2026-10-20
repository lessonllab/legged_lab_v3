"""Stage-one contact-prediction-conditioned residual, above frozen model14700.

This is the interface experiment, not a multi-step viability planner. Actor
inputs remain policy480 + terrain225 + course_tracking3. Native contact/state
truth is used only to create delayed auxiliary labels. Predicted location is a
distribution over DISCRETE map cells, never a barycentre used as a foot target.
The original locomotion/terrain rewards and native Gaussian likelihood remain.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.models.mlp_model import MLPModel

VERSION = 1
ACTOR_CLASS = 'beamdojo_support_transition_v1:SupportTransitionActor'


def declaration():
    return dict(version=VERSION, stage='first interface experiment; no multi-step selector yet',
        input='unchanged actor policy480/terrain225/course_tracking3; no new privileged actor input',
        representation='two discrete15x15 next-touchdown distributions and per-foot time/handoff predictions',
        handoff_delay_s=.12, prediction_lookback_ticks=[4,8,12,16,20,24],
        target_fields=['remaining_touchdown_time/0.5','handoff_body_vx','handoff_body_vy',
                       'handoff_gravity_x/0.3','handoff_gravity_y/0.3',
                       'handoff_body_omega_z','handoff_positive_vertical_load_share'],
        contact_event='first Fz>5N after >=3 consecutive noncontact frames; initial/reset support excluded',
        rollout_label_scope='same episode, first subsequent contact of that foot;120ms handoff observed before reset',
        missing_outcomes='unobserved/reset-before-handoff are not labelled success; this is regression, not viability certification',
        actor_base='14700 MLP and normalization frozen; new bounded29D residual; original std trainable',
        deterministic_initialization='exact source14700 action mean at zero residual',
        delta_rad=dict(leg=.08,waist=.04,arm=.03),action_scale=.25,
        active='pure forward0.3..1.0m/s and an observed gap in the forward heightmap; zero elsewhere',
        task_reward_changes=[], pose_or_arm_reward_changes=[],
        auxiliary=dict(region_ce=.02,handoff_huber=.02,residual_l2=.005,batch=1024,capacity=65536),
        gating_scope='flat/stand/backward/turn deterministic actor remains source14700; closed-loop outcomes still evaluated',
        comparison='ordinary1000 is an engineering equal-update control; freezing/architecture/auxiliary differ, not a mechanism-isolated ablation')


class SupportTransitionActor(MLPModel):
    def __init__(self, *args, transition_joint_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.obs_dim != 708 or self.obs_groups != ['policy','terrain','course_tracking']:
            raise ValueError('Expected the model14700 native708-input actor')
        if transition_joint_names is None or len(transition_joint_names) != 29:
            raise ValueError('Provide actual native29 joint order')
        self.mlp.requires_grad_(False)
        self.obs_normalizer.requires_grad_(False)
        self.transition_encoder = nn.Sequential(nn.Linear(708,256),nn.ELU(),nn.Linear(256,128),nn.ELU())
        self.transition_region = nn.Linear(128,450)
        self.transition_handoff = nn.Linear(128,14)
        self.transition_residual = nn.Sequential(nn.Linear(128+450+14,128),nn.ELU(),
                                               nn.Linear(128,64),nn.ELU(),nn.Linear(64,29))
        nn.init.zeros_(self.transition_residual[-1].weight)
        nn.init.zeros_(self.transition_residual[-1].bias)
        caps = [.04 if 'waist_' in n else (.03 if any(x in n for x in
                ('shoulder_','elbow_','wrist_')) else .08) for n in transition_joint_names]
        self.register_buffer('transition_action_caps',torch.tensor(caps)/.25)
        self.last_transition_delta = None
        self.last_transition_gate = None
        self.last_transition_region = None
        self.last_transition_handoff = None
        self.diagnostic_disable_residual = False

    def update_normalization(self, obs):
        # The inherited base must not drift through changing normalizer stats.
        pass

    def prediction(self, raw):
        latent=self.transition_encoder(self.obs_normalizer(raw))
        return latent,self.transition_region(latent).reshape(-1,2,225),self.transition_handoff(latent).reshape(-1,2,7)

    @staticmethod
    def gate(obs):
        c=obs['policy'][:,42:45]
        forward=(c[:,0]>=.3)&(c[:,0]<=1.01)&(c[:,1].abs()<.05)&(c[:,2].abs()<.05)
        # Exact existing observation only; no environment group or route ID.
        m=obs['terrain'].reshape(-1,15,15)
        gaps=(m[:,7:14,2:13]>4.9).any(dim=(-1,-2))
        return forward&gaps

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        if masks is not None or hidden_state is not None:
            raise ValueError('Feedforward unpadded support-transition actor only')
        raw=torch.cat([obs[k] for k in self.obs_groups],-1)
        with torch.no_grad(): base=self.mlp(self.obs_normalizer(raw))
        latent,regions,handoff=self.prediction(raw)
        self.last_transition_region=regions.detach()
        self.last_transition_handoff=handoff.detach()
        # Full distributions retain distinct candidates; no averaged location.
        condition=torch.cat((latent,regions.softmax(-1).flatten(1),handoff.tanh().flatten(1)),-1)
        delta=self.transition_residual(condition).tanh()*self.transition_action_caps
        gate=self.gate(obs)
        if self.diagnostic_disable_residual: gate=torch.zeros_like(gate)
        delta=delta*gate[:,None]
        self.last_transition_delta=delta
        self.last_transition_gate=gate
        mean=base+delta
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mean)

    def as_jit(self):
        raise NotImplementedError('Use the full native actor; inherited MLP export would omit the residual')

    def as_onnx(self,verbose=False):
        raise NotImplementedError('Support-transition export has not been validated')


class EventReplay:
    """Bounded training-only buffer of observed (prefix, future event) pairs."""
    def __init__(self,device,capacity=65536,seed=724):
        self.capacity=capacity;self.position=0;self.size=0;self.total=0
        self.raw=torch.empty(capacity,708,device=device,dtype=torch.float16)
        self.foot=torch.empty(capacity,device=device,dtype=torch.long)
        self.cell=torch.empty_like(self.foot)
        self.target=torch.empty(capacity,7,device=device)
        self.visible=torch.empty(capacity,device=device,dtype=torch.bool)
        self.generator=torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def add(self,raw,foot,cell,target,visible):
        n=len(raw)
        if n==0:return
        if n>self.capacity:
            raw,foot,cell,target,visible=(x[-self.capacity:] for x in (raw,foot,cell,target,visible));n=self.capacity
        idx=(torch.arange(n,device=raw.device)+self.position)%self.capacity
        self.raw[idx]=raw.to(self.raw.dtype);self.foot[idx]=foot;self.cell[idx]=cell
        self.target[idx]=target;self.visible[idx]=visible
        self.position=(self.position+n)%self.capacity;self.size=min(self.capacity,self.size+n);self.total+=n

    def loss(self,actor,batch=1024):
        if self.size<256:return None
        ids=torch.randint(self.size,(batch,),device=self.raw.device,generator=self.generator)
        _,logits,handoff=actor.prediction(self.raw[ids].float())
        rows=torch.arange(batch,device=ids.device);foot=self.foot[ids]
        chosen=logits[rows,foot];state=handoff[rows,foot]
        mask=self.visible[ids]
        ce=F.cross_entropy(chosen,self.cell[ids],reduction='none')
        ce=(ce*mask).sum()/mask.sum().clamp_min(1)
        huber=F.smooth_l1_loss(state,self.target[ids])
        loss=.02*ce+.02*huber
        with torch.no_grad():
            accuracy=((chosen.argmax(-1)==self.cell[ids])&mask).sum()/mask.sum().clamp_min(1)
            time_error=(state[:,0]-self.target[ids,0]).abs().mean()*.5
        return loss,dict(transition_region_ce=float(ce.detach()),transition_handoff_huber=float(huber.detach()),
            transition_region_cell_accuracy=float(accuracy),transition_time_mae_s=float(time_error),
            transition_visible_label_fraction=float(mask.float().mean()))


def attach_auxiliary(algorithm,replay):
    """Add auxiliary gradients inside native PPO zero/backward/clip/step order.

    Only supervised replay observations are read here; inference never consults
    replay, contacts, root world pose, or labels. Works with existing retention.
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
                penalty=.005*(delta/module.transition_action_caps).square().mean()
                pending.append((penalty,dict(transition_delta_rms_rad=float(delta.detach().square().mean().sqrt()*.25),
                    transition_delta_max_rad=float(delta.detach().abs().max()*.25),
                    transition_active_fraction=float(module.last_transition_gate.float().mean()),
                    transition_residual_penalty=float(penalty.detach()))))
        def zero(*args,**kwargs):
            result=original_zero(*args,**kwargs)
            if len(pending)!=1:raise RuntimeError('Expected one transition residual minibatch')
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
        result.update({k:sum(m.get(k,0.) for m in metrics)/len(metrics) for k in metrics[0]})
        result['transition_label_count']=float(replay.total)
        return result
    algorithm.update=update


def migrate_optimizer(algorithm,source,original_actor):
    """Copy old Adam moments by parameter name, leave new heads fresh."""
    named=[('actor.'+n,p) for n,p in original_actor.named_parameters() if not n.startswith('transition_')]
    named += [('critic.'+n,p) for n,p in algorithm._raw_critic.named_parameters()]
    old_ids=source['optimizer_state_dict']['param_groups'][0]['params']
    if len(old_ids)!=len(named):raise ValueError('Source optimizer parameter-order mismatch')
    current={**{'actor.'+n:p for n,p in algorithm._raw_actor.named_parameters()},
             **{'critic.'+n:p for n,p in algorithm._raw_critic.named_parameters()}}
    copied=0
    for old_id,(name,_) in zip(old_ids,named):
        if old_id not in source['optimizer_state_dict']['state']:continue
        parameter=current[name];state=source['optimizer_state_dict']['state'][old_id]
        for key,value in state.items():
            if isinstance(value,torch.Tensor) and key in ('exp_avg','exp_avg_sq') and value.shape!=parameter.shape:
                raise ValueError('Adam shape mismatch '+name)
        algorithm.optimizer.state[parameter]={k:(v.to(parameter.device).clone() if isinstance(v,torch.Tensor) else deepcopy(v))
                                              for k,v in state.items()}
        copied+=1
    return copied
