"""Training-only joint contact reference potential; original rewards unchanged.

Actor/critic inputs do not include the locked plan. This is bounded trajectory
potential shaping, not a guarantee of invariant approximate PPO updates or of
whole-body feasibility. Timeout handling matches this repository's PRE-state
dual-critic bootstrap, including simultaneous timeout and termination masks.
"""
import torch

WEIGHTS = (.30, .15, .20, .10, .15, .10)
GAMMA = .99


def tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def minimum_jerk(time, start, end):
    u=((time-start)/(end-start).clamp_min(1e-6)).clamp(0.,1.)
    return u.pow(3)*(10.-15.*u+6.*u.square())


def foot_reference(initial, first_target, second_target, first_foot,
                   second_enabled, phase_times, now):
    """Physical left/right XY reference with immutable absolute deadlines."""
    n=len(initial); rows=torch.arange(n,device=initial.device)
    foot=first_foot.long().clamp(0,1);other=1-foot
    target=initial[...,:2].clone()
    a=minimum_jerk(now,phase_times[:,0],phase_times[:,1])
    b=minimum_jerk(now,phase_times[:,3],phase_times[:,4])
    target[rows,foot]=initial[rows,foot,:2]+a[:,None]*(first_target[:,:2]-initial[rows,foot,:2])
    second=torch.where(second_enabled[:,None],second_target[:,:2],initial[rows,other,:2])
    target[rows,other]=initial[rows,other,:2]+b[:,None]*(second-initial[rows,other,:2])
    return target


def bounded_reference_cost(actual, plan, reference, now):
    """One bounded weighted cost; load slots explicitly [old A, new B]."""
    n=len(actual['com']);rows=torch.arange(n,device=actual['com'].device)
    first=plan['first_support_foot'].long().clamp(0,1)
    second_enabled=plan['selected_branch']==1
    footref=foot_reference(plan['initial_foot_positions'],plan['first_target'],plan['second_target'],
        first,second_enabled,plan['phase_times'],now)
    # NaN payload is sanitized only for arithmetic, never to make a valid plan.
    clean=lambda x:torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
    foot_error=((actual['foot_positions'][...,:2]-clean(footref))/.08).square().mean(-1)
    tracked=torch.ones((n,2),device=foot_error.device,dtype=torch.bool)
    # Hold-B leaves old A's subsequent unsupported swing unconstrained.
    tracked[rows,1-first]=second_enabled|(now<plan['phase_times'][:,3])
    foot_cost=torch.where(tracked,foot_error,0.).sum(-1)/tracked.sum(-1).clamp_min(1)
    foot_cost=foot_cost.clamp(0.,1.)
    order=torch.stack((1-first,first),-1)
    contact=actual['contact'].gather(1,order)
    force=actual['contact_fz'].clamp_min(0.).gather(1,order)
    shares=reference['foot_normal_load_fractions']
    contact_cost=(contact.float()-(shares>.05).float()).square().mean(-1).clamp(0.,1.)
    # Both feet off does not invent a normalized support fraction.
    total=force.sum(-1,keepdim=True)
    observed=force/total.clamp_min(1e-6)
    load_cost=(observed-shares).square().mean(-1).clamp(0.,1.)
    load_cost=torch.where(total.squeeze(-1)>5.,load_cost,torch.ones_like(load_cost))
    com_cost=((actual['com'][:,:2]-reference['com'][:,:2])/.10).square().mean(-1).clamp(0.,1.)
    velocity_cost=((actual['velocity'][:,:2]-reference['velocity'][:,:2])/.5).square().mean(-1).clamp(0.,1.)
    band=actual['mass'][:,None]*9.81*.5*.3*torch.tensor([.04,.04,.08],device=force.device)
    momentum_cost=((actual['momentum']-reference['momentum'])/band.clamp_min(1e-6)).square().mean(-1).clamp(0.,1.)
    components=torch.stack((foot_cost,contact_cost,load_cost,com_cost,velocity_cost,momentum_cost),-1)
    finite=torch.isfinite(components).all(-1)
    for key in ('com','velocity','momentum','mass','foot_positions','contact_fz'):
        finite &= torch.isfinite(actual[key]).reshape(n,-1).all(-1)
    valid=plan['plan_valid']&reference['valid']&finite&(actual['mass']>0.)
    valid &= torch.isfinite(plan['initial_foot_positions']).all((-1,-2))
    valid &= torch.isfinite(plan['first_target']).all(-1)&torch.isfinite(plan['phase_times']).all(-1)
    valid &= (~second_enabled)|torch.isfinite(plan['second_target']).all(-1)
    valid &= ((plan['first_support_foot']==0)|(plan['first_support_foot']==1))
    valid &= (plan['selected_branch']==0)|(plan['selected_branch']==1)
    cost=(clean(components)*torch.tensor(WEIGHTS,device=force.device)).sum(-1).clamp(0.,1.)
    return dict(cost=cost,components=clean(components),valid=valid,foot_reference_xy=clean(footref))


def potential_transition(previous, current, terminated, timeouts, *, gamma=GAMMA):
    """Raw per-transition F; timeout mask follows actual pre-state bootstrap.

    Do not clip positive/negative deltas separately. Creation, expiration and
    reset boundaries must participate in the same potential sequence.
    """
    if not 0.<gamma<1.:raise ValueError('gamma must lie in (0,1)')
    if previous.shape!=current.shape or terminated.shape!=previous.shape or timeouts.shape!=previous.shape:
        raise ValueError('Potential and done mask shapes must match')
    if not torch.isfinite(previous).all() or not torch.isfinite(current).all():raise ValueError('Nonfinite potential')
    endpoint=torch.where(terminated,torch.zeros_like(current),current)
    endpoint=torch.where(timeouts,previous,endpoint)
    reward=gamma*endpoint-previous
    cached=torch.where(terminated|timeouts,torch.zeros_like(current),current)
    return reward,cached


def reset_state(env, ids):
    from .joint_support_training_runtime_v1 import reset_state as reset_plan
    reset_plan(env,ids)
    if hasattr(env,'_joint_support_training_v1_previous_phi'):
        env._joint_support_training_v1_previous_phi[ids]=0.
    # RewardManager reset occurs after reward computation; preserve its final
    # transition/log and clear only episode potential, never other environments.


@torch.no_grad()
def joint_support_reference_reward(env):
    if getattr(env.cfg,'beamdojo_joint_support_v1_arm',None)!='reference':
        raise ValueError('Explicit joint reference experiment required')
    if getattr(env.cfg,'beamdojo_joint_support_v1_inference',False):
        # Privileged planner is training supervision, not part of actor inference.
        # Keep the term present for contract restoration; original rewards and
        # all course qualification continue to run in the native environment.
        return torch.zeros(env.num_envs,device=env.device)
    if getattr(env,'_joint_support_training_v1_reward_tick',None)==env.common_step_counter:
        return env._joint_support_training_v1_reward_rate
    from .joint_support_training_runtime_v1 import update
    from .contact_transition_reference_v3 import sample_two_contact_reference
    from .support_transfer_reward import centroidal_state
    plan=update(env,done=env.termination_manager.dones,
        event_batch_size=int(getattr(env.cfg,'beamdojo_joint_support_v1_event_batch_size',8)),
        max_events_per_step=int(getattr(env.cfg,'beamdojo_joint_support_v1_max_events_per_step',8)))
    now=float(env.common_step_counter)*env.step_dt
    reference=sample_two_contact_reference(plan['phase_times'],plan['com_path'],plan['velocity_path'],
        plan['momentum_path'],plan['cop'],plan['cmp'],plan['height'],plan['plan_mass'],now,
        foot_normal_load_fractions=plan['foot_normal_load_fractions'],plan_valid=plan['plan_valid'])
    robot=env.scene['robot'];data=robot.data;sensor=env.scene.sensors['contact_forces']
    if not hasattr(env,'_joint_support_training_v1_reward_ids'):
        names=['left_ankle_roll_link','right_ankle_roll_link']
        env._joint_support_training_v1_reward_ids=(robot.find_bodies(names,preserve_order=True)[0],sensor.find_bodies(names,preserve_order=True)[0])
        env._joint_support_training_v1_reward_masses=tensor(data.body_mass).clone()
        env._joint_support_training_v1_reward_inertias=tensor(data.body_inertia).clone().reshape(env.num_envs,-1,3,3)
    feet,contacts=env._joint_support_training_v1_reward_ids
    com,v,L,m=centroidal_state(env._joint_support_training_v1_reward_masses,tensor(data.body_com_pos_w),
        tensor(data.body_com_lin_vel_w),tensor(data.body_ang_vel_w),env._joint_support_training_v1_reward_inertias,tensor(data.body_com_quat_w))
    fz=tensor(sensor.data.net_forces_w)[:,contacts,2]
    actual=dict(com=com,velocity=v,momentum=L,mass=m,foot_positions=tensor(data.body_pos_w)[:,feet],contact=fz>5.,contact_fz=fz)
    result=bounded_reference_cost(actual,plan,reference,torch.full((env.num_envs,),now,device=env.device))
    moving=tensor(env.command_manager.get_command('base_velocity'))[:,:2].norm(dim=-1)>=.2
    valid=result['valid']&moving
    phi=torch.where(valid,-result['cost'],0.)
    previous=getattr(env,'_joint_support_training_v1_previous_phi',torch.zeros_like(phi))
    timeouts=env.termination_manager.time_outs.clone()
    terminated=env.termination_manager.terminated.clone()
    gamma=float(getattr(env.cfg,'beamdojo_joint_support_v1_gamma',GAMMA))
    raw,cached=potential_transition(previous,phi,terminated,timeouts,gamma=gamma)
    rate=raw/env.step_dt
    env._joint_support_training_v1_previous_phi=cached
    env._joint_support_training_v1_reward_tick=env.common_step_counter
    env._joint_support_training_v1_reward_rate=rate
    env._joint_support_training_v1_last=dict(raw=raw.clone(),cost=result['cost'],potential=phi,valid=valid,
        component_costs=result['components'],timeouts=timeouts,terminated=terminated)
    env._joint_support_training_v1_log={
        'JointReference/plan_coverage':valid.float().mean(),
        'JointReference/moving_without_plan':(moving&~valid).float().mean(),
        'JointReference/mean_cost':torch.where(valid,result['cost'],0.).mean(),
        'JointReference/mean_cost_when_valid':torch.where(valid,result['cost'],0.).sum()/valid.sum().clamp_min(1),
        'JointReference/shaping_transition_mean':raw.mean(),
        'JointReference/shaping_transition_abs':raw.abs().mean(),
        'JointReference/two_branch_fraction':(valid&(plan['selected_branch']==1)).float().mean(),
    }
    for key,value in plan.get('statistics',{}).items():
        if isinstance(value,(int,float)):
            env._joint_support_training_v1_log['JointReference/runtime_'+key]=float(value)
    for j,name in enumerate(('foot_xy','contact_schedule','support_load','com_xy','velocity','momentum')):
        env._joint_support_training_v1_log['JointReference/cost_'+name]=torch.where(valid,result['components'][:,j],0.).mean()
        env._joint_support_training_v1_log['JointReference/cost_when_valid_'+name]=torch.where(valid,result['components'][:,j],0.).sum()/valid.sum().clamp_min(1)
    return rate
