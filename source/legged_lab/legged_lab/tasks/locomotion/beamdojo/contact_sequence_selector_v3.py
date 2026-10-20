"""Pure union selector for matched seven-phase hold-B/two-contact candidates.

No action, reward, or runtime commitment is installed. The caller must call
once at an accepted release opportunity and lock the selected absolute times;
repeated selection must not continually postpone a landing deadline.
"""
import torch


def select_contact_sequence(hold_result,two_result,decision_time,*,first_targets=None,second_targets=None):
    """Choose minimum original objective over the two feasible candidate sets.

    Branch IDs: 0=hold-B, 1=two-contact, -1=no solution. Equal objectives prefer
    hold-B, then the lowest candidate index. Inputs must have the SAME [N,C]
    candidate enumeration, seven durations, H and two relative time axes.
    The two-contact second deadline/target has NO semantics for a hold branch.
    Missing or invalid selected payload is NaN, never a valid origin target.
    Optional target arrays are world foot-link positions [N,C,3]; provided
    nonfinite targets invalidate the corresponding branch candidate.
    """
    h,t=hold_result,two_result
    if h.get('second_contact_enabled') is not False or t.get('second_contact_enabled') is not True:
        raise ValueError('Explicit hold=False and two=True branches required')
    objective=h['candidate_objective']
    if objective.ndim!=2:raise ValueError('candidate objective must be [N,C]')
    n,c=objective.shape;device=objective.device;dtype=objective.dtype
    shapes={'candidate_objective':(n,c),'feasible':(n,c),'friction_ratio':(n,c,7),
        'phase_times':(n,c,8),'phase_durations':(n,c,7),'total_horizon':(n,c),
        'candidate_touchdown_times':(n,c),'second_candidate_touchdown_times':(n,c),
        'first_touchdown_deadline':(n,c),'second_touchdown_deadline':(n,c),
        'com_path':(n,c,8,3),'velocity_path':(n,c,8,3),'momentum_path':(n,c,8,3),
        'cop':(n,c,7,2),'cmp':(n,c,7,2),'foot_normal_load_fractions':(n,c,7,2),'height':(n,)}
    for result in (h,t):
        for key,shape in shapes.items():
            if key not in result or result[key].shape!=shape:raise ValueError('Branch shape mismatch: '+key)
            if result[key].device!=device:raise ValueError('Branches must share device')
        if result['feasible'].dtype!=torch.bool:raise ValueError('feasible must be boolean')
    same=('phase_times','phase_durations','total_horizon','candidate_touchdown_times','second_candidate_touchdown_times',
          'first_touchdown_deadline','second_touchdown_deadline','height')
    for key in same:
        if not torch.equal(h[key],t[key]):raise ValueError('Branch time/profile mismatch: '+key)
    for key in ('first_support_foot','second_support_foot'):
        if key not in h or key not in t or h[key].shape!=(n,) or not torch.equal(h[key],t[key]):raise ValueError('Branch physical foot ordering mismatch')
    phases=h['phase_times'];durations=h['phase_durations']
    if not torch.isfinite(phases).all() or not torch.isfinite(durations).all():raise ValueError('Nonfinite shared timeline')
    if not torch.allclose(phases[...,0],torch.zeros_like(phases[...,0])):raise ValueError('Input phase times must start at relative zero')
    if not torch.allclose(phases[...,1:]-phases[...,:-1],durations,atol=1e-6,rtol=1e-6):raise ValueError('Phase duration identity mismatch')
    for left,right in ((phases[...,-1],h['total_horizon']),(phases[...,1],h['candidate_touchdown_times']),
                       (phases[...,4],h['second_candidate_touchdown_times']),
                       (h['first_touchdown_deadline'],h['candidate_touchdown_times']),
                       (h['second_touchdown_deadline'],h['second_candidate_touchdown_times'])):
        if not torch.allclose(left,right,atol=1e-6,rtol=1e-6):raise ValueError('Deadline/horizon identity mismatch')
    time=torch.as_tensor(decision_time,device=device,dtype=dtype)
    if time.ndim==0:time=time.expand(n)
    if time.shape!=(n,) or not torch.isfinite(time).all():raise ValueError('decision_time must be finite scalar or [N]')
    target_valid=[]
    for value in (first_targets,second_targets):
        if value is None:target_valid.append(torch.ones((n,c),device=device,dtype=torch.bool))
        else:
            if value.shape!=(n,c,3) or value.device!=device:raise ValueError('Target arrays must be [N,C,3] on branch device')
            target_valid.append(torch.isfinite(value).all(-1))
    gates=[];ranked=[]
    hold_shares=torch.tensor([[1,0],[.75,.25],[.25,.75],[0,1],[0,1],[0,1],[0,1]],device=device,dtype=dtype)
    two_shares=torch.tensor([[1,0],[.75,.25],[.25,.75],[0,1],[.25,.75],[.75,.25],[1,0]],device=device,dtype=dtype)
    for branch,result in enumerate((h,t)):
        expected=(hold_shares if branch==0 else two_shares)[None,None].expand(n,c,-1,-1)
        if not torch.equal(result['foot_normal_load_fractions'],expected):raise ValueError('Physical branch load fractions mismatch')
        score=result['candidate_objective'];friction=result['friction_ratio']
        gate=result['feasible']&torch.isfinite(score)&(score>=0)&torch.isfinite(friction).all(-1)&(friction>=0).all(-1)&(friction<=.5).all(-1)
        gate &= (durations>0).all(-1)&target_valid[0]
        if branch==1:gate &= target_valid[1]
        for field in ('com_path','velocity_path','momentum_path','cop','cmp'):
            gate &= torch.isfinite(result[field]).flatten(2).all(-1)
        gate &= torch.isfinite(result['height'])[:,None]&(result['height']>0)[:,None]
        gates.append(gate);ranked.append(torch.where(gate,score,float('inf')))
    if c:
        scores=torch.cat(ranked,-1);best_score,best=scores.min(-1);valid=torch.isfinite(best_score)
        branch=torch.where(valid,best//c,-1);index=torch.where(valid,best%c,-1)
    else:
        valid=torch.zeros(n,device=device,dtype=torch.bool);branch=torch.full((n,),-1,device=device,dtype=torch.long);index=branch.clone()
        best_score=torch.full((n,),float('nan'),device=device,dtype=dtype)
    def gather(key):
        shape=h[key].shape[2:];out=torch.full((n,*shape),float('nan'),device=device,dtype=h[key].dtype)
        for branch_id,result in enumerate((h,t)):
            rows=(valid&(branch==branch_id)).nonzero().flatten()
            if len(rows):out[rows]=result[key][rows,index[rows]]
        return out
    selected={key:gather(key) for key in ('com_path','velocity_path','momentum_path','cop','cmp','foot_normal_load_fractions')}
    selected['phase_times']=gather('phase_times')+time[:,None]
    first_deadline=gather('candidate_touchdown_times')+time
    second_semantic=valid&(branch==1)
    second_deadline=torch.where(second_semantic,gather('second_candidate_touchdown_times')+time,float('nan'))
    def target(value,semantic):
        if value is None:return None,torch.zeros_like(valid)
        out=torch.full((n,3),float('nan'),device=device,dtype=value.dtype);rows=semantic.nonzero().flatten()
        if len(rows):out[rows]=value[rows,index[rows]]
        return out,semantic.clone()
    first_target,first_target_valid=target(first_targets,valid)
    second_target,second_target_valid=target(second_targets,second_semantic)
    return dict(valid=valid,selected_branch=branch,selected_index=index,selected_objective=torch.where(valid,best_score,float('nan')),
        first_touchdown_deadline=first_deadline,second_touchdown_deadline=second_deadline,second_touchdown_valid=second_semantic,
        trajectory_valid_until=selected['phase_times'][:,-1],first_target=first_target,first_target_valid=first_target_valid,
        second_target=second_target,second_target_valid=second_target_valid,height=torch.where(valid,h['height'],float('nan')),
        hold_eligible=gates[0],two_eligible=gates[1],**selected)
