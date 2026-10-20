"""Necessary-condition pruning with original Cartesian combination identities.

No friction/objective pruning. Hold-B is independent of second position except
finite-input gating, so its first finite second-bank entry is the canonical
representative (index zero for the runtime's finite geometry bank). Retained
indices remain in global order. Both branches use original float32 timing and
kinematic expressions, preserving the complete set of potentially feasible
references. The original planner/selector remain unchanged.
"""
import torch
from .contact_transition_planner_v3 import plan_two_contact_transition
from .contact_sequence_selector_v3 import select_contact_sequence


def axes(first_count,second_count,device):
    combo=torch.cartesian_prod(torch.arange(first_count,device=device),torch.arange(second_count,device=device),torch.arange(4,device=device))
    return combo,torch.tensor([.36,.40,.44,.48],device=device)[combo[:,2]]


def necessary_masks(args,first_centers,second_points,second_mask,second_centers):
    """Return hold/two masks over the FULL original combination axis."""
    device=args[0].device;combo,tau2=axes(first_centers.shape[1],second_centers.shape[1],device)
    n=len(args[0]);row=torch.arange(n,device=device);foot=args[7].long().clamp(0,1)
    first,second=combo[:,0],combo[:,1];tau=args[10].float()[:,first];t2=tau2[None].expand(n,-1)
    centers=first_centers.float()[:,first];next_centers=second_centers.float()[:,second];feet=args[11].float()
    intermediate=t2-tau-.08;final=.6-t2-.08
    timing=(tau>0)&(intermediate>=.04-1e-6)&(final>=.04-1e-6)&(intermediate>0)&(final>0)
    firstspeed=torch.linalg.vector_norm(centers-feet[row,foot][:,None],dim=-1)/tau.clamp_min(1e-4)
    firststep=torch.linalg.vector_norm(centers[...,:2]-feet[row,1-foot][:,None,:2],dim=-1)
    secondspeed=torch.linalg.vector_norm(next_centers-feet[row,1-foot][:,None],dim=-1)/intermediate.clamp_min(1e-4)
    secondstep=torch.linalg.vector_norm(next_centers[...,:2]-centers[...,:2],dim=-1)
    shared=timing&(firstspeed<=2.5)&(firststep<=.7)
    # The original hold branch still checks finite second centers/valid points.
    finite_second=torch.isfinite(second_centers).all(-1)&(torch.isfinite(second_points).all(-1)|~second_mask).all(-1)
    canonical=finite_second.long().argmax(-1)
    hold=shared&(second[None]==canonical[:,None])&finite_second.any(-1)[:,None]
    two=shared&(secondspeed<=2.5)&(secondstep<=.7)
    return combo,tau2,hold,two,dict(timing=timing,first_kinematic=(firstspeed<=2.5)&(firststep<=.7),second_kinematic=(secondspeed<=2.5)&(secondstep<=.7),canonical_hold_second=canonical)


def padded_indices(mask):
    counts=mask.sum(-1);k=max(int(counts.max()),1)
    ids=torch.arange(mask.shape[1],device=mask.device)[None].expand_as(mask)
    ordered=torch.where(mask,ids,mask.shape[1]).sort(dim=-1,stable=True).values[:,:k]
    valid=ordered<mask.shape[1]
    return ordered.clamp_max(mask.shape[1]-1),valid


def gather_candidates(value,index):
    shape=(*index.shape,*value.shape[2:]);g=index.reshape(*index.shape,*([1]*(value.ndim-2))).expand(shape)
    return value.gather(1,g)


def evaluate(args,first_centers,second_points,second_mask,second_centers,combo,tau2,indices,enabled,chunk_size):
    first=combo[:,0][indices];second=combo[:,1][indices]
    a=list(args)
    for j in (8,9,10):a[j]=gather_candidates(args[j],first)
    fc=gather_candidates(first_centers,first);sc=gather_candidates(second_centers,second)
    sp=gather_candidates(second_points,second);sm=gather_candidates(second_mask,second)
    result=plan_two_contact_transition(*a,sp,sm,tau2[indices],candidate_centers=fc,second_candidate_centers=sc,
        second_contact_enabled=enabled,planning_horizon=.6,friction=.5,chunk_size=chunk_size)
    return result,fc,sc


def strict_gate(result,first_targets,second_targets,enabled):
    gate=result['feasible']&torch.isfinite(result['candidate_objective'])&(result['candidate_objective']>=0)
    fr=result['friction_ratio'];gate &= torch.isfinite(fr).all(-1)&(fr>=0).all(-1)&(fr<=.5).all(-1)
    gate &= (result['phase_durations']>0).all(-1)&torch.isfinite(first_targets).all(-1)
    if enabled:gate &= torch.isfinite(second_targets).all(-1)
    for key in ('com_path','velocity_path','momentum_path','cop','cmp'):gate &= torch.isfinite(result[key]).flatten(2).all(-1)
    gate &= torch.isfinite(result['height'])[:,None]&(result['height']>0)[:,None]
    return gate


@torch.no_grad()
def plan_pruned(args,first_centers,second_points,second_mask,second_centers,now,*,chunk_size=256):
    combo,tau2,holdmask,twomask,_=necessary_masks(args,first_centers,second_points,second_mask,second_centers)
    n=len(args[0]);row=torch.arange(n,device=args[0].device);winners=[];winner_valid=[];evaluated=[]
    for enabled,mask in ((False,holdmask),(True,twomask)):
        indices,padding=padded_indices(mask)
        result,fc,sc=evaluate(args,first_centers,second_points,second_mask,second_centers,combo,tau2,indices,enabled,chunk_size)
        gate=strict_gate(result,fc,sc,enabled)&padding
        ranked=torch.where(gate,result['candidate_objective'],float('inf'));score,best=ranked.min(-1)
        winners.append(indices[row,best]);winner_valid.append(torch.isfinite(score));evaluated.append(padding.sum(-1))
        del result
    final_indices=torch.stack(winners,-1) # hold winner then two winner; original tie priority retained.
    h,fc,sc=evaluate(args,first_centers,second_points,second_mask,second_centers,combo,tau2,final_indices,False,chunk_size)
    t,_,_=evaluate(args,first_centers,second_points,second_mask,second_centers,combo,tau2,final_indices,True,chunk_size)
    h['feasible'] &= torch.stack((winner_valid[0],torch.zeros_like(winner_valid[0])),-1)
    t['feasible'] &= torch.stack((torch.zeros_like(winner_valid[1]),winner_valid[1]),-1)
    chosen=select_contact_sequence(h,t,now,first_targets=fc,second_targets=sc)
    index=final_indices[row,chosen['selected_index'].clamp_min(0)]
    chosen['selected_index']=torch.where(chosen['valid'],index,-1)
    chosen['combo_ids']=torch.where(chosen['valid'][:,None],combo[index],-1)
    chosen['enumerated_combinations']=torch.full((n,),len(combo),device=index.device,dtype=torch.long)
    chosen['hold_evaluated_candidates']=evaluated[0];chosen['two_evaluated_candidates']=evaluated[1]
    return chosen
