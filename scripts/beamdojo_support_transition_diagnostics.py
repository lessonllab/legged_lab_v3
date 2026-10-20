"""Frozen-policy interventions on the residual, leaving the base gait intact."""
import torch


def residual_mask(joint_names, mode, device=None):
    if mode not in ('all', 'no_arms', 'leg_pitch_roll'):
        raise ValueError('Unknown residual intervention: '+mode)
    values=[]
    for name in joint_names:
        arm=any(part in name for part in ('shoulder_', 'elbow_', 'wrist_'))
        if mode=='all': keep=True
        elif mode=='no_arms': keep=not arm
        else: keep=not arm and 'waist_' not in name and 'hip_yaw_' not in name
        values.append(float(keep))
    return torch.tensor(values,device=device)


def intervene(actor, joint_names, mode):
    """Evaluation only: recompute native mean tails after masking raw delta.

    Filtering the already bounded output would be a different intervention.
    Neither checkpoint tensors nor observation inputs are modified here.
    """
    if mode=='all': return None
    if not hasattr(actor,'last_transition_delta'):
        raise ValueError('Residual intervention requires a support-transition actor')
    mask=residual_mask(joint_names,mode,next(actor.parameters()).device)
    def filter_delta(module,args,kwargs,output):
        if kwargs.get('stochastic_output',False):
            raise RuntimeError('Diagnostic residual intervention is deterministic evaluation only')
        obs=args[0]
        raw=torch.cat([obs[k] for k in module.obs_groups],-1)
        with torch.no_grad():
            base=module.mlp(module.obs_normalizer(raw))
            delta=module.last_transition_delta*mask
            module.last_transition_delta=delta
            return module.distribution.deterministic_output(base+delta)
    return actor.register_forward_hook(filter_delta,with_kwargs=True)
