"""Training-only direct reference cost, charged only inside valid plan windows.

This changes the training objective: it is not PBRS and makes no policy
invariance claim. The frozen v1 six costs and locked planner are reused, but
there are no potential creation, expiry, terminal or timeout jumps. Return a
rate -lambda*C; RewardManager alone integrates it once with control dt.
Privileged planning is disabled during actor inference. No-plan windows are
uncharged; coverage and dropped/no-solution events must accompany cost logs.
The inherited closed interval includes creation and H=.6 endpoint samples:
at dt=.02, at most 31 charged samples per plan (.62 s of discrete exposure).
Expiry returns zero and never refunds earlier costs.
"""
import math
import torch
from .joint_support_training_reward_v1 import bounded_reference_cost, tensor, WEIGHTS

DEFAULT_LAMBDA = 1.0
COMPONENT_NAMES = ('foot_xy', 'contact_schedule', 'support_load', 'com_xy', 'velocity', 'momentum')


def direct_cost_rate(cost, valid, coefficient=DEFAULT_LAMBDA):
    """Bounded cost rate, without any dt or potential-state dependence."""
    if not math.isfinite(coefficient) or coefficient < 0.:
        raise ValueError('Direct reference lambda must be finite and nonnegative')
    if cost.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError('Cost and boolean validity must share shape')
    if not torch.isfinite(cost[valid]).all() or ((cost[valid] < 0.) | (cost[valid] > 1.)).any():
        raise ValueError('Valid reference cost must be finite and within [0,1]')
    return -coefficient * torch.where(valid, cost, torch.zeros_like(cost))


def reset_state(env, ids):
    from .joint_support_training_runtime_v1 import reset_state as reset_plan
    reset_plan(env, ids)
    if hasattr(env, '_joint_support_training_v2_episode_integrated_cost'):
        env._joint_support_training_v2_episode_integrated_cost[ids] = 0.
    # Native autoreset follows reward calculation. Preserve this tick's rate
    # and last-transition diagnostics; another reward call must not charge twice.


def _terminal_snapshot(env, done, now, moving):
    """Keep the existing pre-reset plan for a terminal transition, never create one."""
    if not bool(done.any()):
        return None
    from .joint_support_training_runtime_v1 import state
    current = state(env)
    ids = (done & current.valid & (now <= current.lock_until) & moving).nonzero().flatten()
    # update(done=...) invalidates the validity flag, but preserves the locked
    # payload until native reset. Save only these terminal IDs, not 4096 paths.
    return ids


@torch.no_grad()
def joint_support_reference_reward(env):
    if getattr(env.cfg, 'beamdojo_joint_support_v2_arm', None) != 'reference':
        raise ValueError('Explicit v2 direct-reference experiment required')
    if getattr(env.cfg, 'beamdojo_joint_support_v2_inference', False):
        return torch.zeros(env.num_envs, device=env.device)
    if getattr(env, '_joint_support_training_v2_reward_tick', None) == env.common_step_counter:
        return env._joint_support_training_v2_reward_rate
    coefficient = float(getattr(env.cfg, 'beamdojo_joint_support_v2_lambda', DEFAULT_LAMBDA))
    if not math.isfinite(coefficient) or coefficient < 0.:
        raise ValueError('Direct reference lambda must be finite and nonnegative')
    if not math.isfinite(env.step_dt) or env.step_dt <= 0.:
        raise ValueError('Positive finite control dt required')
    from .joint_support_training_runtime_v1 import update
    from .contact_transition_reference_v3 import sample_two_contact_reference
    from .support_transfer_reward import centroidal_state
    now = float(env.common_step_counter) * env.step_dt
    moving = tensor(env.command_manager.get_command('base_velocity'))[:, :2].norm(dim=-1) >= .2
    done = env.termination_manager.dones.clone()
    prior = _terminal_snapshot(env, done, now, moving)
    plan = update(env, done=done,
        event_batch_size=int(getattr(env.cfg, 'beamdojo_joint_support_v2_event_batch_size', 8)),
        max_events_per_step=int(getattr(env.cfg, 'beamdojo_joint_support_v2_max_events_per_step', 8)))
    if prior is not None and len(prior):
        plan = dict(plan)
        plan['plan_valid'] = plan['plan_valid'].clone()
        plan['plan_valid'][prior] = True
    reference = sample_two_contact_reference(plan['phase_times'], plan['com_path'], plan['velocity_path'],
        plan['momentum_path'], plan['cop'], plan['cmp'], plan['height'], plan['plan_mass'], now,
        foot_normal_load_fractions=plan['foot_normal_load_fractions'], plan_valid=plan['plan_valid'])
    data = env.scene['robot'].data
    sensor = env.scene.sensors['contact_forces']
    if not hasattr(env, '_joint_support_training_v2_reward_ids'):
        names = ['left_ankle_roll_link', 'right_ankle_roll_link']
        env._joint_support_training_v2_reward_ids = (
            env.scene['robot'].find_bodies(names, preserve_order=True)[0],
            sensor.find_bodies(names, preserve_order=True)[0])
    feet, contacts = env._joint_support_training_v2_reward_ids
    # Match frozen v1: cache mass and inertia once, preserving the same
    # centroidal-state definition while changing only the reward formulation.
    if not hasattr(env, '_joint_support_training_v2_reward_masses'):
        env._joint_support_training_v2_reward_masses = tensor(data.body_mass).clone()
        env._joint_support_training_v2_reward_inertias = tensor(data.body_inertia).clone().reshape(env.num_envs, -1, 3, 3)
    com, velocity, momentum, mass = centroidal_state(env._joint_support_training_v2_reward_masses, tensor(data.body_com_pos_w),
        tensor(data.body_com_lin_vel_w), tensor(data.body_ang_vel_w),
        env._joint_support_training_v2_reward_inertias, tensor(data.body_com_quat_w))
    fz = tensor(sensor.data.net_forces_w)[:, contacts, 2]
    actual = dict(com=com, velocity=velocity, momentum=momentum, mass=mass,
        foot_positions=tensor(data.body_pos_w)[:, feet], contact=fz > 5., contact_fz=fz)
    result = bounded_reference_cost(actual, plan, reference, torch.full((env.num_envs,), now, device=env.device))
    valid = result['valid'] & moving
    rate = direct_cost_rate(result['cost'], valid, coefficient)
    integrated = -rate * env.step_dt  # Positive cost actually deducted this transition.
    if not hasattr(env, '_joint_support_training_v2_episode_integrated_cost'):
        env._joint_support_training_v2_episode_integrated_cost = torch.zeros_like(rate)
    env._joint_support_training_v2_episode_integrated_cost += integrated
    env._joint_support_training_v2_reward_tick = env.common_step_counter
    env._joint_support_training_v2_reward_rate = rate
    env._joint_support_training_v2_last = dict(rate=rate.clone(), raw=rate * env.step_dt,
        integrated_cost=integrated.clone(), cost=result['cost'].clone(), valid=valid.clone(),
        component_costs=result['components'].clone(), terminated=env.termination_manager.terminated.clone(),
        timeouts=env.termination_manager.time_outs.clone(),
        episode_integrated_cost=env._joint_support_training_v2_episode_integrated_cost.clone())
    masked_cost = torch.where(valid, result['cost'], 0.)
    count = valid.sum().clamp_min(1)
    without = moving & ~valid
    speed = actual['velocity'][:, :2].norm(dim=-1)
    log = {
        'JointReferenceDirect/lambda': coefficient,
        'JointReferenceDirect/plan_coverage': valid.float().mean(),
        'JointReferenceDirect/moving_without_plan': (moving & ~valid).float().mean(),
        'JointReferenceDirect/mean_cost': masked_cost.mean(),
        'JointReferenceDirect/mean_cost_when_valid': masked_cost.sum() / count,
        'JointReferenceDirect/direct_rate_mean': rate.mean(),
        'JointReferenceDirect/integrated_cost_mean': integrated.mean(),
        'JointReferenceDirect/two_branch_fraction': (valid & (plan['selected_branch'] == 1)).float().mean(),
        'JointReferenceDirect/terminal_charged_fraction': (valid & done).float().mean(),
        'JointReferenceDirect/planar_com_speed_when_valid': torch.where(valid, speed, 0.).sum() / count,
        'JointReferenceDirect/planar_com_speed_moving_without_plan': torch.where(without, speed, 0.).sum() / without.sum().clamp_min(1),
    }
    for key, value in plan.get('statistics', {}).items():
        if isinstance(value, (int, float)):
            log['JointReferenceDirect/runtime_' + key] = float(value)
    for j, name in enumerate(COMPONENT_NAMES):
        component = torch.where(valid, result['components'][:, j], 0.)
        log['JointReferenceDirect/cost_' + name] = component.mean()
        log['JointReferenceDirect/cost_when_valid_' + name] = component.sum() / count
        log['JointReferenceDirect/saturated_when_valid_' + name] = (valid & (result['components'][:, j] >= 1.)).sum() / count
    env._joint_support_training_v2_log = log
    return rate
