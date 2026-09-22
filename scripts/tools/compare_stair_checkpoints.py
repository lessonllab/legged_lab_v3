"""Compare frozen checkpoints on fixed stair heights/speeds and flat speed caps."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply_inverse
from legged_lab.tasks.locomotion.amp.mdp.foot_support import support_costs
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
from isaaclab.managers import CurriculumTermCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import legged_lab.tasks
from legged_lab.tasks.locomotion.amp.mdp.stair_course import StairTargetCommand
from legged_lab.tasks.locomotion.amp.mdp.style_state import tensor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=2001)
    p.add_argument('--eval-envs', type=int, default=192)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sampled', action='store_true')
    p.add_argument('--audit-foot-support', action='store_true', help='Measure loaded sole-edge proxies over all moving frames.')
    p.add_argument('--task', default='LeggedLab-Isaac-AMP-Stairs-G1-Play-v0',
                   choices=['LeggedLab-Isaac-AMP-Stairs-G1-Play-v0', 'LeggedLab-Isaac-AMP-Stairs-G1-Play-v1',
                            'LeggedLab-Isaac-AMP-Stairs-Long-G1-Play-v0'])
    p.add_argument('--stair-speeds', nargs='+', type=float, default=[.45, .65])
    p.add_argument('--flat-speeds', nargs='+', type=float, default=[.8, 1.2, 1.8])
    add_launcher_args(p)
    args=p.parse_args()
    if any(not 0. < speed < float('inf') for speed in args.stair_speeds + args.flat_speeds):
        p.error('Evaluation speeds must be finite and positive')
    task=args.task
    cfg=load_cfg_from_registry(task,'env_cfg_entry_point')
    cfg.scene.num_envs=args.eval_envs; cfg.seed=args.seed
    cfg.observations.policy.enable_corruption=args.sampled
    cfg.sim.physics=cfg.sim.physics.default.copy()
    agent=handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(task,'rsl_rl_cfg_entry_point'),importlib.metadata.version('rsl-rl-lib'))
    active=False; records=[]; acc={}; fixed_caps=None; labels=None
    foot_acc={}; edge_run=None; max_edge_run=None
    original_update=StairTargetCommand._update_metrics
    original_resample=StairTargetCommand._resample_command
    def resample(c, env_ids):
        original_resample(c,env_ids)
        if fixed_caps is not None:
            ids=c._ids(env_ids)
            c.speed_cap[ids]=fixed_caps[ids]
            c._update_selected(ids)
    def update(c):
        valid=~c._skip_progress_transition.clone()
        original_update(c)
        if not active:return
        speed=tensor(c.robot.data.root_lin_vel_w)[:,:2].norm(dim=-1)
        command=c.vel_command_b[:,:2].norm(dim=-1)
        moving=valid & (command>.3)
        fast=valid & (command>1.)
        for name,value in [('frames',valid.float()),('moving_frames',moving.float()),
                           ('speed',speed*moving),('command',command*moving),
                           ('fast_frames',fast.float()),('fast_speed',speed*fast)]:
            acc[name]+=value
        if args.audit_foot_support:
            body_pos=tensor(c.robot.data.body_pos_w)[:,c.ankle_ids]
            body_quat=tensor(c.robot.data.body_quat_w)[:,c.ankle_ids]
            heights=[]
            for i, side in enumerate(('left','right')):
                hits=tensor(env.scene[f'{side}_sole_scanner'].data.ray_hits_w)
                q=body_quat[:,i:i+1].expand(-1,55,-1)
                heights.append(quat_apply_inverse(q,hits-body_pos[:,i:i+1])[...,2].reshape(-1,5,11))
            h=torch.stack(heights,1)
            contact=env.scene['contact_forces']
            foot_ids=[contact.body_names.index(f'{side}_ankle_roll_link') for side in ('left','right')]
            force=tensor(contact.data.net_normal_forces_w)[:,foot_ids,2].clamp_min(0.)
            age=tensor(contact.data.current_contact_time)[:,foot_ids]
            _,edge,fraction=support_costs(h.reshape(-1,1,5,11),torch.full_like(force.reshape(-1,1),100.),
                                        torch.ones_like(force.reshape(-1,1)),support_target=.9,immediate_edge=True)
            edge=edge.reshape(-1,2)>0
            fraction=fraction.reshape(-1,2)
            loaded=(force>20.) & moving[:,None]
            near=loaded & edge
            overhang=near & (fraction<.9)
            # All-frame audit: includes final partial episodes; exclude resets,
            # swing, and commanded stops so standing cannot dilute the rates.
            edge_run[:] = torch.where(near,edge_run+env.step_dt,0.)
            max_edge_run[:] = torch.maximum(max_edge_run,edge_run)
            for name,value in [('loaded',loaded),('near_edge',near),('edge_under_supported',overhang),
                               ('edge_after_80ms',near & (age>=.08)),
                               ('edge_run_over_100ms',near & (edge_run>=.1)),
                               ('support_fraction',fraction*loaded),
                               ('vertical_impulse',force*loaded*env.step_dt),
                               ('edge_vertical_impulse',force*near*env.step_dt)]:
                foot_acc[name]+=value.sum(-1)
    def reset_audit(env,env_ids):
        c=env.command_manager.get_term('base_velocity'); ids=c._ids(env_ids)
        if active:
            geometry=c.exit_geometry()
            crossing_mode = getattr(c, 'promotion_version', 0) >= 7
            for i in ids[tensor(env.episode_length_buf)[ids]>0].tolist():
                row={'env':i,'case':labels[i], 'group':int(c.group[i]),
                     'duration':float(tensor(env.episode_length_buf)[i])*env.step_dt,
                     'failed':bool(tensor(env.termination_manager.terminated)[i]),
                     'traversed':bool(c.stair_traversed[i]),
                     'settled':bool(c.stair_completed[i]),
                     'passed':bool(c.stair_completed[i] & geometry[i] & ~tensor(env.termination_manager.terminated)[i].bool()),
                     'tracking_xy':float(c.metrics['moving_tracking_exp_vel_xy'][i])}
                row.update({k:float(v[i]) for k,v in acc.items()})
                if crossing_mode and bool(c.stairs[i]):
                    row['traversed'] = bool(c.stair_traversed[i] | geometry[i])
                    row['passed'] = row['traversed'] and not row['failed']
                if hasattr(c, 'speed_success'):
                    row['terminal_geometry'] = bool(geometry[i])
                    row['stair_phase'] = int(c.stair_phase[i])
                    for key, value in c.metrics.items():
                        if 'cruise_' in key:
                            row[key] = float(value[i])
                    row['speed_passed'] = bool(c.speed_success()[i])
                records.append(row)
            for v in acc.values():v[ids]=0
            if args.audit_foot_support:edge_run[ids]=0
        c.skip_curriculum_once[ids]=False
        return {}
    cfg.curriculum.terrain_levels=CurriculumTermCfg(func=reset_audit)
    StairTargetCommand._update_metrics=update
    StairTargetCommand._resample_command=resample
    results=[]
    with launch_simulation(cfg,args):
        env=gym.make(task,cfg=cfg).unwrapped
        try:
            wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
            runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=env.device)
            c=env.command_manager.get_term('base_velocity')
            for checkpoint in args.checkpoints:
                active=False; fixed_caps=None; records=[]
                runner.load(checkpoint,map_location=env.device); runner.alg.eval_mode()
                levels=tensor(c.terrain.terrain_levels)
                labels=['other']*args.eval_envs
                fixed_caps=torch.full((args.eval_envs,),.45,device=env.device)
                for group in (0,1,2,3):
                    ids=torch.where(c.group==group)[0]
                    for j,i in enumerate(ids.tolist()):
                        if group in (1,2):
                            level=(1,2,4)[j%3]; speed_index=(j//3)%len(args.stair_speeds)
                            levels[i]=level; c.stair_phase[i]=0
                            speed=args.stair_speeds[speed_index]; fixed_caps[i]=speed
                            labels[i]=f'{"ascent" if group==1 else "descent"}/{(8,10,14)[j%3]}cm/{speed}'
                        elif group==0:
                            levels[i]=0; speed=args.flat_speeds[j%len(args.flat_speeds)]; fixed_caps[i]=speed
                            labels[i]=f'flat/{speed}'
                        else:levels[i]=1
                c.sync_origins(); c.skip_curriculum_once[:]=True
                torch.manual_seed(args.seed); env.reset()
                runner.alg.actor.reset(torch.ones(args.eval_envs,dtype=torch.bool,device=env.device))
                acc={k:torch.zeros(args.eval_envs,device=env.device) for k in ('frames','moving_frames','speed','command','fast_frames','fast_speed')}
                foot_acc={k:torch.zeros(args.eval_envs,device=env.device) for k in (
                    'loaded','near_edge','edge_under_supported','edge_after_80ms','edge_run_over_100ms',
                    'support_fraction','vertical_impulse','edge_vertical_impulse')}
                edge_run=torch.zeros(args.eval_envs,2,device=env.device)
                max_edge_run=torch.zeros_like(edge_run)
                active=True; obs=wrapped.get_observations()
                with torch.inference_mode():
                    for step in range(args.steps):
                        actions=runner.alg.actor(obs,stochastic_output=args.sampled)
                        obs,_,done,_=wrapped.step(actions); runner.alg.actor.reset(done)
                        if (step+1)%500==0:print('COMPARE_PROGRESS',Path(checkpoint).name,step+1,flush=True)
                active=False
                summary={}
                for case in sorted(set(labels)):
                    rows=[r for r in records if r['case']==case]; n=len(rows)
                    if not n:continue
                    moving=sum(r['moving_frames'] for r in rows); fast=sum(r['fast_frames'] for r in rows)
                    summary[case]={'episodes':n,'failed':sum(r['failed'] for r in rows),
                        'traversed':sum(r['traversed'] for r in rows),'settled':sum(r['settled'] for r in rows),
                        'passed':sum(r['passed'] for r in rows),
                        'mean_duration':sum(r['duration'] for r in rows)/n,
                        'moving_speed':sum(r['speed'] for r in rows)/max(1,moving),
                        'moving_command':sum(r['command'] for r in rows)/max(1,moving),
                        'fast_command_speed':sum(r['fast_speed'] for r in rows)/max(1,fast),
                        'tracking_xy':sum(r['tracking_xy'] for r in rows)/n}
                foot_audit={}
                if args.audit_foot_support:
                    for case in sorted(set(labels)):
                        ids=torch.tensor([i for i,label in enumerate(labels) if label==case],device=env.device)
                        counts={k:float(v[ids].sum()) for k,v in foot_acc.items()}
                        denom=max(1.,counts['loaded'])
                        foot_audit[case]={**counts,
                            'near_edge_fraction':counts['near_edge']/denom,
                            'edge_under_supported_fraction':counts['edge_under_supported']/denom,
                            'edge_after_80ms_fraction':counts['edge_after_80ms']/denom,
                            'edge_run_over_100ms_fraction':counts['edge_run_over_100ms']/denom,
                            'mean_support_fraction':counts['support_fraction']/denom,
                            'edge_vertical_impulse_fraction':counts['edge_vertical_impulse']/max(1e-6,counts['vertical_impulse']),
                            'max_continuous_edge_seconds':float(max_edge_run[ids].max())}
                results.append({'checkpoint':checkpoint,'summary':summary,'episodes':records,'foot_audit':foot_audit})
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps({'seed':args.seed,'steps':args.steps,'num_envs':args.eval_envs,
                    'task':task, 'stair_speed_caps':args.stair_speeds, 'flat_speed_caps':args.flat_speeds,
                    'terrain_profile':getattr(c, 'terrain_profile', 'short8_v1'),
                    'tile_size':list(cfg.scene.terrain.terrain_generator.size), 'episode_length_s':cfg.episode_length_s,
                    'sampled_actions_and_noisy_observations':args.sampled,
                    'completion_mode':'crossing_without_stop' if getattr(c, 'promotion_version', 0) >= 7 else 'exit_settling',
                    'foot_audit_scope':'All valid moving frames (command >0.3 m/s); loaded foot Fz>20 N. Terrain-ray proxies, not actual contact area or pressure. Edge includes the outer 2 cm ring. Fractions use loaded foot-frames; max duration may include normal toe-off. No claim of measured edge contact points.',
                    'scope':'Frozen curriculum; fixed stair heights 8/10/14 cm; speed caps are recorded above. PLAY reset distribution. Completed episodes only; final partial episodes excluded. Common seed and case assignment, later resets depend on policy trajectories. passed uses the recorded completion_mode, not speed mastery; inspect actual versus commanded speed separately.',
                    'results':results},indent=2))
                print('COMPARE_RESULT',Path(checkpoint).name,json.dumps(summary),flush=True)
        finally:
            StairTargetCommand._update_metrics=original_update
            StairTargetCommand._resample_command=original_resample
            env.close()

if __name__=='__main__':main()
