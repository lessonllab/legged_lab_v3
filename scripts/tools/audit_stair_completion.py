"""Frozen-policy stair completion audit. No policy updates or course changes."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import gymnasium as gym
import torch
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
from isaaclab.managers import CurriculumTermCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import legged_lab.tasks
from legged_lab.tasks.locomotion.amp.mdp.stair_course import StairTargetCommand
from legged_lab.tasks.locomotion.amp.mdp.style_state import tensor


def inspect(c):
    d=c.robot.data
    root=tensor(d.root_pos_w); feet=tensor(d.body_pos_w)[:,c.ankle_ids];goal=c.pos_command_w
    distance=(root[:,:2]-goal[:,:2]).norm(dim=-1)
    speed=tensor(d.root_lin_vel_w)[:,:2].norm(dim=-1)
    angular=tensor(d.root_ang_vel_w).norm(dim=-1)
    height=root[:,2]-goal[:,2]
    masks={
        'near':distance<=.4,
        'feet_x':(feet[:,:,0]>=goal[:,None,0]-.6).all(-1),
        'feet_z':((feet[:,:,2]-goal[:,None,2]).abs()<.12).all(-1),
        'bounds':(feet[:,:,0]<goal[:,None,0]+.3).all(-1)&((feet[:,:,1]-goal[:,None,1]).abs()<3.95).all(-1),
        'upright':tensor(d.projected_gravity_b)[:,2]<-.90,
        'height':(height>.4)&(height<1.1),
        'slow':speed<.2,
        'angular':angular<.3,
    }
    masks['geometric']=torch.stack([masks[k] for k in ('near','feet_x','feet_z','bounds','upright','height')]).all(0)
    masks['all']=masks['geometric']&masks['slow']&masks['angular']
    masks['all_except_z']=masks['near']&masks['feet_x']&masks['bounds']&masks['upright']&masks['height']&masks['slow']&masks['angular']
    values=torch.stack([distance,speed,angular,height,feet[:,0,2]-goal[:,2],feet[:,1,2]-goal[:,2],
                        root[:,0]-tensor(c.terrain.env_origins)[:,0],goal[:,2],root[:,2],
                        c.vel_command_b[:,0],c.stair_hold],-1)
    return masks,values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2200)
    p.add_argument('--modes',nargs='+',default=['deterministic','sampled'],choices=['deterministic','sampled'])
    p.add_argument('--training-observations',action='store_true',help='Also enable policy observation corruption')
    add_launcher_args(p);args=p.parse_args()
    task='LeggedLab-Isaac-AMP-Stairs-G1-Play-v0'
    cfg=load_cfg_from_registry(task,'env_cfg_entry_point');cfg.scene.num_envs=64;cfg.seed=42
    cfg.observations.policy.enable_corruption=args.training_observations
    cfg.sim.physics=cfg.sim.physics.default.copy()
    agent=load_cfg_from_registry(task,'rsl_rl_cfg_entry_point')
    agent=handle_deprecated_rsl_rl_cfg(agent,importlib.metadata.version('rsl-rl-lib'))
    records=[];mode='initial';buffers={}; frame_summary={}
    original=StairTargetCommand._update_metrics
    def update(c):
        valid=~c._skip_progress_transition.clone()
        original(c)
        if mode=='initial':return
        masks,values=inspect(c)
        if not buffers:
            buffers.update(ever={k:torch.zeros_like(c.flat_pool) for k in masks},
                           hold=torch.zeros_like(c.stair_hold), max_hold=torch.zeros_like(c.stair_hold),
                           max_actual_hold=torch.zeros_like(c.stair_hold),
                           min_distance=torch.full_like(c.stair_hold,100.),last=values.clone())
        for k,v in masks.items():buffers['ever'][k]|=v&valid
        buffers['hold'][:]=torch.where(masks['all']&valid,buffers['hold']+c._env.step_dt,0.)
        buffers['max_hold'][:]=torch.maximum(buffers['max_hold'],buffers['hold'])
        buffers['max_actual_hold'][:]=torch.maximum(buffers['max_actual_hold'],c.stair_hold)
        buffers['min_distance'][:]=torch.minimum(buffers['min_distance'],values[:,0])
        buffers['last'][:]=values
        for g in (1,2):
            near=(c.group==g)&valid&masks['near']
            row=frame_summary.setdefault(f'{mode}/{g}',torch.zeros(len(masks)+1,dtype=torch.long,device=c.device))
            row += torch.stack([near.sum()]+[(near&v).sum() for v in masks.values()])
    def reset_audit(env,env_ids):
        c=env.command_manager.get_term('base_velocity');ids=c._ids(env_ids)
        valid=ids[c.stairs[ids]&(tensor(env.episode_length_buf)[ids]>0)]
        if mode!='initial' and buffers:
            masks,values=inspect(c)
            for i in valid.tolist():
                records.append({'mode':mode,'env':i,'group':int(c.group[i]),'level':int(tensor(c.terrain.terrain_levels)[i]),
                                'duration':float(env.episode_length_buf[i])*env.step_dt,
                                'failed':bool(tensor(env.termination_manager.terminated)[i]),
                                'ever':{k:bool(v[i]) for k,v in buffers['ever'].items()},
                                'max_recomputed_hold':float(buffers['max_hold'][i]),
                                'max_actual_hold':float(buffers['max_actual_hold'][i]),
                                'final_actual_hold':float(c.stair_hold[i]),
                                'window_completed':bool(c.stair_completed[i]),
                                'terminal_geometry':bool(masks['geometric'][i]),
                                'min_distance':float(buffers['min_distance'][i]),
                                'final_values':values[i].cpu().tolist(),
                                'goal':c.pos_command_w[i].cpu().tolist(),
                                'origin':tensor(c.terrain.env_origins)[i].cpu().tolist()})
            for v in buffers['ever'].values():v[ids]=False
            for k in ('hold','max_hold','max_actual_hold'):buffers[k][ids]=0
            buffers['min_distance'][ids]=100.
        c.skip_curriculum_once[ids]=False
        return {}
    cfg.curriculum.terrain_levels=CurriculumTermCfg(func=reset_audit)
    StairTargetCommand._update_metrics=update
    with launch_simulation(cfg,args):
        env=gym.make(task,cfg=cfg).unwrapped
        try:
            wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
            runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=env.device)
            for label in args.modes:
                mode='initial';buffers.clear();torch.manual_seed(42)
                runner.load(args.checkpoint,map_location=env.device);runner.alg.eval_mode()
                mode=label;obs=wrapped.get_observations()
                with torch.inference_mode():
                    for step in range(args.steps):
                        action=runner.alg.actor(obs,stochastic_output=label=='sampled')
                        obs,_,done,_=wrapped.step(action);runner.alg.actor.reset(done)
                        if (step+1)%1000==0:print('AUDIT_PROGRESS',label,step+1,flush=True)
            summary={}
            for label in args.modes:
                for group in (1,2):
                    rows=[r for r in records if r['mode']==label and r['group']==group];n=max(1,len(rows))
                    summary[f'{label}/{group}']={'episodes':len(rows),'failed':sum(r['failed'] for r in rows),
                        'ever':{k:sum(r['ever'][k] for r in rows) for k in buffers['ever']},
                        'hold_1s_recomputed':sum(r['max_recomputed_hold']>=1 for r in rows),
                        'hold_1s_actual':sum(r['max_actual_hold']>=1 for r in rows),
                        'final_pass':sum(r['final_actual_hold']>=1 and not r['failed'] for r in rows),
                        'window_final_pass':sum(r['window_completed'] and r['terminal_geometry'] and not r['failed'] for r in rows),
                        'mean_duration':sum(r['duration'] for r in rows)/n}
            args.output.parent.mkdir(parents=True,exist_ok=True)
            frame_summary={k:dict(zip(['near_frames']+list(buffers['ever']),v.cpu().tolist())) for k,v in frame_summary.items()}
            args.output.write_text(json.dumps({'checkpoint':args.checkpoint,'steps_per_mode':args.steps,
                'scope':'Frozen policy and curriculum; 64 environments; strict and rolling-window completion compared on the same trajectories.',
                'training_observations':args.training_observations,
                'value_keys':['distance','speed','angular_speed','root_above_goal','left_ankle_above_goal','right_ankle_above_goal',
                              'local_x','goal_z','root_z','command_x','actual_hold'],
                'summary':summary,'near_frames':frame_summary,'episodes':records},indent=2))
            print('AUDIT_RESULT',json.dumps(summary),flush=True)
        finally:
            StairTargetCommand._update_metrics=original
            env.close()


if __name__=='__main__':main()
