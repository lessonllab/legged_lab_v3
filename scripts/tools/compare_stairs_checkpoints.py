"""Matched first-episode diagnostics for frozen long-stair policy checkpoints."""
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
import legged_lab.tasks
from legged_lab.rsl_rl.amp.ppo_amp import PPOAMP
from legged_lab.tasks.locomotion.amp.mdp.stair_play_matrix import configure_stair_control
from legged_lab.tasks.locomotion.amp.mdp.style_state import tensor
from legged_lab.tasks.locomotion.amp.mdp.locomotion_progress import yaw_frame_velocity
from rsl_rl.runners import OnPolicyRunner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num_envs',type=int,default=32)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--stair_speed',type=float,default=.65)
    p.add_argument('--cases',nargs='+',default=['up20','down20','reverse','turn_left','turn_right'],
                   choices=['up20','down20','slowup20','slowdown20','boxes','rough','reverse','turn_left','turn_right'])
    add_launcher_args(p);args=p.parse_args()
    task='LeggedLab-Isaac-AMP-Stairs-Long-G1-Play-v0'
    cfg=load_cfg_from_registry(task,'env_cfg_entry_point')
    configure_stair_control(cfg,.65)
    cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
    cfg.sim.physics=cfg.sim.physics.default.copy()
    cfg.episode_length_s=60.
    agent=handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(task,'rsl_rl_cfg_entry_point'),importlib.metadata.version('rsl-rl-lib'))
    state={'active':False};result={'scope':'Frozen deterministic first episodes; matched seed 42 and reset perturbations; 17 steps, 20 cm, speed cap 0.65 m/s; 60 s stair limit; 8 s body-frame reverse -0.4 m/s and turn +/-0.6 rad/s diagnostics. No optimizer or curriculum updates.','results':[]}
    result['seed']=args.seed;result['stair_speed']=args.stair_speed
    result['scope']='Frozen deterministic matched first episodes; stairs 17 steps, 20 cm, 60 s; slow cases 0.3 m/s; boxes difficulty 3 and rough difficulty 5 at 0.6 m/s for 30 s; reverse/turn 8 s. No policy updates.'
    def capture(env,env_ids):
        if not state['active']:return {}
        ids=torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
        ids=ids[~state['finished'][ids]]
        c=env.command_manager.get_term('base_velocity')
        for i in ids.tolist():
            failed=bool(tensor(env.termination_manager.terminated)[i])
            state['episodes'][i]={'failed':failed,'crossed':bool(c.stairs[i] & c.exit_geometry()[i]) and not failed,
                'arrivals':float(c.metrics['valid_targets_reached'][i]),
                'duration_s':float(env.episode_length_buf[i])*env.step_dt,
                'final_xy':tensor(c.robot.data.root_pos_w)[i,:2].cpu().tolist()}
        state['finished'][ids]=True
        return {}
    cfg.curriculum.terrain_levels=CurriculumTermCfg(func=capture)
    with launch_simulation(cfg,args):
        env=gym.make(task,cfg=cfg).unwrapped
        try:
            wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
            runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=env.device)
            c=env.command_manager.get_term('base_velocity');original=c.compute
            for checkpoint in args.checkpoints:
                saved=torch.load(checkpoint,map_location=env.device,weights_only=False)
                PPOAMP.load(runner.alg,saved,{'actor':True,'critic':True,'optimizer':False},strict=True)
                policy=runner.get_inference_policy(device=env.device)
                for case in args.cases:
                    state['active']=False;c.compute=original
                    settings={'up20':('上楼',7,args.stair_speed,3001),'down20':('下楼',7,args.stair_speed,3001),
                              'slowup20':('上楼',7,.3,3001),'slowdown20':('下楼',7,.3,3001),
                              'boxes':('块状地形',3,.6,1500),'rough':('起伏路面',5,.6,1500)}
                    kind,level,cap,limit=settings.get(case,('平地',7,.65,400))
                    c.select_terrain(kind,level,cap)
                    torch.manual_seed(args.seed);env.reset();policy.reset(torch.ones(env.num_envs,dtype=torch.bool,device=env.device))
                    if case in ('reverse','turn_left','turn_right'):
                        def fixed(dt,case=case):
                            original(dt);c.vel_command_b.zero_();c.is_standing_env[:]=False
                            if case=='reverse':c.vel_command_b[:,0]=-.4
                            else:c.vel_command_b[:,2]=.6 if case=='turn_left' else -.6
                        c.compute=fixed;c.compute(0.)
                    obs=env.observation_manager.compute()
                    state.update(active=True,finished=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device),episodes={})
                    starts=tensor(c.robot.data.root_pos_w)[:,:2].clone()
                    sums=torch.zeros(env.num_envs,4,device=env.device);counts=torch.zeros(env.num_envs,device=env.device)
                    with torch.inference_mode():
                        for step in range(limit):
                            alive=~state['finished'].clone()
                            if step>=50:
                                v=yaw_frame_velocity(c.robot.data)
                                vals=torch.stack((v[:,0],v[:,1],tensor(c.robot.data.root_ang_vel_b)[:,2],(tensor(c.robot.data.root_pos_w)[:,:2]-starts).norm(dim=-1)),dim=-1)
                                sums[alive]+=vals[alive];counts[alive]+=1
                            obs,_,done,_=wrapped.step(policy(obs));policy.reset(done)
                            if (step+1)%500==0:print('PROGRESS',Path(checkpoint).stem,case,step+1,'finished',int(state['finished'].sum()),flush=True)
                            if state['finished'].all():break
                    state['active']=False
                    means=sums/counts.clamp_min(1)[:,None]
                    episodes=[]
                    for i in range(env.num_envs):
                        row=state['episodes'].get(i,{'failed':False,'crossed':False,'arrivals':float(c.metrics['valid_targets_reached'][i]),'duration_s':(step+1)*env.step_dt,'final_xy':tensor(c.robot.data.root_pos_w)[i,:2].cpu().tolist()})
                        row.update(env=i,mean_vx=float(means[i,0]),mean_vy=float(means[i,1]),mean_yaw_rate=float(means[i,2]),mean_drift_m=float(means[i,3]),measurement_steps=int(counts[i]))
                        episodes.append(row)
                    row={'checkpoint':str(checkpoint),'case':case,'speed_cap':cap,'difficulty':level,'trials':env.num_envs,'crossed':sum(e['crossed'] for e in episodes),'failed':sum(e['failed'] for e in episodes),'mean_vx':float(means[:,0].mean()),'mean_yaw_rate':float(means[:,2].mean()),'mean_drift_m':float(means[:,3].mean()),'episodes':episodes,'starts':starts.cpu().tolist()}
                    result['results'].append(row)
                    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
                    print('RESULT',json.dumps({k:v for k,v in row.items() if k not in ('episodes','starts')}),flush=True)
            c.compute=original
        finally:env.close()

if __name__=='__main__':main()
