"""Frozen-policy command and AMP diagnosis; never updates policy/normalizers."""
import argparse
import copy
import json
import importlib.metadata
from pathlib import Path
import torch
import gymnasium as gym
from isaaclab.app.sim_launcher import add_launcher_args,launch_simulation
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
import legged_lab.tasks
from legged_lab.tasks.locomotion.amp.mdp.style_state import tensor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--action_noise',type=float,default=0.,help='Gaussian action noise for frozen-policy ablation; no training.')
    add_launcher_args(p);args=p.parse_args()
    task='LeggedLab-Isaac-AMP-Depth-Target-G1-Play-v0'
    cfg=load_cfg_from_registry(task,'env_cfg_entry_point')
    agent=load_cfg_from_registry(task,'rsl_rl_cfg_entry_point')
    agent=handle_deprecated_rsl_rl_cfg(agent,importlib.metadata.version('rsl-rl-lib'))
    cfg.sim.physics=copy.deepcopy(cfg.sim.physics.default)
    cfg.scene.num_envs=60;cfg.seed=42
    gen=cfg.scene.terrain.terrain_generator
    gen.num_rows=1;gen.num_cols=6;gen.difficulty_range=(0.,0.)
    for sub in gen.sub_terrains.values():sub.proportion=1.
    cfg.commands.base_velocity.rel_standing_envs=0.
    with launch_simulation(cfg,args):
        raw=gym.make(task,cfg=cfg).unwrapped
        try:
            env=RslRlVecEnvWrapper(raw,clip_actions=agent.clip_actions)
            agent.device=raw.device
            runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=raw.device)
            runner.load(args.checkpoint,map_location=raw.device)
            policy=runner.get_inference_policy(device=raw.device)
            disc=runner.alg.amp_discriminator
            cmd=raw.command_manager.get_term('base_velocity')
            original=cmd._update_selected
            modes=['forward','left_half','right_half','left_full','right_full','target']
            fixed=torch.tensor([[.6,0,0],[0,0,.5],[0,0,-.5],[0,0,1.],[0,0,-1.],[0,0,0]],device=raw.device)
            # Interleave modes so every terrain column contains every command group.
            group=torch.arange(raw.num_envs,device=raw.device)%6
            def update(ids):
                original(ids)
                selected=cmd._ids(ids)
                selected=selected[group[selected]!=5]
                cmd.vel_command_b[selected]=fixed[group[selected]]
            cmd._update_selected=update
            obs=env.get_observations()
            fields=['command_vx','command_wz','actual_vx','actual_wz','abs_yaw_error','mean_joint_speed','height',
                    'style','disc_agent','disc_demo','disc_constant_demo','stand_penalty_active','zero_forward_turn',
                    'done','timeout','body_gravity_z']
            totals=torch.zeros(6,len(fields),device=raw.device);counts=torch.zeros(6,device=raw.device)
            blocks={'gravity':(0,3),'q':(3,32),'qd':(32,61),'v':(61,64),'w':(64,67)}
            ablations={key:[] for key in blocks};demo_norm={key:[] for key in blocks}
            for step in range(args.steps):
                with torch.inference_mode():
                    commands=cmd.command.clone()
                    action=policy(obs)
                    if args.action_noise:
                        action=action+torch.randn_like(action)*args.action_noise
                    obs,_,dones,_=env.step(action);policy.reset(dones)
                    if step<20:continue
                    data=raw.scene['robot'].data
                    agent_obs=obs['disc'];demo=obs['disc_demo']
                    style,score=disc.predict_style_reward(agent_obs,raw.step_dt)
                    _,demo_score=disc.predict_style_reward(demo,raw.step_dt)
                    _,constant=disc.predict_style_reward(demo[:,-1:].expand_as(demo).contiguous(),raw.step_dt)
                    values=torch.stack((commands[:,0],commands[:,2],tensor(data.root_lin_vel_b)[:,0],
                        tensor(data.root_ang_vel_w)[:,2],(commands[:,2]-tensor(data.root_ang_vel_w)[:,2]).abs(),
                        tensor(data.joint_vel).abs().mean(-1),tensor(data.root_pos_w)[:,2]-raw.scene.env_origins[:,2],
                        style,score,demo_score,constant,(commands[:,:2].norm(dim=-1)<.06).float(),
                        ((commands[:,0]<.06)&(commands[:,2].abs()>.3)).float(),dones.float(),
                        tensor(raw.termination_manager.time_outs).float(),agent_obs[:,-1,2]),-1)
                    for i in range(6):
                        mask=group==i;totals[i]+=values[mask].sum(0);counts[i]+=mask.sum()
                    if step%20==0:
                        for key,(a,b) in blocks.items():
                            hybrid=agent_obs.clone();hybrid[...,a:b]=demo[...,a:b]
                            _,hybrid_score=disc.predict_style_reward(hybrid,raw.step_dt)
                            ablations[key].append(float(hybrid_score.mean()))
                            normalized=disc.normalize_disc_obs(demo)
                            demo_norm[key].append(float(normalized[...,a:b].abs().mean()))
            result={'checkpoint':str(Path(args.checkpoint).resolve()),'steps':args.steps,'num_envs':raw.num_envs,
                    'action_noise_std':args.action_noise,
                    'initialization':'training reference initialization','terrain':'six types, lowest difficulty',
                    'groups':{name:dict(zip(fields,(totals[i]/counts[i]).cpu().tolist())) for i,name in enumerate(modes)},
                    'hybrid_agent_with_demo_block_score':{k:sum(v)/len(v) for k,v in ablations.items()},
                    'demo_normalized_mean_abs':{k:sum(v)/len(v) for k,v in demo_norm.items()},
                    'note':'Hybrid state scores are off-distribution probes, not causal proof. Dones are per-step fractions.'}
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2))
            print('AUDIT',json.dumps(result),flush=True)
        finally:raw.close()


if __name__=='__main__':main()
