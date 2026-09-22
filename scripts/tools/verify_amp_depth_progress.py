"""Measure first-episode progress before auto-reset with a fixed visual policy.

Run through scripts/run_with_rsl5.sh. Does not train or update terrain levels.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import gymnasium as gym
import torch
from tensordict import TensorDict
from rsl_rl.models import CNNModel
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from isaaclab.managers import CurriculumTermCfg
from isaaclab_tasks.utils import load_cfg_from_registry
import legged_lab.tasks  # noqa: F401


def tensor(x):
    return x if isinstance(x, torch.Tensor) else x.torch


def record_terminal_progress(env, env_ids):
    """Curriculum executes before reset writes: preserve the real terminal pose."""
    probe = getattr(env, '_progress_probe', None)
    if probe is not None:
        robot = env.scene['robot']
        for i in env_ids.tolist():
            if probe['finished'][i]:
                continue
            pos = tensor(robot.data.root_pos_w)[i, :2]
            origin_distance = torch.linalg.vector_norm(pos - env.scene.env_origins[i, :2]).item()
            path = probe['path'][i] + torch.linalg.vector_norm(pos - probe['previous'][i]).item()
            duration = env.episode_length_buf[i].item() * env.step_dt
            n = max(probe['samples'][i], 1)
            probe['records'].append(dict(
                env_id=i,
                terrain_column=(int(env.scene.terrain.terrain_types[i])
                                if env.scene.terrain.cfg.terrain_type == 'generator' else None),
                terrain_level=(int(env.scene.terrain.terrain_levels[i])
                               if env.scene.terrain.cfg.terrain_type == 'generator' else 0),
                duration_s=duration, curriculum_distance_m=origin_distance,
                displacement_from_initial_m=torch.linalg.vector_norm(pos-probe['initial'][i]).item(),
                path_length_m=path, exceeds_4m=origin_distance > 4.,
                terminated=bool(env.reset_terminated[i]), timeout=bool(env.reset_time_outs[i]),
                mean_body_vx_m_s=probe['vx_sum'][i]/n,
                mean_xy_error_m_s=probe['xy_error_sum'][i]/n,
                mean_yaw_error_rad_s=probe['yaw_error_sum'][i]/n,
            ))
            probe['finished'][i] = True
    if env.scene.terrain.cfg.terrain_type == 'generator':
        return torch.mean(env.scene.terrain.terrain_levels.float())
    return torch.tensor(0., device=env.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--task', default='LeggedLab-Isaac-AMP-Depth-G1-v0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--yaw_rate', type=float, default=0.)
    parser.add_argument('--speed', type=float, default=.5)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error('num_envs must be positive')
    task = args.task
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    cfg.sim.physics = copy.deepcopy(cfg.sim.physics.default)
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.max_init_terrain_level = 0
        cfg.scene.terrain.terrain_generator.seed = args.seed
    cfg.curriculum.terrain_levels = CurriculumTermCfg(func=record_terminal_progress)
    cfg.observations.policy.enable_corruption = False
    cfg.events.physics_material = None
    cfg.events.add_base_mass = None
    cfg.events.push_robot = None
    # Replace reference-aligned commands: fixed values are present even in reset history.
    cfg.commands.base_velocity = UniformVelocityCommandCfg(
        asset_name='robot', resampling_time_range=(1000.,1000.),
        rel_standing_envs=0., heading_command=False, rel_heading_envs=0., debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(args.speed,args.speed),
            lin_vel_y=(0.,0.), ang_vel_z=(args.yaw_rate,args.yaw_rate), heading=None),
    )
    if args.device is not None:
        cfg.sim.device=args.device
    agent_cfg=handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(task,'rsl_rl_cfg_entry_point'),
                                          importlib.metadata.version('rsl-rl-lib'))
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    with launch_simulation(cfg,args):
        env=gym.make(task,cfg=cfg).unwrapped
        try:
            obs,_=env.reset(seed=args.seed)
            kwargs=agent_cfg.actor.to_dict(); kwargs.pop('class_name')
            actor=CNNModel(TensorDict(obs,batch_size=[args.num_envs]),agent_cfg.obs_groups,'actor',29,**kwargs)
            actor.load_state_dict(checkpoint['actor_state_dict'],strict=True)
            actor.to(env.device).eval()
            pos=tensor(env.scene['robot'].data.root_pos_w)[:,:2].clone()
            probe=dict(initial=pos.clone(),previous=pos.clone(),records=[],finished=[False]*args.num_envs,
                       path=[0.]*args.num_envs,samples=[0]*args.num_envs,vx_sum=[0.]*args.num_envs,
                       xy_error_sum=[0.]*args.num_envs,yaw_error_sum=[0.]*args.num_envs)
            env._progress_probe=probe
            expected=torch.tensor([args.speed,0.,args.yaw_rate],device=env.device)
            with torch.inference_mode():
                for step in range(env.max_episode_length+2):
                    command=env.command_manager.get_command('base_velocity')
                    torch.testing.assert_close(command,expected.expand_as(command))
                    robot=env.scene['robot']; pos=tensor(robot.data.root_pos_w)[:,:2]
                    velocity=tensor(robot.data.root_lin_vel_b)
                    yaw=tensor(robot.data.root_ang_vel_b)[:,2]
                    delta=torch.linalg.vector_norm(pos-probe['previous'],dim=1).cpu().tolist()
                    vx=velocity[:,0].cpu().tolist()
                    error=torch.linalg.vector_norm(velocity[:,:2]-expected[:2],dim=1).cpu().tolist()
                    yaw_error=(yaw-args.yaw_rate).abs().cpu().tolist()
                    for i in range(args.num_envs):
                        if not probe['finished'][i]:
                            probe['path'][i]+=delta[i];probe['samples'][i]+=1
                            probe['vx_sum'][i]+=vx[i];probe['xy_error_sum'][i]+=error[i]
                            probe['yaw_error_sum'][i]+=yaw_error[i]
                    probe['previous']=pos.clone()
                    actions=actor(TensorDict(obs,batch_size=[args.num_envs]),stochastic_output=False)
                    assert torch.isfinite(actions).all(), 'Nonfinite actions'
                    obs,_,_,_,_=env.step(actions)
                    if all(probe['finished']):
                        break
            records=sorted(probe['records'],key=lambda r:r['env_id'])
            assert len(records)==args.num_envs, 'Incomplete first episodes'
            assert all(r['terrain_level']==0 for r in records)
            summary={key:sum(r[key] for r in records)/len(records) for key in
                     ['exceeds_4m','timeout','terminated','duration_s','curriculum_distance_m',
                      'displacement_from_initial_m','path_length_m','mean_body_vx_m_s',
                      'mean_xy_error_m_s','mean_yaw_error_rad_s']}
            summary['max_curriculum_distance_m']=max(r['curriculum_distance_m'] for r in records)
            output=dict(task=task, terrain_type=cfg.scene.terrain.terrain_type,
                checkpoint=str(args.checkpoint.resolve()),
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                seed=args.seed,num_envs=args.num_envs,speed=args.speed,yaw_rate=args.yaw_rate,
                protocol='First episodes, deterministic mean actions, original reference reset, level 0; '
                         'selected task terrain and visual inputs; observation noise/mass/friction randomization disabled; '
                         'terminal pose captured before automatic reset; no curriculum updates.',
                summary=summary,episodes=records)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(output,indent=2)+'\n')
            print('RESULT',json.dumps(summary),flush=True)
        finally:
            env.close()


if __name__=='__main__':
    main()
