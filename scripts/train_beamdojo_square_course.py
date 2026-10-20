"""Isaac Sim 6: +X alternating stones from central platform to square goal ring."""
import argparse
from datetime import datetime
import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import traceback


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def parse_args():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--num_envs',type=int,default=4096)
    p.add_argument('--updates',type=int,default=1000)
    p.add_argument('--terrain_level',type=int,choices=range(11),default=0)
    p.add_argument('--run_name',default='square_straight1000_4096')
    p.add_argument('--status_file',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--prepare_only',action='store_true')
    p.add_argument('--probe_steps',type=int,default=0)
    AppLauncher.add_app_launcher_args(p)
    a=p.parse_args();a.checkpoint=a.checkpoint.resolve(strict=True)
    if a.num_envs<=0 or Path(a.run_name).name!=a.run_name:p.error('Invalid env count/run name')
    if a.updates<0 or a.updates%1000 or (a.updates==0 and not a.prepare_only):p.error('PPO budget must be multiples of 1000')
    if a.probe_steps<0 or (a.probe_steps and not a.prepare_only):p.error('Probe requires prepare-only')
    return a


def run(args,cfg,agent,parent):
    import gymnasium as gym
    import torch
    from isaaclab.utils.io import dump_yaml
    from isaaclab.utils.seed import configure_seed
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from beamdojo_common import PROJECT,TASK,contract
    from beamdojo_square_course_profile import make_contract
    import beamdojo_square_course_runtime as runtime
    configure_seed(args.seed)
    directory=PROJECT/'logs/rsl_rl'/agent.experiment_name/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')+'_'+args.run_name)
    directory.mkdir(parents=True)
    state=dict(state='initializing',pid=os.getpid(),run_dir=str(directory),num_envs=args.num_envs,
        parent_checkpoint=str(args.checkpoint),parent_iteration=None,requested_updates=0 if args.prepare_only else args.updates,
        terrain_level=args.terrain_level,runtime='Isaac Sim 6.0.1 / uv Python / RSL 5.4.1',
        started_at=datetime.now().astimezone().isoformat(),learned_crossing_verified=False)
    def status(**values):
        state.update(values,updated_at=datetime.now().astimezone().isoformat())
        write_json(directory/'status.json',state)
        if args.status_file:write_json(args.status_file,state)
    status();env=None
    try:
        dump_yaml(str(directory/'params/env.yaml'),cfg)
        dump_yaml(str(directory/'params/agent.yaml'),agent)
        write_json(directory/'params/source_contract.json',parent)
        for name in ('train_beamdojo_square_course.py','beamdojo_square_course_profile.py',
            'beamdojo_square_course_geometry.py','beamdojo_square_course_runtime.py',
            'beamdojo_feasibility_geometry.py','beamdojo_common.py','run_with_unitree_official.sh','run_with_rsl5.sh'):
            shutil.copy2(PROJECT/'scripts'/name,directory/'params'/name)
        for subtree in ('tasks/locomotion/beamdojo','tasks/locomotion/unitree_official','rsl_rl/beamdojo'):
            shutil.copytree(PROJECT/'source/legged_lab/legged_lab'/subtree,directory/'params'/subtree,
                ignore=shutil.ignore_patterns('__pycache__'))
        env=RslRlVecEnvWrapper(gym.make(TASK,cfg=cfg),clip_actions=agent.clip_actions)
        observations=env.get_observations()
        base=contract(cfg,observations,env.num_actions,None,list(env.unwrapped.scene['robot'].joint_names))
        run_contract=make_contract(base,cfg,parent)
        runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(directory),device=agent.device)
        runner.alg.set_numerical_guard(directory/'numerical_incident.json')
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if saved.get('infos',{}).get('beamdojo_contract')!=parent:raise ValueError('Embedded source contract mismatch')
        runner.load(str(args.checkpoint),load_cfg={'actor':True},map_location=agent.device)
        if runner.alg.optimizer.state:raise ValueError('Actor-only transfer restored optimizer')
        for k,v in saved['actor_state_dict'].items():
            if not torch.equal(runner.alg.actor.state_dict()[k].detach().cpu(),v):raise ValueError(f'Actor migration mismatch {k}')
        algorithm=saved['beamdojo']
        for name,value in algorithm['parameters'].items():setattr(runner.alg,name,value)
        runner.alg.advantage_weights=tuple(algorithm['advantage_weights'])
        runner.alg.learning_rate=algorithm['learning_rate']
        for group in runner.alg.optimizer.param_groups:group['lr']=runner.alg.learning_rate
        runner.current_learning_iteration=0
        observations=env.reset()[0]
        raw=env.unwrapped;tensor=runtime.tensor
        origins=tensor(raw.scene.env_origins)
        expected=tensor(raw.scene.terrain.terrain_origins)[raw.beamdojo_levels,tensor(raw.scene.terrain.terrain_types)]
        if not torch.equal(origins,expected):raise ValueError('Collision/query origins mismatch')
        if not torch.equal(tensor(raw.scene.terrain.terrain_levels),raw.beamdojo_levels):raise ValueError('Collision/query levels mismatch')
        local=tensor(raw.scene['robot'].data.root_pos_w)-origins
        if local[:,:2].abs().max()>.001:raise ValueError('Spawn not centered on platform')
        query=raw._square_course_surface_height
        points=origins[:,None,:].repeat(1,4,1)
        points[:,:,:2]+=torch.tensor([[0.,0.],[1.,0.],[3.5,0.],[4.1,0.]],device=raw.device)
        heights,supported=query(points)
        if not torch.equal(supported,torch.tensor([True,False,True,False],device=raw.device).expand_as(supported)):
            raise ValueError('Central/gap/ring/outside query failed')
        runner.alg.eval_mode()
        with torch.inference_mode():
            mean=runner.alg.actor(observations);sample=runner.alg.actor(observations,stochastic_output=True)
        if not torch.isfinite(mean).all() or not torch.isfinite(sample).all() or sample.abs().max()>100:
            raise ValueError('Initial action guard failed')
        run_contract['training_budget']=dict(added_updates=0 if args.prepare_only else args.updates,chunk_updates=1000)
        run_contract['square_course']['optimizer_initial_learning_rate']=runner.alg.learning_rate
        run_contract['square_course']['source_checkpoint']=str(args.checkpoint)
        run_contract['square_course']['source_checkpoint_sha256']=source_sha256
        run_contract['square_course']['source_iteration']=int(saved['iter'])
        run_contract['square_course']['actual_ppo_parameters']=algorithm['parameters']
        run_contract['square_course']['actual_advantage_weights']=list(runner.alg.advantage_weights)
        run_contract['action_mean_tail_bounds']=runner.alg.action_mean_tail_bounds_report()
        run_contract['action_exploration_scale']=runner.alg.action_exploration_scale_report()
        write_json(directory/'contract.json',run_contract)
        status(parent_iteration=int(saved['iter']),initialization='actor_normalization_only',critic_reset=True,
            optimizer_reset=True,completed_updates=0,geometry_validation=dict(collision_query_origin_match=True,
            collision_query_level_match=True,spawn_centered=True,query_truth_checked=True),
            initial_actions=dict(mean_max=float(mean.abs().max()),sample_max=float(sample.abs().max()),finite=True))
        native_save=runner.save
        def save(path,infos=None):
            completed=runner.current_learning_iteration+1
            native_save(path,dict(beamdojo_contract=run_contract,beamdojo_square_course=dict(completed_updates=completed),previous_infos=infos))
            status(last_checkpoint=str(path),completed_updates=completed)
        runner.save=save
        native_log=runner.logger.log
        def log(**kwargs):
            native_log(**kwargs)
            losses={k:float(v) for k,v in kwargs['loss_dict'].items()}
            if not all(math.isfinite(v) for v in losses.values()):raise ValueError('Nonfinite PPO statistics')
            reward=list(runner.logger.rewbuffer)
            status(current_iteration=kwargs['it'],completed_updates=kwargs['it']+1,
                losses=losses,mean_episode_reward=float(sum(reward)/len(reward)) if reward else None,
                collection_time_s=float(kwargs['collect_time']),learning_time_s=float(kwargs['learn_time']))
        runner.logger.log=log
        if args.prepare_only:
            with torch.inference_mode():
                for _ in range(args.probe_steps):observations,_,_,_=env.step(runner.alg.actor(observations))
            status(state='prepared',probe_steps=args.probe_steps,policy_updates=0,
                course_metrics={k:float(v) for k,v in runtime.course_metrics(raw).items()})
            return
        status(state='training')
        for _ in range(args.updates//1000):
            runner.learn(1000,init_at_random_ep_len=False)
            runner.save(str(directory/f'model_{runner.current_learning_iteration}.pt'))
            runner.current_learning_iteration+=1
        status(state='budget_complete',completed_updates=args.updates)
    except BaseException as error:
        status(state='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if env is not None:env.close()


def main():
    args=parse_args()
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(args)
    exit_code=0
    try:
        from isaaclab.app.sim_launcher import launch_simulation
        from beamdojo_square_course_profile import build
        parent=json.loads((args.checkpoint.parent/'contract.json').read_text())
        cfg,agent=build(parent,args.num_envs,args.seed,args.device or 'cuda:0',args.terrain_level)
        agent.max_iterations=args.updates;agent.save_interval=100
        with launch_simulation(cfg,args):run(args,cfg,agent,parent)
    except BaseException:
        exit_code=1
        traceback.print_exc()
        raise
    finally:launcher.app.close(exit_code=exit_code)


if __name__=='__main__':main()
