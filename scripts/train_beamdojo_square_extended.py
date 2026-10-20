"""Extended mixed flat/stone continuation, full PPO resume in 1000-update blocks."""
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


def active_variant(phase,completed_phase_updates):
    if phase=='regular':return 0
    return 1 if completed_phase_updates<1000 else (3 if completed_phase_updates<3000 else (5 if completed_phase_updates<5000 else 7))


def parse_args():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--num_envs',type=int,default=4096)
    p.add_argument('--updates',type=int,default=10000)
    p.add_argument('--terrain_level',type=int,choices=range(11),default=0)
    p.add_argument('--phase',choices=['regular','generalize'],default='generalize')
    p.add_argument('--run_name',default='square_extended_flat50_10000_4096')
    p.add_argument('--status_file',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--prepare_only',action='store_true')
    p.add_argument('--flat_fraction',type=float,default=.5)
    p.add_argument('--probe_steps',type=int,default=0)
    p.add_argument('--startup_probe_steps',type=int,default=0)
    AppLauncher.add_app_launcher_args(p)
    a=p.parse_args();a.checkpoint=a.checkpoint.resolve(strict=True)
    if not 0<=a.flat_fraction<1:p.error('flat_fraction must be in [0,1)')
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
    from beamdojo_square_extended_profile import make_contract
    import beamdojo_square_mixed_runtime as runtime
    configure_seed(args.seed)
    directory=PROJECT/'logs/rsl_rl'/agent.experiment_name/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')+'_'+args.run_name)
    directory.mkdir(parents=True)
    state=dict(state='initializing',pid=os.getpid(),run_dir=str(directory),num_envs=args.num_envs,
        parent_checkpoint=str(args.checkpoint),parent_iteration=None,requested_updates=0 if args.prepare_only else args.updates,
        terrain_level=args.terrain_level,phase=args.phase,flat_fraction=args.flat_fraction,runtime='Isaac Sim 6.0.1 / uv Python / RSL 5.4.1',
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
            'beamdojo_feasibility_geometry.py','beamdojo_common.py','run_with_unitree_official.sh','run_with_rsl5.sh',
            'train_beamdojo_square_mixed.py','beamdojo_square_mixed_profile.py','beamdojo_square_mixed_geometry.py',
            'beamdojo_square_mixed_runtime.py','beamdojo_square_diagram_playback.py','beamdojo_square_terminal.py','beamdojo_flat_rehearsal.py',
            'train_beamdojo_square_extended.py','beamdojo_square_extended_profile.py','beamdojo_square_extended_geometry.py','beamdojo_square_extended_runtime.py'):
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
        embedded=saved.get('infos',{}).get('beamdojo_contract')
        if json.loads(json.dumps(embedded,allow_nan=False))!=parent:raise ValueError('Embedded source contract mismatch')
        runner.load(str(args.checkpoint),map_location=agent.device)
        for k,v in saved['actor_state_dict'].items():
            if not torch.equal(runner.alg.actor.state_dict()[k].detach().cpu(),v):raise ValueError(f'Actor resume mismatch {k}')
        algorithm=saved['beamdojo']
        runner.current_learning_iteration=int(saved['iter'])+1
        source_info=saved.get('infos',{})
        same_geometry=(parent.get('square_mixed',{}).get('version')==2
            and parent.get('square_mixed',{}).get('geometry_revision')==2
            and parent.get('square_mixed',{}).get('phase')==args.phase
            and parent.get('flat_rehearsal',{}).get('fraction',0.)==args.flat_fraction
            and parent.get('square_mixed',{}).get('specs')==[list(s) for s in cfg.square_mixed_column_specs])
        prior_phase_updates=int(source_info.get('square_extended_phase_completed',0)) if same_geometry else 0
        if prior_phase_updates<0:raise ValueError('Invalid saved extended phase progress')
        cfg.square_mixed_active_variant=active_variant(args.phase,prior_phase_updates)
        raw=env.unwrapped
        curriculum_restore='new geometry/group ownership: levels and counters initialized from CLI; policy/critics/Adam fully restored'
        if same_geometry and source_info.get('square_mixed_curriculum') is not None:
            if source_info['square_mixed_curriculum']['num_envs']==raw.num_envs:
                runtime.restore(raw,source_info['square_mixed_curriculum'])
                curriculum_restore='exact extended curriculum resume'
            elif not args.prepare_only:raise ValueError('Training resume environment count differs from saved curriculum')
        if args.flat_fraction:
            raw.beamdojo_levels[raw.flat_rehearsal_mask]=0
            raw.square_win_streak[raw.flat_rehearsal_mask]=0;raw.square_fail_streak[raw.flat_rehearsal_mask]=0
            raw.square_choose_variants(torch.arange(raw.num_envs,device=raw.device));raw.square_refresh_geometry(raw,None)
        rng=source_info.get('square_mixed_rng')
        if rng:
            torch.set_rng_state(rng['cpu'])
            if torch.cuda.is_available() and rng.get('cuda'):torch.cuda.set_rng_state_all(rng['cuda'])
        first_iteration=runner.current_learning_iteration
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
        points[:,:,:2]+=torch.tensor([[0.,0.],[1.,0.],[5.,0.],[6.1,0.]],device=raw.device)
        heights,supported=query(points)
        if not bool(supported[:,[0,2]].all()) or bool(supported[:,3].any()):
            raise ValueError('Central/ring/outside query failed')
        q=tensor(raw.scene['robot'].data.root_quat_w)
        yaw=torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2))
        yaw_error=torch.atan2(torch.sin(yaw-raw.square_angles),torch.cos(yaw-raw.square_angles))
        if yaw_error.abs().max()>1e-5:raise ValueError('Mixed spawn yaw mismatch')
        command=raw.command_manager.get_command('base_velocity')
        stone=~raw.flat_rehearsal_mask
        if tensor(command)[stone,1:].abs().max()>1e-6:raise ValueError('Expected pure body-forward stone commands')
        flat_count=int(raw.flat_rehearsal_mask.sum());flat_modes=torch.bincount(raw.flat_rehearsal_mode[raw.flat_rehearsal_mask],minlength=8).tolist()
        if flat_count!=int(args.num_envs*args.flat_fraction):raise ValueError('Flat assignment count mismatch')
        families=torch.bincount(raw.square_family[stone],minlength=2).tolist()
        directions=torch.bincount((raw.square_family*4+raw.square_route)[stone],minlength=8).tolist()
        print(f'MIXED_NATIVE_VERIFIED family_counts={families} direction_counts={directions} max_spawn_yaw_error={float(yaw_error.abs().max())}',flush=True)
        runner.alg.eval_mode()
        with torch.inference_mode():
            mean=runner.alg.actor(observations);sample=runner.alg.actor(observations,stochastic_output=True)
        if not torch.isfinite(mean).all() or not torch.isfinite(sample).all() or sample.abs().max()>100:
            raise ValueError('Initial action guard failed')
        run_contract['training_budget']=dict(added_updates=0 if args.prepare_only else args.updates,chunk_updates=1000)
        run_contract['square_mixed']['generalization_exposure']=dict(
            source_phase_progress_preserved=same_geometry,
            phase_update_intervals=[0,1000,3000,5000],variant_maxima=[1,3,5,7],regular_replay_fraction=.25)
        run_contract['square_course']['optimizer_initial_learning_rate']=runner.alg.learning_rate
        run_contract['square_course']['source_checkpoint']=str(args.checkpoint)
        run_contract['square_course']['source_checkpoint_sha256']=source_sha256
        run_contract['square_course']['source_iteration']=int(saved['iter'])
        run_contract['square_course']['actual_ppo_parameters']=algorithm['parameters']
        run_contract['square_course']['actual_advantage_weights']=list(runner.alg.advantage_weights)
        run_contract['action_mean_tail_bounds']=runner.alg.action_mean_tail_bounds_report()
        run_contract['action_exploration_scale']=runner.alg.action_exploration_scale_report()
        # Torch preserves tuples, while contract.json represents arrays as lists.
        # Persist one JSON-compatible representation for both forms of metadata.
        run_contract=json.loads(json.dumps(run_contract,allow_nan=False))
        write_json(directory/'contract.json',run_contract)
        status(curriculum_restore=curriculum_restore,flat_environment_count=flat_count,flat_mode_counts=flat_modes,parent_iteration=int(saved['iter']),initialization='full_training_resume',critic_reset=False,
            optimizer_reset=False,completed_updates=0,prior_phase_updates=prior_phase_updates,phase_completed_updates=prior_phase_updates,family_counts=families,direction_counts=directions,geometry_validation=dict(spawn_yaw_matches_route=True,body_forward_command=True,collision_query_origin_match=True,
            collision_query_level_match=True,spawn_centered=True,query_truth_checked=True),
            initial_actions=dict(mean_max=float(mean.abs().max()),sample_max=float(sample.abs().max()),finite=True))
        native_save=runner.save
        def save(path,infos=None):
            completed=runner.current_learning_iteration-first_iteration+1
            rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
            native_save(path,dict(beamdojo_contract=run_contract,square_mixed_curriculum=runtime.snapshot(raw),square_mixed_rng=rng,square_extended_phase_completed=prior_phase_updates+completed,square_mixed_phase_completed=prior_phase_updates+completed,completed_updates=completed,previous_infos=infos))
            status(last_checkpoint=str(path),completed_updates=completed,phase_completed_updates=prior_phase_updates+completed)
        runner.save=save
        native_log=runner.logger.log
        def log(**kwargs):
            native_log(**kwargs)
            losses={k:float(v) for k,v in kwargs['loss_dict'].items()}
            if not all(math.isfinite(v) for v in losses.values()):raise ValueError('Nonfinite PPO statistics')
            reward=list(runner.logger.rewbuffer)
            status(current_iteration=kwargs['it'],completed_updates=kwargs['it']-first_iteration+1,
                phase_completed_updates=prior_phase_updates+kwargs['it']-first_iteration+1,
                curriculum_summary=dict(mean_level=float(raw.beamdojo_levels[stone].float().mean()),min_level=int(raw.beamdojo_levels[stone].min()),max_level=int(raw.beamdojo_levels[stone].max()),
                    level_counts=torch.bincount(raw.beamdojo_levels[stone],minlength=11).cpu().tolist(),
                    episodes=raw._square_mixed_state['episodes'].cpu().tolist(),successes=raw._square_mixed_state['successes'].cpu().tolist(),
                    failures=raw._square_mixed_state['failures'].cpu().tolist(),timeouts=raw._square_mixed_state['timeouts'].cpu().tolist()),
                terminal_summary={k:float(v.sum()) for k,v in raw._square_extended_counts.items()},
                losses=losses,mean_episode_reward=float(sum(reward)/len(reward)) if reward else None,
                collection_time_s=float(kwargs['collect_time']),learning_time_s=float(kwargs['learn_time']))
        runner.logger.log=log
        if args.prepare_only:
            with torch.inference_mode():
                for _ in range(args.probe_steps):observations,_,_,_=env.step(runner.alg.actor(observations))
            status(state='prepared',probe_steps=args.probe_steps,policy_updates=0,
                course_metrics={k:float(v) for k,v in runtime.course_metrics(raw).items()})
            return
        if args.startup_probe_steps:
            with torch.inference_mode():
                for _ in range(args.startup_probe_steps):observations,_,_,_=env.step(runner.alg.actor(observations))
            if not all(torch.isfinite(value).all() for value in observations.values()):raise ValueError('Nonfinite startup observations')
            status(startup_probe_steps=args.startup_probe_steps,flat_mode_counts=flat_modes,startup_metrics={k:float(v) for k,v in runtime.course_metrics(raw).items()})
            env.reset()
        status(state='training')
        for chunk in range(args.updates//1000):
            cfg.square_mixed_active_variant=active_variant(args.phase,prior_phase_updates+chunk*1000)
            status(chunk=chunk+1,active_variant_max=cfg.square_mixed_active_variant)
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
        from beamdojo_square_extended_profile import build
        parent=json.loads((args.checkpoint.parent/'contract.json').read_text())
        cfg,agent=build(parent,args.num_envs,args.seed,args.device or 'cuda:0',args.phase,args.terrain_level,flat_fraction=args.flat_fraction)
        agent.max_iterations=args.updates;agent.save_interval=100
        with launch_simulation(cfg,args):run(args,cfg,agent,parent)
    except BaseException:
        exit_code=1
        traceback.print_exc()
        raise
    finally:launcher.app.close(exit_code=exit_code)


if __name__=='__main__':main()
