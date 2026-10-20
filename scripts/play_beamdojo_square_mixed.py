"""Play mixed checkpoints with their archived geometry and route-facing runtime.

No simulation starts on import. UI selections hold difficulty/variant/replica
fixed across episodes; training curriculum is used only for episode accounting.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from queue import SimpleQueue
import sys
import time
import traceback

ANGLES = {'cardinal': (0,90,180,-90), 'diagonal': (45,135,-135,-45)}
SNAPSHOT_MODULES = (
    'beamdojo_feasibility_geometry', 'beamdojo_square_course_geometry',
    'beamdojo_square_diagram_playback', 'beamdojo_square_mixed_geometry',
    'beamdojo_square_course_runtime', 'beamdojo_square_mixed_runtime',
    'beamdojo_square_course_profile', 'beamdojo_square_mixed_profile',
)


def contracts_equal(left, right):
    """Compare strict JSON values, allowing saved tuple/list representation only.

    JSON roundtripping also matches serialized dictionary keys. Canonical text
    comparison preserves boolean/number distinctions and rejects nonfinite data.
    The complete contract is compared; no field or raster digest is omitted.
    """
    def canonical(value):
        normalized = json.loads(json.dumps(value, allow_nan=False))
        return json.dumps(normalized, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return canonical(left) == canonical(right)


def validate_contract(contract):
    mixed = contract.get('square_mixed', {})
    if isinstance(mixed.get('version'),bool) or mixed.get('version') not in (1,2) or mixed.get('phase') not in ('regular','generalize'):
        raise ValueError('Expected a supported square_mixed checkpoint contract')
    if mixed['version']==2 and mixed.get('geometry_revision')!=2:
        raise ValueError('Extended checkpoint must identify geometry revision 2')
    specs = mixed.get('specs', [])
    if len(specs) not in (64,80) or any(len(s) != 3 for s in specs):
        raise ValueError('Checkpoint must record all 64 atlas column seed/family/variant triples')
    if len(specs)==80 and contract.get('flat_rehearsal',{}).get('columns')!=list(range(64,80)):
        raise ValueError('80-column atlas requires recorded flat rehearsal columns')
    for column,(seed, family, variant) in enumerate(specs):
        if not all(isinstance(x,int) and not isinstance(x,bool) for x in (seed,family,variant)):
            raise ValueError('Column specifications must be integer triples')
        variant_limit=3 if mixed['version']==1 else 7
        if (column<64 and (family not in (0,1) or not 0<=variant<=variant_limit)) or (column>=64 and (family!=2 or variant!=0)):
            raise ValueError('Invalid column family/variant')
    digest = mixed.get('support_raster_sha256', '')
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Checkpoint is missing a valid support raster digest')
    return mixed


def available_variants(contract, family):
    mixed = validate_contract(contract)
    if family=='flat':
        if len(mixed['specs'])!=80:raise ValueError('No saved flat terrain')
        return [0]
    family_id = list(ANGLES).index(family)
    return sorted({v for _,f,v in mixed['specs'] if f == family_id})


def selection(contract, family, angle, level, variant=0, replica=0):
    """Resolve controls against recorded specs, never invent a new map seed."""
    mixed = validate_contract(contract)
    if family not in ANGLES or angle not in ANGLES[family]:
        raise ValueError('Route angle must belong to selected family')
    if isinstance(level,bool) or not isinstance(level,int) or not 0 <= level <= 10:
        raise ValueError('Difficulty must be an integer in 0..10')
    if isinstance(variant,bool) or not isinstance(variant,int):
        raise ValueError('Variant must be an integer')
    family_id = list(ANGLES).index(family)
    columns = [i for i,(_,f,v) in enumerate(mixed['specs']) if f==family_id and v==variant]
    if isinstance(replica,bool) or not isinstance(replica,int) or not 0 <= replica < len(columns):
        raise ValueError('Variant/replica is absent from the checkpoint atlas')
    col = columns[replica]
    return dict(family=family, family_id=family_id, route=ANGLES[family].index(angle),
                angle_degrees=angle, level=level, variant=variant, replica=replica,
                column=col, layout_seed=mixed['specs'][col][0])



def play_selection(contract,family,angle,level,variant=0,replica=0):
    if family!='flat':return selection(contract,family,angle,level,variant,replica)
    meta=validate_contract(contract)
    if len(meta['specs'])!=80 or angle not in sum(ANGLES.values(),()) or not 0<=replica<16 or variant!=0:
        raise ValueError('Invalid saved flat selection')
    facing='cardinal' if angle in ANGLES['cardinal'] else 'diagonal'
    return dict(family='flat',family_id=list(ANGLES).index(facing),route=ANGLES[facing].index(angle),angle_degrees=angle,
                level=0,variant=0,replica=replica,column=64+replica,layout_seed=meta['specs'][64+replica][0])


def install_manual_command(term,state):
    """Override inside command-manager updates, including automatic resampling."""
    from types import MethodType
    native=term._update_command
    def update(self):
        native()
        if state['manual']:
            self.vel_command_b[:]=self.vel_command_b.new_tensor(state['command'])
            self._env._square_course_state['parked'][:]=False
    term._update_command=MethodType(update,term)



def refresh_policy_observations(env):
    """Manager.compute returns dict; RSL actor selection requires TensorDict."""
    from tensordict import TensorDict
    return TensorDict(env.observation_manager.compute(),batch_size=[env.num_envs])


def heading_diagnostics(env,chosen):
    """Read actual PhysX headings; camera projection never enters these metrics."""
    from beamdojo_square_mixed_runtime import tensor
    def yaw(q):
        x,y,z,w=(float(v) for v in q)
        return math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    robot=env.scene['robot'];data=robot.data
    q=tensor(data.root_quat_w)[0];x,y,z,w=(float(v) for v in q)
    heading=yaw(q)
    body_quats=tensor(data.body_quat_w)[0]
    bodies={name:yaw(body_quats[robot.body_names.index(name)]) for name in
            ('pelvis','torso_link','left_ankle_roll_link','right_ankle_roll_link') if name in robot.body_names}
    return dict(episode_time_s=float(tensor(env.episode_length_buf)[0])*env.step_dt,
                root_quat_xyzw=q.tolist(),root_yaw_deg=heading,
                route_yaw_error_deg=(heading-chosen['angle_degrees']+180)%360-180,
                root_roll_deg=math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))),
                root_pitch_deg=math.degrees(math.asin(max(-1.,min(1.,2*(w*y-z*x))))),
                body_yaw_deg=bodies,max_joint_speed_rad_s=float(tensor(data.joint_vel)[0].abs().max()))


def load_snapshot_modules(checkpoint):
    """Use saved map/runtime code while keeping live application paths intact."""
    archive = Path(checkpoint).parent/'params'
    contract_path=Path(checkpoint).parent/'contract.json'
    metadata=json.loads(contract_path.read_text()) if contract_path.is_file() else {}
    extras=('beamdojo_square_terminal','beamdojo_flat_rehearsal')
    if metadata.get('square_mixed',{}).get('version')==2:
        extras+=('beamdojo_square_extended_geometry','beamdojo_square_extended_runtime','beamdojo_square_extended_profile')
        if metadata.get('gait_preservation'):
            extras+=('beamdojo_gait_preservation_runtime','beamdojo_gait_preservation_rewards','beamdojo_gait_preservation_profile')
    else:
        extras=tuple(name for name in extras if (archive/(name+'.py')).is_file())
    modules=SNAPSHOT_MODULES+extras
    missing = [name+'.py' for name in modules if not (archive/(name+'.py')).is_file()]
    if missing:
        raise ValueError('Checkpoint source snapshot is incomplete: '+', '.join(missing))
    for name in modules:
        spec = importlib.util.spec_from_file_location(name, archive/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)


def apply_selection(env, chosen, ids=None):
    import torch
    import beamdojo_square_mixed_runtime as runtime
    ids = torch.arange(env.num_envs,device=env.device) if ids is None else ids
    if hasattr(env,'flat_rehearsal_mask'):env.flat_rehearsal_mask[ids]=chosen['family']=='flat'
    env.square_family[ids] = chosen['family_id']
    env.square_route[ids] = chosen['route']
    env.square_variant[ids] = chosen['variant']
    env.beamdojo_levels[ids] = chosen['level']
    runtime.tensor(env.scene.terrain.terrain_types)[ids] = chosen['column']
    runtime.refresh_angles(env)
    env.square_refresh_geometry(env,ids)


def verify_spawn(env, chosen,check_command=True):
    """Verify selected collision tile, query assignment, yaw and body command."""
    import torch
    from beamdojo_square_mixed_runtime import tensor
    terrain = env.scene.terrain
    origins = tensor(env.scene.env_origins)
    expected = tensor(terrain.terrain_origins)[env.beamdojo_levels,tensor(terrain.terrain_types)]
    if not torch.equal(origins,expected) or not bool((tensor(terrain.terrain_levels)==chosen['level']).all()):
        raise ValueError('Selected collision origin/level does not match query geometry')
    if not bool((tensor(terrain.terrain_types)==chosen['column']).all()):
        raise ValueError('Selected replica column changed during reset')
    data = env.scene['robot'].data
    if float((tensor(data.root_pos_w)-origins)[:,:2].abs().max()) > .001:
        raise ValueError('Reset is not centered on the selected platform')
    q = tensor(data.root_quat_w)
    yaw = torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2))
    delta = yaw-math.radians(chosen['angle_degrees'])
    if float(torch.atan2(delta.sin(),delta.cos()).abs().max()) > 1e-5:
        raise ValueError('Reset yaw does not face the selected route')
    command = tensor(env.command_manager.get_command('base_velocity'))
    if check_command and (float(command[:,1:].abs().max()) > 1e-6 or not bool((command[:,0]>0).all())):
        raise ValueError('Expected a positive pure body-forward command after reset')


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--viser_port',type=int,default=8090)
    parser.add_argument('--speed',type=float,default=.75)
    parser.add_argument('--initial_family',choices=list(ANGLES),default='cardinal')
    parser.add_argument('--route_degrees',type=int)
    parser.add_argument('--terrain_level',type=int,choices=range(11),default=0)
    parser.add_argument('--variant',type=int,choices=range(8),default=0)
    parser.add_argument('--replica',type=int,default=0)
    parser.add_argument('--follow_view',choices=('front45','front10','manual'),default='front45')
    parser.add_argument('--interactive_commands',action='store_true')
    parser.add_argument('--start_auto_course',action='store_true',
                        help='Keep manual controls available, but start with automatic course braking.')
    parser.add_argument('--manual_command',type=float,nargs=3,default=[.5,0.,0.],metavar=('VX','VY','WZ'))
    parser.add_argument('--diagnose_spawn',action='store_true')
    parser.add_argument('--verify_manual_commands',action='store_true')
    parser.add_argument('--initial_flat',action='store_true')
    parser.add_argument('--verify_difficulty_switches',action='store_true')
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device='cpu',visualizer=['none'])
    args = parser.parse_args(); args.checkpoint = args.checkpoint.resolve(strict=True)
    parent = json.loads((args.checkpoint.parent/'contract.json').read_text())
    metadata = validate_contract(parent)
    if args.initial_flat and not args.interactive_commands:
        parser.error('Flat playback requires --interactive_commands')
    if args.start_auto_course and (args.initial_flat or not args.interactive_commands):
        parser.error('--start_auto_course requires interactive controls on a stone course')
    if not .5 <= args.speed <= 1.:
        parser.error('Cruise speed must be .5..1 m/s')
    angle = args.route_degrees if args.route_degrees is not None else ANGLES[args.initial_family][0]
    chosen = play_selection(parent,'flat' if args.initial_flat else args.initial_family,angle,args.terrain_level,args.variant,args.replica)
    launcher = AppLauncher(args); wrapped = viewer = None; exit_code = 0
    try:
        import gymnasium as gym
        import torch
        from isaaclab.app.sim_launcher import launch_simulation
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        # Resolve project root from live common.py before loading archived modules.
        from beamdojo_common import PROJECT,TASK,contract
        saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if not contracts_equal(saved.get('infos',{}).get('beamdojo_contract'), parent):
            raise ValueError('Embedded checkpoint contract differs from contract.json')
        del saved
        load_snapshot_modules(args.checkpoint)
        import beamdojo_square_mixed_runtime as runtime
        if parent.get('gait_preservation'):
            from beamdojo_gait_preservation_profile import build,make_contract
            from beamdojo_square_extended_profile import tile
        elif metadata['version']==2:
            from beamdojo_square_extended_profile import build,make_contract,tile
        else:
            from beamdojo_square_mixed_profile import build,make_contract,tile
            from beamdojo_square_terminal import install as install_terminal
            install_terminal(runtime)
        from beamdojo_playback_visuals import active_terrain_bounds,show_perception_samples
        cfg,agent = build(parent,1,metadata['seed'],args.device,metadata['phase'],**({'flat_fraction':parent['flat_rehearsal']['fraction']} if parent.get('flat_rehearsal') else {}))
        if not contracts_equal(cfg.square_mixed_column_specs, metadata['specs']):
            raise ValueError('Archived builder does not reproduce recorded column seeds/variants')
        cfg.observations.policy.enable_corruption = False
        cfg.beamdojo_map_noise = False; cfg.events.push_robot = None
        agent.algorithm.num_mini_batches = 1
        with launch_simulation(cfg,args):
            wrapped = RslRlVecEnvWrapper(gym.make(TASK,cfg=cfg),clip_actions=agent.clip_actions)
            env = wrapped.unwrapped; obs = wrapped.get_observations()
            actual = make_contract(contract(cfg,obs,wrapped.num_actions,None,list(env.scene['robot'].joint_names)),cfg,parent)
            for key in ('action_dimension','joint_names','asset_sha256','observation_dimensions','reward_weights','reward_functions','reward_parameters','map'):
                if not contracts_equal(actual[key], parent[key]):
                    raise ValueError('Playback contract mismatch: '+key)
            if actual['square_mixed']['support_raster_sha256'] != metadata['support_raster_sha256']:
                raise ValueError('Archived geometry does not reproduce checkpoint support raster')
            from beamdojo_support_transition_loader import configure_actor
            settings=configure_actor(agent.to_dict(),parent,args.checkpoint)
            runner = OnPolicyRunner(wrapped,settings,log_dir=None,device=agent.device)
            runner.load(str(args.checkpoint),load_cfg={'actor':True},map_location=agent.device)
            policy = runner.get_inference_policy(device=agent.device)
            # This hook runs after curriculum accounting, before physical reset.
            # Hold the user's selected map across both automatic and manual resets.
            env.square_choose_variants = lambda ids: apply_selection(env,chosen,ids)
            apply_selection(env,chosen)
            obs = wrapped.reset()[0]; verify_spawn(env,chosen,check_command=not args.interactive_commands)
            sys.path.insert(0,str(PROJECT/'scripts/rsl_rl'))
            from physx_viser import PhysxViser
            viewer = PhysxViser(env,host='127.0.0.1',port=args.viser_port,follow=True,
                                terrain_bounds=active_terrain_bounds(env),follow_view=args.follow_view)
            gui = viewer.server.gui
            from beamdojo_play_command_panel import AUTO,CommandPanel
            method_name='支撑转换实验' if parent.get('support_transition') else '混合方桩播放'
            gui.add_markdown(f'**{method_name} · {args.checkpoint.name}**\n\n出生面朝所选路线。场景选择需点击应用；运动指令立即生效。',order=-99)
            viewer.velocity_info.visible=False
            viewer.selected.visible=False
            labels={'cardinal':'正向桩路（前后左右）','diagonal':'斜向桩路（四个斜角）','flat':'纯平地'}
            family_labels={label:f for f,label in labels.items() if f!='flat' or (parent.get('flat_rehearsal') and args.interactive_commands)}
            requests = SimpleQueue()
            def angles_for(f):return sum(ANGLES.values(),()) if f=='flat' else ANGLES[f]
            def replicas_for(f,v):
                if f=='flat':return 16
                return sum(1 for _,fi,vi in metadata['specs'] if fi==list(ANGLES).index(f) and vi==int(v))
            with gui.add_folder('场景选择',order=-98):
                family = gui.add_dropdown('地形',options=list(family_labels),initial_value=labels[chosen['family']])
                def selected_family():return family_labels[family.value]
                route = gui.add_dropdown('路线／出生朝向（度）',options=[str(a) for a in angles_for(chosen['family'])],initial_value=str(chosen['angle_degrees']))
                difficulty = gui.add_dropdown('难度（0–10）',options=[str(i) for i in range(11)],initial_value=str(chosen['level']),visible=chosen['family']!='flat')
                with gui.add_folder('地图细节',expand_by_default=False):
                    variant = gui.add_dropdown('几何变体（0为规则）',options=[str(v) for v in available_variants(parent,chosen['family'])],initial_value=str(chosen['variant']),visible=chosen['family']!='flat')
                    replica = gui.add_dropdown('地图样本',options=[str(i) for i in range(replicas_for(chosen['family'],variant.value))],initial_value=str(chosen['replica']))
                apply = gui.add_button('应用场景并回起点')
                reset = gui.add_button('重播当前场景')
                course_info = gui.add_markdown('')
            def refresh_replicas():
                replica.options = [str(i) for i in range(replicas_for(selected_family(),variant.value))]
                if replica.value not in replica.options: replica.value = replica.options[0]
            @family.on_update
            def family_changed(_):
                f=selected_family()
                route.options = [str(a) for a in angles_for(f)]; route.value = route.options[0]
                variant.options = [str(v) for v in available_variants(parent,f)]
                if variant.value not in variant.options: variant.value = variant.options[0]
                difficulty.visible=variant.visible=f!='flat'
                refresh_replicas()
            @variant.on_update
            def variant_changed(_):
                refresh_replicas()
            apply.on_click(lambda _: requests.put(play_selection(parent,selected_family(),int(route.value),int(difficulty.value),int(variant.value),int(replica.value))))
            reset.on_click(lambda _: requests.put(None))
            panel=CommandPanel(gui,interactive=args.interactive_commands,flat=chosen['family']=='flat',cruise=args.speed)
            for i,value in enumerate(args.manual_command):panel.set_axis(i,value)
            if args.start_auto_course:panel.set_mode(AUTO)
            manual_state=panel.snapshot()
            if args.interactive_commands:
                install_manual_command(env.command_manager.get_term('base_velocity'),manual_state)
            with gui.add_folder('回合状态',order=-94,expand_by_default=False):
                info = gui.add_markdown('等待播放。')
            perception = show_perception_samples(viewer,env)
            counts = dict(episodes=0,successes=0,timeouts=0,failures=0)
            if metadata['version']==2:counts.update(arrivals=0,parkings=0)
            native_compute = env.termination_manager.compute
            def capture():
                done = native_compute()
                if bool(done[0]):
                    win = bool(env.beamdojo_qualified_success[0]); timeout = bool(env.termination_manager.time_outs[0])
                    counts['episodes'] += 1; counts['successes'] += int(win)
                    counts['timeouts'] += int(timeout); counts['failures'] += int(not win and not timeout)
                    if metadata['version']==2:
                        counts['arrivals']+=int(env.beamdojo_square_arrived[0])
                        counts['parkings']+=int(env._square_course_state['parked'][0])
                return done
            env.termination_manager.compute = capture
            status_path = PROJECT/f'.runtime/beamdojo_play_{args.viser_port}.json'
            status_path.parent.mkdir(parents=True,exist_ok=True)
            def publish(tick):
                course = tile(chosen['layout_seed'],2 if chosen['family']=='flat' else chosen['family_id'],chosen['variant'],chosen['level'])[0]
                m = course.get('metadata',{'plane_size_m':[8,8]})
                course_info.content = (f'当前 **{chosen["family"]} · {chosen["angle_degrees"]}° · 难度 {chosen["level"]}/10**\n\n'
                    f'变体 {chosen["variant"]} · 样本 {chosen["replica"]} · 种子 {chosen["layout_seed"]}\n\n'
                    f'桩尺寸范围 {m.get("stone_size_range_m",[.28,.28])} m；采样尝试 {m.get("sample_attempt",1)}；扰动缩放 {m.get("jitter_scale",0)}。')
                if chosen['family']=='flat':
                    size=parent.get('flat_rehearsal',{}).get('plane_size_m',[8,8])[0]
                    course_info.content=f'当前 **{size:g}×{size:g} m 纯平地** · 出生朝向 {chosen["angle_degrees"]}° · 样本 {chosen["replica"]}\n\n平地不统计过桩成功；到地块边缘或{cfg.episode_length_s:g}秒后重置。'
                measurements = runtime.read_measurements(env)
                root = runtime.route_frame(env,measurements['root'])[0]
                parked = bool(env._square_course_state['parked'][0]); hold = float(env.beamdojo_course_stand_time[0])
                info.content = f'沿路线 {float(root[0]):.2f} m · 横偏 {float(root[1]):.2f} m\n\n停车 {parked} · 站稳 {hold:.2f}s\n\n回合 {counts["episodes"]} · 成功 {counts["successes"]} · 超时 {counts["timeouts"]} · 失败 {counts["failures"]}'
                if metadata['version']==2:
                    info.content+=f'\n\n到达 {counts["arrivals"]} · 停车 {counts["parkings"]} · 稳停 {counts["successes"]}（升降级只看是否到达）'
                headings=heading_diagnostics(env,chosen)
                info.content+=f'\n\n实际朝向 {headings["root_yaw_deg"]:.2f}° · 相对出生方向偏航 {headings["route_yaw_error_deg"]:+.2f}°'
                actual_velocity=runtime.tensor(env.scene['robot'].data.root_lin_vel_b)[0]
                actual_omega=runtime.tensor(env.scene['robot'].data.root_ang_vel_b)[0]
                effective=runtime.tensor(env.command_manager.get_command('base_velocity'))[0]
                panel.report(effective.tolist(),actual_velocity.tolist(),actual_omega.tolist())
                value = dict(actual_body_velocity=actual_velocity.tolist(),actual_body_omega=actual_omega.tolist(),state='paused' if viewer.pause.value else 'playing',pid=os.getpid(),checkpoint=str(args.checkpoint),url=f'http://127.0.0.1:{args.viser_port}/',
                    play_domain='physical',terrain_family='square_mixed',terrain_level=chosen['level'],route_degrees=chosen['angle_degrees'],
                    selection=chosen,geometry_metadata=m,terminal_revision=parent['square_course'].get('terminal',{}).get('revision','full_support_then_zero_v2'),checkpoint_phase=metadata['phase'],support_raster_sha256=metadata['support_raster_sha256'],
                    root_route_xy_m=root[:2].tolist(),full_ring=bool(env._square_course_state['full_ring'][0]),steps=tick,device=agent.device,command_body_xyz=runtime.tensor(env.command_manager.get_command('base_velocity'))[0].tolist(),
                    physics_terrain_level=int(runtime.tensor(env.scene.terrain.terrain_levels)[0]),physics_terrain_column=int(runtime.tensor(env.scene.terrain.terrain_types)[0]),
                    command_control='manual' if manual_state['manual'] else 'course',requested_command_body=manual_state['command'],cruise_speed_m_s=manual_state['cruise'],paused=bool(viewer.pause.value),follow_camera_view=viewer.follow_view.value,command_panel_revision='single_source_v1',parked=parked,standing_seconds=hold,**headings,**counts)
                tmp = status_path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n'); tmp.replace(status_path)
            if args.verify_difficulty_switches:
                original = chosen.copy()
                for f in ANGLES:
                    for v in available_variants(parent,f):
                        for level in (0,5,10):
                            for a in ANGLES[f]:
                                chosen = selection(parent,f,a,level,v,0)
                                apply_selection(env,chosen); wrapped.reset(); verify_spawn(env,chosen,check_command=not args.interactive_commands)
                chosen = original; apply_selection(env,chosen); obs = wrapped.reset()[0]
                verify_spawn(env,chosen,check_command=not args.interactive_commands); counts.update({k:0 for k in counts})
                print('MIXED_PLAYBACK_SWITCHES_VERIFIED',flush=True)
            if args.verify_manual_commands:
                if not args.interactive_commands:raise ValueError('Manual verification requires interactive_commands')
                if chosen['family']=='flat':
                    from beamdojo_play_command_panel import AUTO
                    panel.set_terrain(False);panel.set_mode(AUTO);panel.set_terrain(True)
                    if panel.snapshot()['command']!=[0.,0.,0.] or not panel.snapshot()['manual']:
                        raise ValueError('Automatic-to-flat transition must select manual stand')
                    print('COMMAND_PANEL_AUTO_TO_FLAT_VERIFIED',flush=True)
                command=env.command_manager.get_term('base_velocity')
                with torch.inference_mode():
                    for preset in ('前进','后退','左原地转向','右原地转向','站立'):
                        panel.set_preset(preset);manual_state.update(panel.snapshot());values=manual_state['command'];command._update_command()
                        obs=refresh_policy_observations(env)
                        for _ in range(10):
                            actions=policy(obs.select(*agent.obs_groups['actor']))
                            if not torch.isfinite(actions).all():raise ValueError('Nonfinite manual verification action')
                            obs,_,done,_=wrapped.step(actions);policy.reset(done)
                            if not torch.allclose(runtime.tensor(command.command)[0],torch.tensor(values,device=env.device)):raise ValueError('Manual command overwritten during step')
                    for i,value in enumerate(args.manual_command):panel.set_axis(i,value)
                    manual_state.update(panel.snapshot());obs=wrapped.reset()[0]
                    command._update_command();obs=refresh_policy_observations(env)
                verify_spawn(env,chosen,check_command=False)
                print('MANUAL_COMMAND_SWITCHES_VERIFIED forward/backward/left-turn/right-turn/stand',flush=True)
                counts.update({k:0 for k in counts})
            command = env.command_manager.get_term('base_velocity')
            command._square_cruise.fill_(manual_state['cruise']);command._update_command()
            obs=refresh_policy_observations(env)
            if args.diagnose_spawn:
                rows=[]
                obs=wrapped.reset()[0];verify_spawn(env,chosen,check_command=not args.interactive_commands)
                policy.reset(torch.ones(env.num_envs,device=env.device,dtype=torch.bool))
                command._update_command();obs=refresh_policy_observations(env)
                def sample(step):
                    return dict(step=step,**heading_diagnostics(env,chosen),
                                root_route_xy_m=runtime.route_frame(env,runtime.tensor(env.scene['robot'].data.root_pos_w)-runtime.tensor(env.scene.env_origins))[0,:2].tolist())
                rows.append(sample(0))
                with torch.inference_mode():
                    for step in range(1,101):
                        obs,_,done,_=wrapped.step(policy(obs.select(*agent.obs_groups['actor'])))
                        policy.reset(done);rows.append(sample(step))
                        if bool(done.any()):break
                diagnostic=dict(checkpoint=str(args.checkpoint),selection=chosen,command=manual_state['command'],samples=rows)
                (status_path.parent/f'beamdojo_spawn_diagnostic_{args.viser_port}.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2)+'\n')
                print('SPAWN_HEADING_DIAGNOSTIC '+json.dumps([rows[i] for i in (0,5,10,25,50,100) if i<len(rows)]),flush=True)
                obs=wrapped.reset()[0];verify_spawn(env,chosen,check_command=not args.interactive_commands)
                policy.reset(torch.ones(env.num_envs,device=env.device,dtype=torch.bool))
                command._update_command();obs=refresh_policy_observations(env)
                counts.update({k:0 for k in counts})
            viewer.set_terrain_bounds(active_terrain_bounds(env)); viewer.update(obs); perception(obs); publish(0)
            tick = 0
            last_pause=bool(viewer.pause.value)
            with torch.inference_mode():
                while launcher.app.is_running():
                    start = time.perf_counter()
                    if not requests.empty():
                        while not requests.empty():
                            requested = requests.get()
                            if requested is not None: chosen = requested
                        panel.set_terrain(chosen['family']=='flat');manual_state.update(panel.snapshot())
                        apply_selection(env,chosen); obs = wrapped.reset()[0]; verify_spawn(env,chosen,check_command=not args.interactive_commands)
                        policy.reset(torch.ones(env.num_envs,device=env.device,dtype=torch.bool))
                        viewer.set_terrain_bounds(active_terrain_bounds(env)); viewer.update(obs); perception(obs,force=True)
                        for client in viewer.server.get_clients().values(): viewer._focus(client)
                        counts.update({k:0 for k in counts}); publish(tick)
                    command = env.command_manager.get_term('base_velocity')
                    prior_effective=runtime.tensor(command.command).clone()
                    previous=manual_state.copy();manual_state.update(panel.snapshot())
                    command._square_cruise.fill_(manual_state['cruise']);command._update_command()
                    changed=previous!=manual_state or not torch.equal(prior_effective,runtime.tensor(command.command))
                    if changed:obs=refresh_policy_observations(env)
                    paused=bool(viewer.pause.value)
                    if changed or paused!=last_pause:
                        viewer.update(obs);publish(tick)
                    last_pause=paused
                    if viewer.pause.value:
                        time.sleep(.02); continue
                    obs,_,done,_ = wrapped.step(policy(obs.select(*agent.obs_groups['actor'])))
                    policy.reset(done); tick += 1
                    if tick%2 == 0: viewer.update(obs); perception(obs)
                    if tick%10 == 0: publish(tick)
                    time.sleep(max(0.,env.step_dt-(time.perf_counter()-start)))
    except BaseException:
        exit_code = 1; traceback.print_exc(); raise
    finally:
        if viewer is not None: viewer.close()
        if wrapped is not None: wrapped.close()
        launcher.app.close(exit_code=exit_code)


if __name__ == '__main__':
    main()
