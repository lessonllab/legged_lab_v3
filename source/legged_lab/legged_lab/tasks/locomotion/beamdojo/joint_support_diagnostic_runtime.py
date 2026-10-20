"""Neutral one-shot swing planner observer; never changes action, reward or RNG."""
import torch


def tensor(value): return value if isinstance(value,torch.Tensor) else value.torch


def state(env):
    if hasattr(env,'_joint_support_observer'): return env._joint_support_observer
    n=env.num_envs; device=env.device
    z=lambda *shape:torch.zeros(*shape,device=device)
    s=dict(off=z(n,2).long(),on=z(n,2).long(),armed=z(n,2).bool(),active=z(n).bool(),valid=z(n).bool(),
        episode_id=z(n).long(),swing_id=z(n).long(),swing_foot=z(n).long(),plan_start_time=z(n),plan_valid=z(n).bool(),trajectory_valid_until=z(n),
        touchdown_deadline=z(n),foot_target=z(n,3),phase_times=z(n,5),plan_mass=z(n),com_path=z(n,5,3),
        velocity_path=z(n,5,3),momentum_path=z(n,5,3),last_tick=-1,
        opportunities=0,selected_opportunities=0,planned=0,missed_no_old_support=0,selected_no_old_support=0,record_cap_opportunities=0)
    env._joint_support_observer=s
    return s


def reset_state(env,ids):
    if not hasattr(env,'_joint_support_observer'): return
    s=env._joint_support_observer
    for name in ('off','on','armed','active','valid','plan_valid'): s[name][ids]=0
    s['episode_id'][ids]+=1


def observe_contacts(env,contact):
    """One release trigger per foot; confirmed load rearms the next release."""
    s=state(env); tick=int(env.common_step_counter)
    if s['last_tick']==tick: return torch.zeros_like(contact),torch.zeros_like(contact)
    s['last_tick']=tick
    s['off'].copy_(torch.where(contact,0,s['off']+1));s['on'].copy_(torch.where(contact,s['on']+1,0))
    released=(s['off']>=3)&~s['armed']; loaded=(s['on']>=2)&s['armed']
    s['armed']|=released;s['armed']&=~loaded
    selected_load=loaded.gather(1,s['swing_foot'][:,None]).squeeze(1)&s['active']
    s['active'][selected_load]=False
    s['opportunities']+=int(released.sum())
    return released,loaded


def targets(env):
    """Read-only absolute-deadline references for a future, separate adapter.

    `valid` means a candidate passed this reduced-model gate, not whole-body
    feasibility. active false means no current swing target may be enforced.
    active tracks swing only; plan_valid and trajectory_valid_until retain the
    post-touchdown path until H. A later swing explicitly replaces swing_id.
    Paths have 5 knots; phase_times are absolute simulation times in seconds.
    """
    s=state(env)
    names=('active','valid','plan_valid','plan_mass','trajectory_valid_until','episode_id','swing_id','swing_foot','plan_start_time','touchdown_deadline',
           'foot_target','phase_times','com_path','velocity_path','momentum_path')
    output={name:s[name].clone() for name in names}
    output['plan_valid'] &= float(env.common_step_counter)*env.step_dt<=output['trajectory_valid_until']
    output.update({name:value.clone() for name,value in s.items() if name.startswith('selected_') and isinstance(value,torch.Tensor)})
    return output


def diagnostics(env):
    s=state(env)
    return dict(all_env_release_opportunities=s['opportunities'],selected_env_release_opportunities=s['selected_opportunities'],
                planned=s['planned'],all_env_missing_old_support=s['missed_no_old_support'],
                selected_env_missing_old_support=s['selected_no_old_support'],record_cap_opportunities=s['record_cap_opportunities'],
                record_cap_opportunities_scope='Selected environments with old support and no active plan; dropped planning opportunities after recorder cap')


def cache_plan(env,ids,new_foot,result,centers,tau):
    """Write selected neutral references once; an existing active swing is immutable."""
    s=state(env)
    if bool(s['active'][ids].any()): raise ValueError('Cannot overwrite active swing deadline')
    best=result['best_index']; valid=best>=0; safe=best.clamp_min(0)
    row=torch.arange(len(ids),device=ids.device);now=float(env.common_step_counter)*env.step_dt
    chosen_tau=tau[row,safe]
    s['active'][ids]=True;s['valid'][ids]=valid;s['swing_id'][ids]+=1;s['swing_foot'][ids]=new_foot
    s['plan_valid'][ids]=valid;s['trajectory_valid_until'][ids]=now+.6
    s['plan_start_time'][ids]=now;s['touchdown_deadline'][ids]=now+chosen_tau
    s['foot_target'][ids]=centers[row,safe]
    s['phase_times'][ids]=now+torch.stack((torch.zeros_like(chosen_tau),chosen_tau,chosen_tau+.04,chosen_tau+.08,torch.full_like(chosen_tau,.6)),-1)
    if 'phase_times' in result:s['phase_times'][ids]=now+result['phase_times'][row,safe]
    for name in ('com_path','velocity_path','momentum_path'):s[name][ids]=result[name][row,safe]
    for name in ('cop','cmp','ldot','height','actual_horizontal_momentum_rate','stage_cop','stage_cmp','stage_ldot',
                 'stage_new_load','applied_Ldot_start','applied_Ldot_end','foot_force_start','foot_force_end',
                 'input_vertical_velocity','vertical_velocity_valid'):
        if name in result:
            value=result[name]
            if value.shape[:2]==tau.shape:selected=value[row,safe]
            elif value.shape==(len(ids),):selected=value
            else:continue
            key='selected_'+name
            if key not in s:s[key]=torch.zeros((env.num_envs,*selected.shape[1:]),device=selected.device,dtype=selected.dtype)
            s[key][ids]=selected
    s['planned']+=len(ids)


@torch.no_grad()
def sample(env,action,*,stride=10,max_envs=8,planning_enabled=True):
    from .geometry import g1_sole_points,transform_sole_points
    from .upstream_terrain import surface_height
    from .support_transfer_reward import centroidal_state
    from .contact_transition_planner import plan_contact_transition
    data=env.scene['robot'].data;robot=env.scene['robot'];sensor=env.scene.sensors['contact_forces']
    if not hasattr(env,'_joint_support_ids'):
        names=['left_ankle_roll_link','right_ankle_roll_link']
        env._joint_support_ids=(robot.find_bodies(names,preserve_order=True)[0],sensor.find_bodies(names,preserve_order=True)[0],g1_sole_points(device=env.device))
    feet,contacts,local=env._joint_support_ids
    contact=tensor(sensor.data.net_forces_w)[:,contacts,2]>5.
    released,loaded=observe_contacts(env,contact);s=state(env);tick=int(env.common_step_counter)
    selected=torch.arange(env.num_envs,device=env.device)<max_envs
    s['selected_opportunities']+=int((released&selected[:,None]).sum())
    eligible=released&contact.flip(-1)&selected[:,None]&~s['active'][:,None]
    s['missed_no_old_support']+=int((released&~contact.flip(-1)).sum())
    s['selected_no_old_support']+=int((released&~contact.flip(-1)&selected[:,None]).sum())
    if not planning_enabled:
        s['record_cap_opportunities']+=int(eligible.sum());return []
    ids,newfoot=eligible.nonzero(as_tuple=True);records=[]
    if len(ids):
        # Reading existing native caches avoids advancing native reward/reset state.
        if not hasattr(env,'_upstream_heightfields') or not hasattr(env,'_beamdojo_corridor_state'):
            raise ValueError('Native physical terrain and reference caches must already exist')
        footpos=tensor(data.body_pos_w)[:,feet];quat=tensor(data.body_quat_w)[:,feet]
        soles=transform_sole_points(footpos,quat[...,[3,0,1,2]],local)
        heights,physical=surface_height(env,soles,exact=True)
        support=physical&((soles[...,2]-heights).abs()<=.035)&contact[...,None]
        supportpoints=soles.clone();supportpoints[...,2]=heights
        # Independent 9 XY positions x 3 times, anchored once at swing release.
        values=torch.tensor([-.1,0.,.1],device=env.device)
        offsets=torch.cartesian_prod(values,values).repeat_interleave(3,0)
        times=torch.tensor([.1,.2,.3],device=env.device).repeat(9)
        yaw=env._beamdojo_corridor_state['reference_yaw']
        worldoffset=torch.stack((yaw[:,None].cos()*offsets[None,:,0]-yaw[:,None].sin()*offsets[None,:,1],
                                 yaw[:,None].sin()*offsets[None,:,0]+yaw[:,None].cos()*offsets[None,:,1]),-1)
        swing=(~contact).long().argmax(-1);index=torch.arange(env.num_envs,device=env.device)
        nominal=footpos[index,swing,:2]+.2*tensor(data.body_link_lin_vel_w)[index,torch.as_tensor(feet,device=env.device)[swing],:2]
        centers=torch.zeros(env.num_envs,27,3,device=env.device)
        centers[...,:2]=nominal[:,None]+worldoffset
        top=tensor(env.scene.env_origins)[:,2]
        centers[...,2]=top[:,None]-local[:,2].mean()
        q=quat[index,swing];footyaw=torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1].square()+q[:,2].square()))
        rotated=torch.stack((footyaw[:,None].cos()*local[None,:,0]-footyaw[:,None].sin()*local[None,:,1],
                            footyaw[:,None].sin()*local[None,:,0]+footyaw[:,None].cos()*local[None,:,1]),-1)
        candidate=centers[:,:,None,:].repeat(1,1,len(local),1)
        candidate[...,:2]+=rotated[:,None];candidate[...,2]=top[:,None,None]
        ground,valid=surface_height(env,candidate,exact=True)
        point_valid=valid&((ground-top[:,None,None]).abs()<=.025)
        full_sole=point_valid.all(-1)
        mask=point_valid&full_sole[...,None];candidate[...,2]=ground
        masses=tensor(data.body_mass)[ids];inertias=tensor(data.body_inertia)[ids].reshape(len(ids),-1,3,3)
        com,velocity,momentum,mass=centroidal_state(masses,tensor(data.body_com_pos_w)[ids],tensor(data.body_com_lin_vel_w)[ids],
            tensor(data.body_ang_vel_w)[ids],inertias,tensor(data.body_com_quat_w)[ids])
        command=tensor(env.command_manager.get_command('base_velocity'))[ids,:2];angle=yaw[ids]
        worldcommand=torch.stack((angle.cos()*command[:,0]-angle.sin()*command[:,1],angle.sin()*command[:,0]+angle.cos()*command[:,1]),-1)
        tau=times[None].expand(len(ids),-1)
        result=plan_contact_transition(supportpoints[ids],support[ids],com,velocity,momentum,mass,worldcommand,newfoot,
            candidate[ids],mask[ids],tau,footpos[ids],candidate_centers=centers[ids],planning_horizon=.6)
        cache_plan(env,ids,newfoot,result,centers[ids],tau)
        s['plan_mass'][ids]=mass
        for row,eid in enumerate(ids.tolist()):
            records.append(dict(kind='decision',env_id=eid,episode_id=int(s['episode_id'][eid]),swing_id=int(s['swing_id'][eid]),tick=tick,
                simulation_time_s=tick*env.step_dt,swing_foot=int(newfoot[row]),candidate_centers=centers[eid].clone(),
                candidate_offsets_reference_xy=offsets.clone(),touchdown_times=times.clone(),candidate_absolute_deadlines=tick*env.step_dt+times,
                candidate_full_sole_supported=full_sole[eid].clone(),actual_contact=contact[eid].clone(),requested_action=action[eid].clone(),
                model={k:v[row].clone() for k,v in result.items() if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==len(ids)},
                scope='Hypothetical flat-yaw full-sole support; not predicted contact. Model outputs never executed.'))
    if tick%stride==0:
        retained=s['plan_valid']&(tick*env.step_dt<=s['trajectory_valid_until'])
        tracking_ids=(selected&(s['active']|retained|(loaded.any(-1)&(s['swing_id']>0)))).nonzero().flatten()
        if len(tracking_ids):
            from .contact_transition_reference import sample_contact_reference
            actual_com,actual_velocity,actual_momentum,actual_mass=centroidal_state(
                tensor(data.body_mass)[tracking_ids],tensor(data.body_com_pos_w)[tracking_ids],tensor(data.body_com_lin_vel_w)[tracking_ids],
                tensor(data.body_ang_vel_w)[tracking_ids],tensor(data.body_inertia)[tracking_ids].reshape(len(tracking_ids),-1,3,3),tensor(data.body_com_quat_w)[tracking_ids])
            reference=sample_contact_reference(s['phase_times'][tracking_ids],s['com_path'][tracking_ids],s['velocity_path'][tracking_ids],
                s['momentum_path'][tracking_ids],s['selected_cop'][tracking_ids],s['selected_cmp'][tracking_ids],
                s['selected_height'][tracking_ids],s['plan_mass'][tracking_ids],tick*env.step_dt)
            reference_valid=retained[tracking_ids]&reference['in_interval']
        for row,eid in enumerate(tracking_ids.tolist()):
            valid_reference=bool(reference_valid[row])
            records.append(dict(kind='tracking',env_id=eid,episode_id=int(s['episode_id'][eid]),swing_id=int(s['swing_id'][eid]),tick=tick,
                simulation_time_s=tick*env.step_dt,active=bool(s['active'][eid]),target_valid=bool(s['valid'][eid]),
                touchdown_deadline=float(s['touchdown_deadline'][eid]),remaining_seconds=float(s['touchdown_deadline'][eid])-tick*env.step_dt,
                trajectory_valid_until=float(s['trajectory_valid_until'][eid]),reference_valid=valid_reference,
                actual_com=actual_com[row].clone(),actual_velocity=actual_velocity[row].clone(),actual_momentum=actual_momentum[row].clone(),actual_mass=actual_mass[row].clone(),
                contact_fz=tensor(sensor.data.net_forces_w)[eid,contacts,2].clone(),
                actual_foot_link_velocity=tensor(data.body_link_lin_vel_w)[eid,feet].clone(),
                reference={k:v[row].clone() for k,v in reference.items()},
                reference_errors=dict(com=actual_com[row]-reference['com'][row],velocity=actual_velocity[row]-reference['velocity'][row],
                                      momentum=actual_momentum[row]-reference['momentum'][row]) if valid_reference else None,
                actual_contact=contact[eid].clone(),actual_foot_position=tensor(data.body_pos_w)[eid,feet].clone(),requested_action=action[eid].clone()))
    return records
