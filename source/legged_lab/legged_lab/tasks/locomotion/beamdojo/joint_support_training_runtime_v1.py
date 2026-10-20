"""Training-only event-locked support-reference runtime; never changes actor inputs.

Truth geometry and full centroidal state are privileged training supervision.
References pass only reduced-model gates. No action/reward is computed here.
A release selects once; neither the next release nor repeated reward calls can
move its first deadline or common H endpoint. Partial old-foot masks retain all
35 actual sample points. Complete coplanar 7x5 rectangles use the exact four
hull corners; terrain legality always uses all 35 samples before compression.
"""
from __future__ import annotations
import torch
from .contact_transition_planner_v3 import plan_two_contact_transition
from .contact_sequence_selector_v3 import select_contact_sequence
from .contact_transition_reference_v3 import sample_two_contact_reference

CORNERS=(0,4,30,34)


def tensor(value):return value if isinstance(value,torch.Tensor) else value.torch


class LockedContactPlans:
    """Pure tensor lifecycle, independent of native simulation imports."""
    def __init__(self,n,device='cpu'):
        self.n=n;self.device=torch.device(device);self.last_tick=-1;self.payload={};self.runtime_config=None
        float_shapes={'phase_times':(8,),'com_path':(8,3),'velocity_path':(8,3),'momentum_path':(8,3),
            'cop':(7,2),'cmp':(7,2),'foot_normal_load_fractions':(7,2),'height':(),
            'first_touchdown_deadline':(),'second_touchdown_deadline':(),'trajectory_valid_until':(),
            'first_target':(3,),'second_target':(3,),'selected_objective':(),'initial_foot_positions':(2,3)}
        for key,shape in float_shapes.items():self.payload[key]=torch.full((n,*shape),torch.nan,device=device)
        for key in ('first_target_valid','second_target_valid','second_touchdown_valid'):self.payload[key]=torch.zeros(n,device=device,dtype=torch.bool)
        for key in ('selected_branch','selected_index'):self.payload[key]=torch.full((n,),-1,device=device,dtype=torch.long)
        self.payload['combo_ids']=torch.full((n,3),-1,device=device,dtype=torch.long)
        self.off=torch.zeros(n,2,device=device,dtype=torch.long);self.on=torch.zeros_like(self.off)
        self.armed=torch.zeros(n,2,device=device,dtype=torch.bool)
        self.episode_id=torch.zeros(n,device=device,dtype=torch.long);self.plan_id=torch.zeros_like(self.episode_id)
        self.lock_until=torch.full((n,),-torch.inf,device=device);self.valid=torch.zeros(n,device=device,dtype=torch.bool)
        self.first_foot=torch.full((n,),-1,device=device,dtype=torch.long);self.second_foot=self.first_foot.clone()
        self.first_landed=torch.zeros_like(self.valid);self.second_landed=torch.zeros_like(self.valid)
        self.first_landing_time=torch.full((n,),torch.nan,device=device);self.second_landing_time=self.first_landing_time.clone()
        self.mass=torch.ones(n,device=device);self.no_solution=torch.zeros_like(self.valid);self.budget_dropped=torch.zeros_like(self.valid)
        self.counts=dict(release_opportunities=0,locked_releases=0,eligible_events=0,accepted_events=0,no_solution=0,budget_dropped=0,full_corner_plans=0,partial_35point_plans=0)

    def reset(self,ids):
        self.off[ids]=0;self.on[ids]=0;self.armed[ids]=False;self.episode_id[ids]+=1
        self.lock_until[ids]=-torch.inf;self.valid[ids]=False;self.first_foot[ids]=-1;self.second_foot[ids]=-1
        self.first_landed[ids]=False;self.second_landed[ids]=False;self.first_landing_time[ids]=torch.nan;self.second_landing_time[ids]=torch.nan
        self.no_solution[ids]=False;self.budget_dropped[ids]=False
        for value in self.payload.values():
            if value.dtype==torch.bool:value[ids]=False
            elif value.is_floating_point():value[ids]=torch.nan
            else:value[ids]=-1

    def observe(self,contact,tick,now,moving,done=None):
        if tick==self.last_tick:return torch.empty(0,device=self.device,dtype=torch.long),torch.empty(0,device=self.device,dtype=torch.long)
        self.last_tick=tick
        self.valid &= now<=self.lock_until
        self.off=torch.where(contact,0,self.off+1);self.on=torch.where(contact,self.on+1,0)
        released=(self.off>=3)&~self.armed;loaded=(self.on>=2)&self.armed
        self.armed|=released;self.armed&=~loaded
        active=self.valid&(now<=self.lock_until)
        row=torch.arange(self.n,device=self.device)
        first=active&~self.first_landed&loaded[row,self.first_foot.clamp_min(0)]
        # A-prime event counts only after B landed, and only on a two-contact branch.
        second=active&self.first_landed&~self.second_landed&(self.second_foot>=0)&loaded[row,self.second_foot.clamp_min(0)]
        self.first_landed[first]=True;self.first_landing_time[first]=now
        self.second_landed[second]=True;self.second_landing_time[second]=now
        locked=now<=self.lock_until
        self.counts['release_opportunities']+=int(released.sum());self.counts['locked_releases']+=int((released&locked[:,None]).sum())
        eligible=released&contact.flip(-1)&~locked[:,None]&moving[:,None]
        if done is not None:
            self.valid[done]=False;eligible &= ~done[:,None]
        return eligible.nonzero(as_tuple=True)

    def apply_budget(self,ids,feet,tick,capacity):
        if not isinstance(capacity,int) or capacity<1:raise ValueError('Positive integer event capacity required')
        self.counts['eligible_events']+=len(ids)
        if len(ids)<=capacity:return ids,feet
        order=torch.argsort((ids-int(tick))%self.n,stable=True);ids,feet=ids[order],feet[order]
        dropped=ids[capacity:];self.budget_dropped[dropped]=True;self.counts['budget_dropped']+=len(dropped)
        return ids[:capacity],feet[:capacity]

    def commit(self,ids,first_foot,selected,mass,now,horizon=.6,initial_foot_positions=None):
        if bool((now<=self.lock_until[ids]).any()):raise ValueError('Cannot overwrite locked contact plan')
        self.plan_id[ids]+=1;self.lock_until[ids]=now+horizon
        self.valid[ids]=selected['valid'];self.no_solution[ids]=~selected['valid'];self.budget_dropped[ids]=False
        self.first_foot[ids]=torch.where(selected['valid'],first_foot,-1)
        self.second_foot[ids]=torch.where(selected['second_touchdown_valid'],1-first_foot,-1)
        self.first_landed[ids]=False;self.second_landed[ids]=False;self.first_landing_time[ids]=torch.nan;self.second_landing_time[ids]=torch.nan
        self.mass[ids]=mass
        if initial_foot_positions is not None:
            if 'initial_foot_positions' not in self.payload:self.payload['initial_foot_positions']=torch.full((self.n,2,3),torch.nan,device=self.device)
            self.payload['initial_foot_positions'][ids]=initial_foot_positions
        # Eligible-combo matrices are diagnostics, not per-step training state.
        for key,value in selected.items():
            if not isinstance(value,torch.Tensor) or key in ('hold_eligible','two_eligible'):continue
            if key not in self.payload:
                self.payload[key]=torch.zeros((self.n,*value.shape[1:]),device=value.device,dtype=value.dtype)
                if value.is_floating_point():self.payload[key].fill_(torch.nan)
                elif value.dtype!=torch.bool:self.payload[key].fill_(-1)
            self.payload[key][ids]=value
        self.counts['accepted_events']+=len(ids);self.counts['no_solution']+=int((~selected['valid']).sum())

    def references(self,now,moving=None,done=None):
        valid=self.valid&(now<=self.lock_until)
        if moving is not None:valid &= moving
        if done is not None:valid &= ~done
        output={k:v for k,v in self.payload.items()}
        output.update(valid=valid,plan_valid=valid,now=now,episode_id=self.episode_id,plan_id=self.plan_id,first_support_foot=self.first_foot,
            second_support_foot=self.second_foot,physical_slots=torch.where((self.first_foot>=0)[:,None],torch.stack((1-self.first_foot,self.first_foot),-1),-1),
            first_landed=self.first_landed,second_landed=self.second_landed,first_landing_time=self.first_landing_time,
            second_landing_time=self.second_landing_time,no_solution=self.no_solution,budget_dropped=self.budget_dropped,
            lock_until=self.lock_until,mass=self.mass,plan_mass=self.mass)
        if 'phase_times' not in self.payload:return output
        p=self.payload
        ref=sample_two_contact_reference(p['phase_times'],p['com_path'],p['velocity_path'],p['momentum_path'],p['cop'],p['cmp'],p['height'],self.mass,now,
                foot_normal_load_fractions=p['foot_normal_load_fractions'],plan_valid=valid)
        output['reference']=ref
        output['first_touchdown_valid']=valid&~self.first_landed&p['first_target_valid']
        output['second_touchdown_valid']=valid&~self.second_landed&p['second_target_valid']
        output['first_timing_error']=torch.where(self.first_landed,self.first_landing_time-p['first_touchdown_deadline'],torch.nan)
        output['second_timing_error']=torch.where(self.second_landed,self.second_landing_time-p['second_touchdown_deadline'],torch.nan)
        return output


def exact_corner_rows(args,second_points,second_mask):
    """Exact hull reduction only for a full old-foot 7x5 rectangle on one plane."""
    points,mask=args[:2];n=len(points)
    if points.shape[2]!=35:return torch.zeros(n,device=points.device,dtype=torch.bool)
    row=torch.arange(n,device=points.device);old=1-args[7];oldpoints=points[row,old];oldmask=mask[row,old]
    okay=oldmask.all(-1)&((oldpoints[...,2]-oldpoints[:,0:1,2]).abs().amax(-1)<=1e-6)
    for p,m in ((args[8],args[9]),(second_points,second_mask)):
        full=m.all(-1);empty=~m.any(-1)
        coplanar=(p[...,2]-p[...,0:1,2]).abs().amax(-1)<=1e-6
        okay &= ((full&coplanar)|empty).all(-1)
    return okay


def combination_axes(first_count,second_count,device):
    combo=torch.cartesian_prod(torch.arange(first_count,device=device),torch.arange(second_count,device=device),torch.arange(4,device=device))
    return combo,torch.tensor([.36,.40,.44,.48],device=device)[combo[:,2]]


@torch.no_grad()
def plan_batch_full(args,first_centers,second_points,second_mask,second_centers,now,*,corner_chunk_size=4096,partial_chunk_size=256):
    """All combinations and both branches; no V1-friction pruning or future data."""
    combo,tau2=combination_axes(args[8].shape[1],second_points.shape[1],args[0].device)
    first,second=combo[:,0],combo[:,1];n=len(args[0]);selected_parts=[]
    fast=exact_corner_rows(args,second_points,second_mask)
    for use_corners in (True,False):
        ids=(fast==use_corners).nonzero().flatten()
        if not len(ids):continue
        a=[x[ids] for x in args]
        for j in (8,9,10):a[j]=a[j][:,first]
        sp=second_points[ids][:,second];sm=second_mask[ids][:,second]
        if use_corners:
            ci=torch.tensor(CORNERS,device=ids.device)
            a[0]=a[0][:,:,ci];a[1]=a[1][:,:,ci];a[8]=a[8][:,:,ci];a[9]=a[9][:,:,ci];sp=sp[:,:,ci];sm=sm[:,:,ci]
        fc=first_centers[ids][:,first];sc=second_centers[ids][:,second]
        kw=dict(candidate_centers=fc,second_candidate_centers=sc,planning_horizon=.6,friction=.5,
                chunk_size=corner_chunk_size if use_corners else partial_chunk_size)
        h=plan_two_contact_transition(*a,sp,sm,tau2[None].expand(len(ids),-1),second_contact_enabled=False,**kw)
        t=plan_two_contact_transition(*a,sp,sm,tau2[None].expand(len(ids),-1),second_contact_enabled=True,**kw)
        chosen=select_contact_sequence(h,t,now,first_targets=fc,second_targets=sc)
        chosen['combo_ids']=torch.where(chosen['valid'][:,None],combo[chosen['selected_index'].clamp_min(0)],-1)
        selected_parts.append((ids,chosen))
    output={}
    for ids,part in selected_parts:
        for key,value in part.items():
            if not isinstance(value,torch.Tensor):continue
            if key not in output:output[key]=torch.empty((n,*value.shape[1:]),device=value.device,dtype=value.dtype)
            output[key][ids]=value
    return output,fast


@torch.no_grad()
def plan_batch(args,first_centers,second_points,second_mask,second_centers,now,*,corner_chunk_size=4096,partial_chunk_size=256):
    """Exact necessary-condition pruning of the same complete Cartesian bank.

    Mathematical full-rectangle hulls agree; float32 four-corner vs35 arithmetic
    can differ by ~2e-4 in world-coordinate velocity/CoP on saved stone cases.
    Partial sole masks always retain the original 35-point route.
    """
    from .contact_sequence_pruning_v1 import plan_pruned
    n=len(args[0]);fast=exact_corner_rows(args,second_points,second_mask);output={}
    for use_corners in (True,False):
        ids=(fast==use_corners).nonzero().flatten()
        if not len(ids):continue
        a=[x[ids] for x in args];sp=second_points[ids];sm=second_mask[ids]
        if use_corners:
            ci=torch.tensor(CORNERS,device=ids.device)
            a[0]=a[0][:,:,ci];a[1]=a[1][:,:,ci];a[8]=a[8][:,:,ci];a[9]=a[9][:,:,ci];sp=sp[:,:,ci];sm=sm[:,:,ci]
        part=plan_pruned(a,first_centers[ids],sp,sm,second_centers[ids],now,chunk_size=corner_chunk_size if use_corners else partial_chunk_size)
        for key,value in part.items():
            if not isinstance(value,torch.Tensor):continue
            if key not in output:output[key]=torch.empty((n,*value.shape[1:]),device=value.device,dtype=value.dtype)
            output[key][ids]=value
    return output,fast


def indexed_surface_height(env,points,ids):
    """Exact upstream raster query on selected env rows, without ensure_state mutation."""
    from .upstream_terrain import course_size,RESOLUTION
    if not hasattr(env,'_upstream_heightfields'):raise ValueError('Physical terrain cache must exist before training reference query')
    origins=tensor(env.scene.env_origins)[ids];n=len(ids);shape=points.shape[:-1]
    local=(points-origins.reshape((n,)+(1,)*(points.ndim-2)+(3,))).reshape(n,-1,3)
    size=torch.as_tensor(course_size(env.cfg),device=points.device,dtype=points.dtype)
    inside=torch.isfinite(local[...,:2]).all(-1)&(local[...,:2]>=-size/2).all(-1)&(local[...,:2]<size/2).all(-1)
    pixels=((local[...,:2]+size/2)/RESOLUTION).long().clamp_min(0)
    maximum=torch.as_tensor(env._upstream_heightfields.shape[1:],device=points.device)-1
    x,y=torch.minimum(pixels,maximum).unbind(-1)
    h=env._upstream_heightfields[env.beamdojo_levels[ids,None],x,y]
    h=torch.where(inside,h,-torch.ones_like(h))
    return (h+origins[:,2:3]).reshape(shape),(inside&(h>=-.1)).reshape(shape)


@torch.no_grad()
def collect_inputs(env,ids,swing):
    from .geometry import transform_sole_points
    from .support_transfer_reward import centroidal_state
    from .joint_support_diagnostic_runtime_v2 import terrain_search_lattice,select_legal_positions
    feet,_,local=env._joint_training_foot_ids_v1;data=env.scene['robot'].data
    footpos=tensor(data.body_pos_w)[ids][:,feet];quat=tensor(data.body_quat_w)[ids][:,feet]
    soles=transform_sole_points(footpos,quat[...,[3,0,1,2]],local)
    ground,physical=indexed_surface_height(env,soles,ids)
    force=tensor(env.scene.sensors['contact_forces'].data.net_forces_w)[ids][:,env._joint_training_foot_ids_v1[1],2]
    supported=physical&((soles[...,2]-ground).abs()<=.035)&(force>5)[...,None]
    supportpoints=soles.clone();supportpoints[...,2]=ground
    masses=tensor(data.body_mass)[ids]
    com,velocity,momentum,mass=centroidal_state(masses,tensor(data.body_com_pos_w)[ids],tensor(data.body_com_lin_vel_w)[ids],
        tensor(data.body_ang_vel_w)[ids],tensor(data.body_inertia)[ids].reshape(len(ids),-1,3,3),tensor(data.body_com_quat_w)[ids])
    yaw=env._beamdojo_corridor_state['reference_yaw'][ids];command=tensor(env.command_manager.get_command('base_velocity'))[ids,:2]
    worldcmd=torch.stack((yaw.cos()*command[:,0]-yaw.sin()*command[:,1],yaw.sin()*command[:,0]+yaw.cos()*command[:,1]),-1)
    top=tensor(env.scene.env_origins)[ids,2]
    grid,anchors,offsets=terrain_search_lattice(com,velocity,worldcmd,yaw,footpos,swing,top)
    row=torch.arange(len(ids),device=ids.device)
    def bank(foot,second):
        q=quat[row,foot];angle=torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1].square()+q[:,2].square()))
        rotated=torch.stack((angle[:,None].cos()*local[None,:,0]-angle[:,None].sin()*local[None,:,1],angle[:,None].sin()*local[None,:,0]+angle[:,None].cos()*local[None,:,1]),-1)
        points=torch.zeros(len(ids),len(offsets),35,3,device=ids.device);points[...,:2]=grid[:,:,None]+rotated[:,None];points[...,2]=top[:,None,None]
        h,geometric=indexed_surface_height(env,points,ids);point_valid=geometric&((h-top[:,None,None]).abs()<=.025)
        if second:
            lateral=torch.stack((-yaw.sin(),yaw.cos()),-1);separation=((footpos[:,0,:2]-footpos[:,1,:2])*lateral).sum(-1).abs()
            shift=(separation*.5).clamp(.06,.18)*torch.where(foot==0,1.,-1.)
            anchor=com[:,:2]+.44*worldcmd+shift[:,None]*lateral
            chosen,valid,_=select_legal_positions(grid,point_valid.all(-1)&((grid-footpos[row,foot,:2][:,None]).norm(dim=-1)<=.75),anchor[:,None].expand(-1,3,-1))
            chosen,valid=chosen[:,:6],valid[:,:6]
        else:
            legal=point_valid.all(-1)&((grid-footpos[row,1-foot,:2][:,None]).norm(dim=-1)<=.7)&((grid-footpos[row,foot,:2][:,None]).norm(dim=-1)<=.75)
            chosen,valid,_=select_legal_positions(grid,legal,anchors);chosen=chosen.repeat_interleave(3,-1);valid=valid.repeat_interleave(3,-1)
        selected=points.gather(1,chosen[...,None,None].expand(-1,-1,35,3));selected[...,2]=h.gather(1,chosen[...,None].expand(-1,-1,35))
        mask=point_valid.gather(1,chosen[...,None].expand(-1,-1,35))&valid[...,None]
        centers=torch.zeros(len(ids),chosen.shape[1],3,device=ids.device);centers[...,:2]=grid.gather(1,chosen[...,None].expand(-1,-1,2));centers[...,2]=top[:,None]-local[:,2].mean()
        return selected,mask,centers
    first,firstmask,firstcenters=bank(swing,False);second,secondmask,secondcenters=bank(1-swing,True)
    tau=torch.tensor([.1,.2,.3],device=ids.device).repeat(18)[None].expand(len(ids),-1)
    args=(supportpoints,supported,com,velocity,momentum,mass,worldcmd,swing,first,firstmask,tau,footpos)
    return args,firstcenters,second,secondmask,secondcenters


def state(env):
    if not hasattr(env,'_joint_support_training_v1'):env._joint_support_training_v1=LockedContactPlans(env.num_envs,env.device)
    return env._joint_support_training_v1


def reset(env,ids):
    if hasattr(env,'_joint_support_training_v1'):env._joint_support_training_v1.reset(ids)


@torch.no_grad()
def update(env,*,done=None,event_batch_size=None,max_events_per_step=None):
    """Call once from postphysics reward evaluation; caller supplies current done mask.

    No reward/action is emitted. Deployment observations remain unchanged.
    Sparse capacity drops are explicit counters, never delayed replans of an
    already released foot. Config defaults to eight event environments per
    native step, batches of eight; this does NOT supervise all 4096 environments.
    Native launcher must enforce the authorized 14:35:20 UTC deadline.
    """
    from .geometry import g1_sole_points
    if event_batch_size is None:event_batch_size=int(getattr(env.cfg,'beamdojo_joint_support_event_batch_size',8))
    if max_events_per_step is None:max_events_per_step=int(getattr(env.cfg,'beamdojo_joint_support_max_events_per_step',8))
    partial_chunk_size=int(getattr(env.cfg,'beamdojo_joint_support_partial_chunk_size',256))
    if partial_chunk_size<1:raise ValueError('Positive partial chunk size required')
    if event_batch_size<1 or max_events_per_step<1:raise ValueError('Positive event batch and capacity required')
    if not hasattr(env,'_joint_training_foot_ids_v1'):
        names=['left_ankle_roll_link','right_ankle_roll_link'];robot=env.scene['robot'];sensor=env.scene.sensors['contact_forces']
        env._joint_training_foot_ids_v1=(robot.find_bodies(names,preserve_order=True)[0],sensor.find_bodies(names,preserve_order=True)[0],g1_sole_points(device=env.device))
    s=state(env);now=float(env.common_step_counter)*env.step_dt
    config=(event_batch_size,max_events_per_step,partial_chunk_size)
    if s.runtime_config is None:s.runtime_config=config
    elif s.runtime_config!=config:raise ValueError('Training reference budget changed during run')
    contacts=tensor(env.scene.sensors['contact_forces'].data.net_forces_w)[:,env._joint_training_foot_ids_v1[1],2]>5
    moving=tensor(env.command_manager.get_command('base_velocity'))[:,:2].norm(dim=-1)>=.2
    if done is not None:done=tensor(done).bool()
    ids,foot=s.observe(contacts,int(env.common_step_counter),now,moving,done)
    ids,foot=s.apply_budget(ids,foot,int(env.common_step_counter),max_events_per_step)
    for start in range(0,len(ids),event_batch_size):
        batch=ids[start:start+event_batch_size];new=foot[start:start+event_batch_size]
        args,centers,second,secondmask,secondcenters=collect_inputs(env,batch,new)
        chosen,fast=plan_batch(args,centers,second,secondmask,secondcenters,now,partial_chunk_size=partial_chunk_size)
        s.commit(batch,new,chosen,args[5],now,initial_foot_positions=args[11])
        s.counts['full_corner_plans']+=int(fast.sum());s.counts['partial_35point_plans']+=int((~fast).sum())
    result=s.references(now,moving,done);result['actual_contact']=contacts
    result['actual_foot_positions']=tensor(env.scene['robot'].data.body_pos_w)[:,env._joint_training_foot_ids_v1[0]]
    result['contact_fz']=tensor(env.scene.sensors['contact_forces'].data.net_forces_w)[:,env._joint_training_foot_ids_v1[1],2]
    result['statistics']=dict(s.counts)
    eligible=s.counts['eligible_events'];accepted=s.counts['accepted_events']
    result['statistics'].update(accepted_per_eligible=accepted/eligible if eligible else 0.,budget_drop_per_eligible=s.counts['budget_dropped']/eligible if eligible else 0.,
        valid_plans_per_eligible=(accepted-s.counts['no_solution'])/eligible if eligible else 0.,event_cap=max_events_per_step,event_batch_size=event_batch_size,
        coverage_scope='Moving, nonterminal release opportunities with old support and no unexpired lock; sparse rotating cap, not all-environment supervision.')
    return result


# Explicit integration name: call after the native reset for only the reset IDs.
reset_state=reset
