"""Square-course state/parking only. CPU-safe import; no new motion rewards.

Sim6 input quaternions are XYZW. Continuous planar envelope containment in one
convex ring rectangle is certified by transformed four corners; this does not
prove pressure, stability or swing feasibility. Native imports are delayed.
"""
import torch


def tensor(x):
    return x if isinstance(x, torch.Tensor) else x.torch


def params(env):
    p = dict(getattr(env.cfg, 'beamdojo_square_course_params', {}))
    if 'start_platform_m' not in p or 'end_platform_m' not in p:
        raise ValueError('Explicit square-course start/end platform rectangles required')
    defaults = dict(target_root_x=3.25, braking_start_x=3.05, route_half_width=.42,
                    stand_seconds=1., end_half_width=.6, boundary_limit=3.9)
    defaults.update(p)
    if not defaults['braking_start_x'] < defaults['target_root_x']:
        raise ValueError('Braking must start before target')
    return defaults


def ensure_state(env):
    if hasattr(env, '_square_course_state'):
        return env._square_course_state
    params(env)
    n, device = env.num_envs, env.device
    z = lambda: torch.zeros(n, device=device)
    b = lambda: torch.zeros(n, device=device, dtype=torch.bool)
    s = dict(last_tick=torch.full((n,), -1, device=device, dtype=torch.long),
             parked=b(), full_ring=b(), departed=b(), initialized=b(),
             last_xy=torch.zeros(n,2,device=device), start_xy=torch.zeros(n,2,device=device))
    env._square_course_state = s
    for name in ('distance','contact_samples','unsafe_samples','stand_time','signed_progress',
                 'net_displacement','valid_contact_seconds'):
        setattr(env, 'beamdojo_course_'+name, z())
    for name in ('beamdojo_course_failed','beamdojo_qualified_success','_course_finished',
                 '_course_boundary','_course_milestone','_upstream_started'):
        setattr(env, name, b())
    return s


def transform_xyzw(position, quaternion, points):
    q = quaternion[...,None,:]
    p = points.expand(*position.shape[:-1], -1, 3)
    v = q[...,:3].expand_as(p)
    t = 2*torch.cross(v,p,dim=-1)
    return position[...,None,:]+p+q[...,3:]*t+torch.cross(v,t,dim=-1)


def read_measurements(env):
    from legged_lab.tasks.locomotion.beamdojo.geometry import g1_sole_points
    robot=env.scene['robot']; sensor=env.scene.sensors['contact_forces']; d=robot.data
    if not hasattr(env, '_square_course_ids'):
        names=['left_ankle_roll_link','right_ankle_roll_link']
        env._square_course_ids=(robot.find_bodies(names,preserve_order=True)[0],
                               sensor.find_bodies(names,preserve_order=True)[0],
                               g1_sole_points(device=env.device))
    feet, contact_ids, points=env._square_course_ids
    pos=tensor(d.body_pos_w)[:,feet]; quat=tensor(d.body_quat_w)[:,feet]
    soles=transform_xyzw(pos,quat,points)
    # Corners of the actual G1 sampling envelope; expand by 1cm safety margin.
    lo=points.min(0).values; hi=points.max(0).values
    corners=torch.stack([lo,lo.clone(),hi,hi.clone()])
    corners[1,0]=hi[0]; corners[3,0]=lo[0]
    corners[:,0] += torch.tensor([-.01,.01,.01,-.01],device=env.device)
    corners[:,1] += torch.tensor([-.01,-.01,.01,.01],device=env.device)
    origin=tensor(env.scene.env_origins)
    corners=transform_xyzw(pos,quat,corners)-origin[:,None,None,:]
    local=soles-origin[:,None,None,:]
    callback=getattr(env,'_square_course_surface_height',None)
    if callback is not None:
        height, valid=callback(soles)
        supported=valid & ((soles[...,2]-height).abs()<=.035)
    else:
        rect=getattr(env,'beamdojo_square_course_rectangles',None)
        if rect is None: rect=params(env).get('rectangles_m')
        if rect is None: raise ValueError('Course rectangles or exact surface callback required')
        rect=torch.as_tensor(rect,device=env.device,dtype=local.dtype)
        if rect.ndim==2:rect=rect[None].expand(env.num_envs,-1,-1)
        supported=torch.zeros_like(local[...,0],dtype=torch.bool)
        for j in range(rect.shape[1]):
            r=rect[:,j]; usable=torch.isfinite(r).all(-1)&(r[:,2]>r[:,0])&(r[:,3]>r[:,1])
            inside=(local[...,0]>=r[:,None,None,0])&(local[...,0]<=r[:,None,None,2])
            inside &= (local[...,1]>=r[:,None,None,1])&(local[...,1]<=r[:,None,None,3])
            supported |= inside & usable[:,None,None]
        supported &= local[...,2].abs()<=.035
    return dict(root=tensor(d.root_pos_w)-origin, corners=corners,
                contact_fz=tensor(sensor.data.net_forces_w)[:,contact_ids,2],
                lin=tensor(d.root_lin_vel_w), ang=tensor(d.root_ang_vel_b),
                gravity=tensor(d.projected_gravity_b), supported=supported.all(-1))


def update_course(env, measurements=None):
    """One accumulated sample per environment/tick; explicit measurements for CPU tests."""
    s=ensure_state(env); p=params(env); tick=int(env.common_step_counter)
    active=s['last_tick']!=tick
    if not bool(active.any()): return s
    m=read_measurements(env) if measurements is None else measurements
    root=m['root']; corners=m['corners']; contact=m['contact_fz']>5
    rect=torch.as_tensor(p['end_platform_m'],device=root.device,dtype=root.dtype)
    full=((corners[...,0]>=rect[0])&(corners[...,0]<=rect[2])&
          (corners[...,1]>=max(float(rect[1]),-p['end_half_width']))&
          (corners[...,1]<=min(float(rect[3]),p['end_half_width']))&
          (corners[...,2].abs()<=.035)).all(dim=(-1,-2))
    finite=torch.isfinite(root).all(-1)&torch.isfinite(corners).all(dim=(-1,-2,-3))
    finite &= torch.isfinite(m['lin']).all(-1)&torch.isfinite(m['ang']).all(-1)&torch.isfinite(m['gravity']).all(-1)
    tilt=torch.acos((-m['gravity'][:,2]).clamp(-1,1))
    terms=getattr(env.cfg,'terminations',None)
    h=getattr(getattr(terms,'base_height',None),'params',{}).get('minimum_height',.2)
    angle=getattr(getattr(terms,'bad_orientation',None),'params',{}).get('limit_angle',.8)
    departed=s['departed']|(root[:,0]>p['start_platform_m'][2])
    boundary=(root[:,:2].abs()>=p['boundary_limit']).any(-1)
    if p['route_half_width'] is not None: boundary |= departed & (root[:,1].abs()>p['route_half_width'])
    failure=(root[:,2]<h)|(tilt>angle)|~finite|boundary
    failure |= env.beamdojo_course_failed
    expired=tensor(env.episode_length_buf)>=env.max_episode_length
    stable=full&contact.all(-1)&(m['lin'].norm(dim=-1)<.15)&(m['ang'].norm(dim=-1)<.3)&(tilt<.35)
    command=tensor(env.command_manager.get_command('base_velocity'))
    zero_command=torch.isfinite(command).all(-1)&(command[:,:2].norm(dim=-1)<1e-4)&(command[:,2].abs()<1e-4)
    stable &= s['parked'] & zero_command & ~failure & ~expired & (root[:,0]>=p['target_root_x']-.05)
    duration=torch.where(stable,env.beamdojo_course_stand_time+env.step_dt,0.)
    success=(duration>=p['stand_seconds']-1e-6)&~failure&~expired
    first=~s['initialized']; delta=(root[:,:2]-s['last_xy']).norm(dim=-1)
    delta=torch.where(first,0.,delta)
    unsafe=contact & ~m['supported']
    safe=contact.any(-1)&~unsafe.any(-1)&~failure
    values={'beamdojo_course_stand_time':duration,'beamdojo_course_failed':failure,
            'beamdojo_qualified_success':success,'_course_finished':success,
            '_course_boundary':boundary,'_course_milestone':full,
            'beamdojo_course_distance':env.beamdojo_course_distance+torch.where(safe,delta,0.),
            'beamdojo_course_contact_samples':env.beamdojo_course_contact_samples+contact.sum(-1),
            'beamdojo_course_unsafe_samples':env.beamdojo_course_unsafe_samples+unsafe.sum(-1),
            'beamdojo_course_valid_contact_seconds':env.beamdojo_course_valid_contact_seconds+safe*env.step_dt}
    s['start_xy'][active&first]=root[active&first,:2]
    values['beamdojo_course_signed_progress']=root[:,0]-s['start_xy'][:,0]
    values['beamdojo_course_net_displacement']=(root[:,:2]-s['start_xy']).norm(dim=-1)
    for name,value in values.items():getattr(env,name)[active]=value[active]
    s['full_ring'][active]=full[active];s['departed'][active]=departed[active]
    s['last_xy'][active]=root[active,:2];s['initialized'][active]=True
    env._upstream_started[active]=True;s['last_tick'][active]=tick
    return s


def parking_command(env, cruise, root_x, full_ring):
    """Cruise .5..1.0; no stop before complete ring entry; parked is absorbing."""
    s=ensure_state(env);p=params(env)
    s['parked'] |= full_ring & (root_x>=p['target_root_x'])
    remaining=(p['target_root_x']-root_x)/(p['target_root_x']-p['braking_start_x'])
    speed=cruise.clamp(.5,1.)*remaining.clamp(0,1)
    speed=torch.where(~full_ring,speed.clamp_min(.25),speed.clamp_min(.10))
    speed=torch.where(s['parked'],0.,speed)
    return torch.stack((speed,torch.zeros_like(speed),torch.zeros_like(speed)),-1)


def course_success(env):
    update_course(env);return env._course_finished


def course_boundary(env):
    update_course(env);return env._course_boundary


def time_out(env):
    update_course(env)
    return (tensor(env.episode_length_buf)>=env.max_episode_length) & ~env.beamdojo_course_failed


def course_metrics(env):
    ensure_state(env)
    return {'Metrics/square_success':env.beamdojo_qualified_success.float().mean(),
            'Metrics/square_stand_seconds':env.beamdojo_course_stand_time.mean(),
            'Metrics/course_valid_distance':env.beamdojo_course_distance.mean(),
            'Metrics/square_unsafe_fraction':(env.beamdojo_course_unsafe_samples/env.beamdojo_course_contact_samples.clamp_min(1)).mean()}


def reset_course(env, env_ids):
    s=ensure_state(env)
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
    for name,value in s.items():
        value[ids]=int(env.common_step_counter) if name=='last_tick' else 0
    for name in ('distance','contact_samples','unsafe_samples','stand_time','signed_progress','net_displacement','valid_contact_seconds','failed'):
        getattr(env,'beamdojo_course_'+name)[ids]=0
    for name in ('beamdojo_qualified_success','_course_finished','_course_boundary','_course_milestone','_upstream_started'):
        getattr(env,name)[ids]=False
    for name,(history,updated) in getattr(env,'_upstream_observation_histories',{}).items():
        history[ids]=0.;updated[ids]=-1
        if hasattr(env,'_upstream_history_pending'):env._upstream_history_pending.setdefault(name,[]).append(ids.clone())
    env._upstream_critic_frame_cache=None;env._upstream_heightmap_cache=None
    return {}


def make_command_class(_base_class=None):
    """Delay Isaac import; optional base injection is only a CPU interface-test seam."""
    if _base_class is None:
        from isaaclab.envs.mdp.commands import UniformVelocityCommand
        _base_class=UniformVelocityCommand
    class SquareCourseVelocityCommand(_base_class):
        def _resample_command(self,env_ids):
            super()._resample_command(env_ids)
            if not hasattr(self,'_square_cruise'):
                self._square_cruise=torch.full((self._env.num_envs,),.75,device=self.vel_command_b.device)
            self._square_cruise[env_ids]=torch.empty_like(self._square_cruise[env_ids]).uniform_(.5,1.)
            self.vel_command_b[env_ids]=0
            self.vel_command_b[env_ids,0]=self._square_cruise[env_ids]
            s=ensure_state(self._env)
            self.vel_command_b[s['parked']]=0
        def _update_command(self):
            # Direct fresh geometry read avoids accumulating course timers during
            # native command updates (which may occur after reset in same tick).
            m=read_measurements(self._env);p=params(self._env);c=m['corners'];r=p['end_platform_m']
            full=((c[...,0]>=r[0])&(c[...,0]<=r[2])&(c[...,1]>=max(r[1],-p['end_half_width']))&
                  (c[...,1]<=min(r[3],p['end_half_width']))&(c[...,2].abs()<=.035)).all((-1,-2))
            if not hasattr(self,'_square_cruise'):self._square_cruise=self.vel_command_b[:,0].clone().clamp(.5,1.)
            self.vel_command_b[:]=parking_command(self._env,self._square_cruise,m['root'][:,0],full)
    return SquareCourseVelocityCommand
