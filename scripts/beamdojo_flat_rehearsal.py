"""Balanced flat motor-skill rehearsal; CPU-safe command and boundary helpers."""
import torch

MODE_NAMES=('forward','backward','left','right','turn_left','turn_right','stand','mixed')

def sample_commands(mode,device=None):
    mode=torch.as_tensor(mode,device=device,dtype=torch.long)
    values=torch.zeros(len(mode),3,device=mode.device)
    random=lambda lo,hi:torch.empty(len(mode),device=mode.device).uniform_(lo,hi)
    values[:,0]=torch.where(mode==0,random(.2,1.),values[:,0])
    values[:,0]=torch.where(mode==1,random(-.5,-.1),values[:,0])
    values[:,1]=torch.where(mode==2,random(.1,.3),values[:,1])
    values[:,1]=torch.where(mode==3,random(-.3,-.1),values[:,1])
    values[:,2]=torch.where(mode==4,random(.1,.2),values[:,2])
    values[:,2]=torch.where(mode==5,random(-.2,-.1),values[:,2])
    mixed=mode==7
    values[mixed,0]=random(-.5,1.)[mixed]
    values[mixed,1]=random(-.3,.3)[mixed]
    values[mixed,2]=random(-.2,.2)[mixed]
    return values

def flat_mask(env):
    return getattr(env,'flat_rehearsal_mask',torch.zeros(env.num_envs,device=env.device,dtype=torch.bool))

def apply_flat_course_state(env,state,m,tensor):
    ids=flat_mask(env)
    if not bool(ids.any()):return state
    finite=torch.isfinite(m['root']).all(-1)&torch.isfinite(m['gravity']).all(-1)
    for key in ('lin','ang','contact_fz'):finite &= torch.isfinite(m[key]).all(-1)
    tilt=torch.acos((-m['gravity'][:,2]).clamp(-1,1))
    failure=(m['root'][:,2]<.2)|(tilt>.8)|~finite
    env.beamdojo_course_failed[ids]=failure[ids]
    for name in ('beamdojo_qualified_success','_course_finished','_course_boundary','_course_milestone'):
        getattr(env,name)[ids]=False
    state['parked'][ids]=False;state['full_ring'][ids]=False
    env.beamdojo_course_stand_time[ids]=0.
    return state

def neutral_boundary(env):
    flat=flat_mask(env)
    if not bool(flat.any()):return flat
    tensor=lambda x:x if isinstance(x,torch.Tensor) else x.torch
    local=tensor(env.scene['robot'].data.root_pos_w)-tensor(env.scene.env_origins)
    return flat_mask(env)&(local[:,:2].abs()>=3.5).any(-1)
