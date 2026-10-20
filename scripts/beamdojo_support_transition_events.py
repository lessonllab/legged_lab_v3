"""Training-only delayed labels, captured before native autoreset.

No world-state or contact labels are appended to actor observations. Prefix
frames are immutable; a reset invalidates a pending handoff, never turns it
into a successful target. Every label refers to the first subsequent contact
of the named foot after a sustained swing.
"""
import torch
from beamdojo_support_transition_v1 import EventReplay


def tensor(value):
    return value if isinstance(value,torch.Tensor) else value.torch


def landing_cell(foot_world,root_world,yaw):
    delta=foot_world[:,:2]-root_world[:,:2]
    x=yaw.cos()*delta[:,0]+yaw.sin()*delta[:,1]
    y=-yaw.sin()*delta[:,0]+yaw.cos()*delta[:,1]
    xy=torch.stack((x,y),-1)
    visible=torch.isfinite(xy).all(-1)&(xy.abs()<=.7).all(-1)
    grid=((torch.nan_to_num(xy)+.7)/.1).round().long().clamp(0,14)
    return grid[:,0]*15+grid[:,1],visible


class NativeEventCollector:
    def __init__(self,native):
        self.native=native;self.env=native.unwrapped
        self.n=native.num_envs;self.device=native.device;self.dt=self.env.step_dt
        if abs(self.dt-.02)>1e-9:raise ValueError('Event tick contract is50Hz')
        self.robot=self.env.scene['robot'];self.sensor=self.env.scene.sensors['contact_forces']
        names=['left_ankle_roll_link','right_ankle_roll_link']
        self.feet=self.robot.find_bodies(names,preserve_order=True)[0]
        self.contacts=self.sensor.find_bodies(names,preserve_order=True)[0]
        self.replay=EventReplay(self.device)
        self.tick=0;self.ring=40;self.delay=6;self.lags=(4,8,12,16,20,24)
        self.prefix=torch.empty(self.ring,self.n,708,device=self.device,dtype=torch.float16)
        self.root=torch.empty(self.ring,self.n,3,device=self.device)
        self.yaw=torch.empty(self.ring,self.n,device=self.device)
        self.stamps=torch.full((self.ring,),-9999,device=self.device,dtype=torch.long)
        self.episode=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.prefix_episode=torch.full((self.ring,self.n),-1,device=self.device,dtype=torch.long)
        self.air=torch.zeros(self.n,2,device=self.device,dtype=torch.long)
        self.last_touch=torch.zeros_like(self.air)
        self.pending=torch.zeros(self.n,2,device=self.device,dtype=torch.bool)
        self.touch_tick=torch.zeros_like(self.air)
        self.previous_touch=torch.zeros_like(self.air)
        self.touch_position=torch.zeros(self.n,2,3,device=self.device)
        self.rows=torch.arange(self.n,device=self.device)
        self.event_total=0;self.incomplete=0;self.final_frame_labels=0
        self.obs=native.get_observations()

    @torch.no_grad()
    def record_prefix(self,obs):
        slot=self.tick%self.ring;d=self.robot.data
        self.prefix[slot]=torch.cat([obs[k] for k in ('policy','terrain','course_tracking')],-1)
        self.root[slot]=tensor(d.root_pos_w)
        q=tensor(d.root_quat_w)
        self.yaw[slot]=torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2))
        self.prefix_episode[slot]=self.episode;self.stamps[slot]=self.tick

    @torch.no_grad()
    def post_physics(self):
        self.tick+=1;d=self.robot.data
        fz=tensor(self.sensor.data.net_forces_w)[:,self.contacts,2]
        contact=fz>5.
        # Complete prior events BEFORE recording any new event in the same slot.
        ready=self.pending & (self.tick-self.touch_tick>=self.delay)
        terminal=tensor(self.env.reset_buf).bool() if hasattr(self.env,'reset_buf') else torch.zeros(self.n,device=self.device,dtype=torch.bool)
        # Physical outcomes on a terminal frame remain observed data, but do not
        # cross its subsequent reset. These are not tagged as viable/successful.
        velocity=tensor(d.root_lin_vel_b)[:,:2]
        gravity=tensor(d.projected_gravity_b)[:,:2]/.3
        omega=tensor(d.root_ang_vel_b)[:,2:3]
        load=fz.clamp_min(0)/fz.clamp_min(0).sum(-1,keepdim=True).clamp_min(1.)
        for foot in (0,1):
            for lag in self.lags:
                anchor=self.touch_tick[:,foot]-lag
                slot=anchor.remainder(self.ring)
                valid=ready[:,foot]&(self.stamps[slot]==anchor)
                valid &= self.prefix_episode[slot,self.rows]==self.episode
                valid &= anchor>self.previous_touch[:,foot]
                ids=valid.nonzero(as_tuple=False).flatten()
                if ids.numel()==0:continue
                slots=slot[ids]
                raw=self.prefix[slots,ids]
                cell,visible=landing_cell(self.touch_position[ids,foot],self.root[slots,ids],self.yaw[slots,ids])
                target=torch.cat((velocity[ids].new_full((len(ids),1),lag*self.dt/.5),
                                  velocity[ids],gravity[ids],omega[ids],load[ids,foot:foot+1]),-1)
                if not torch.isfinite(target).all():raise ValueError('Nonfinite native handoff label')
                self.replay.add(raw,torch.full_like(ids,foot),cell,target,visible)
                self.final_frame_labels+=int(terminal[ids].sum())
        self.pending &= ~ready
        new_contact=contact&(self.air>=3)
        self.event_total+=int(new_contact.sum())
        self.incomplete+=int((self.pending&new_contact).sum())
        # Retain the normal tensors allocated in __init__. Replacing them with
        # tensors created inside runner inference_mode makes a later explicit
        # reset outside that context fail before the first optimizer update.
        self.previous_touch.copy_(torch.where(new_contact,self.last_touch,self.previous_touch))
        self.last_touch.copy_(torch.where(new_contact,self.tick,self.last_touch))
        self.touch_tick.copy_(torch.where(new_contact,self.tick,self.touch_tick))
        self.touch_position.copy_(torch.where(new_contact[...,None],tensor(d.body_pos_w)[:,self.feet],self.touch_position))
        self.pending |= new_contact
        self.air.copy_(torch.where(contact,0,self.air+1))

    @torch.no_grad()
    def reset_ids(self,ids):
        self.incomplete+=int(self.pending[ids].sum())
        self.episode[ids]+=1;self.air[ids]=0;self.last_touch[ids]=self.tick
        self.pending[ids]=False

    def metadata(self):
        return dict(events=self.event_total,labels=self.replay.total,replay_size=self.replay.size,
                    incomplete_handoffs=self.incomplete,observed_terminal_frame_labels=self.final_frame_labels,
                    label_generator='native physics before reset; actor input unchanged')

    def install(self):
        compute=self.env.reward_manager.compute
        def reward_compute(dt):
            result=compute(dt)
            self.post_physics()
            return result
        self.env.reward_manager.compute=reward_compute
        step=self.native.step
        def wrapped_step(actions):
            self.record_prefix(self.obs)
            result=step(actions)
            self.obs=result[0]
            self.reset_ids(result[2].bool().nonzero(as_tuple=False).flatten())
            return result
        self.native.step=wrapped_step
        reset=self.native.reset
        def wrapped_reset():
            result=reset()
            self.reset_ids(self.rows);self.obs=result[0]
            return result
        self.native.reset=wrapped_reset
        return self
