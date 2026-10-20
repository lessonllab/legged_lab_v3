# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
"""G1 adaptation of ezoicoder/humanoid-gym active reward implementation.

Reference: commit 9a47b7dfe85fdcc737c3c8963d80d05997376575, custom/humanoid_env.py.
All 23 active scales below come from humanoid_config.py, not paper Table VII.
Adaptations: 12 named G1 legs use their bent default pose and mirrored G1 axes;
29 learned joints receive energy/velocity/action penalties. Near-zero commands
request double support and no swing/air-time bonuses. Stateful terms share one
pre-reward transition snapshot and reset independently per environment.
With beamdojo_upstream_aligned, both position rewards also evaluate all 29
controlled joints, as upstream evaluates its entire DOF vector. Upper-body
reference positions are the Unitree asset defaults; leg references retain the
G1 axis/default-pose adaptation above.
"""
import math
import torch

CYCLE_TIME = .64
REFERENCE_SCALE = .17
BASE_HEIGHT_TARGET = .78
SOLE_OFFSET = .035
# Body geometry adaptation: upstream's .89 m XBot base target / .05 m
# ankle-to-ground offset are replaced by Unitree G1's .78 m target and the
# G1 sole transform (.035 m). Reward coefficients and tolerances are unchanged.
TARGET_FEET_HEIGHT = .06
# G1 default FK gives .237 m foot/knee spacing, so retain the upstream
# lower bounds; no morphology evidence requires widening this interval.
MIN_FEET_DISTANCE = .2
MAX_FEET_DISTANCE = .5
MIN_KNEE_DISTANCE = .2
MAX_KNEE_DISTANCE = .25
MAX_CONTACT_FORCE = 700.
REWARD_WEIGHTS = {
    'joint_pos':1.6,'feet_clearance':1.,'feet_contact_number':1.2,
    'feet_air_time':1.,'foot_slip':-.05,'feet_distance':.2,'knee_distance':.2,
    'feet_contact_forces':-.01,'tracking_lin_vel':1.2,'tracking_ang_vel':1.1,
    'vel_mismatch_exp':.5,'low_speed':.2,'track_vel_hard':.5,
    'default_joint_pos':.5,'orientation':1.,'base_height':.2,'base_acc':.2,
    'action_smoothness':-.002,'torques':-1e-5,'dof_vel':-5e-4,'dof_acc':-1e-7,
    'collision':-1.,'foothold':1.,
}
LEG_NAMES = tuple(f'{side}_{joint}_joint' for side in ('left','right')
                  for joint in ('hip_yaw','hip_roll','hip_pitch','knee','ankle_pitch','ankle_roll'))


def tensor(value):
    return value if isinstance(value,torch.Tensor) else value.torch


def upstream_aligned(env):
    return bool(getattr(getattr(env,'cfg',None),'beamdojo_upstream_aligned',False))


def command_moving(command):
    # Explicit user adaptation: pure yaw counts as movement, truly near-zero
    # xy AND yaw requests double support and a non-swing reference.
    return (command[:,:2].norm(dim=-1)>.1)|(command[:,2].abs()>.1)


def phase(env):
    angle=env.episode_length_buf*env.step_dt*(2*math.pi/CYCLE_TIME)
    return torch.stack((angle.sin(),angle.cos()),dim=-1)


def stance_mask(sine, moving):
    stance=torch.stack((sine>=0,sine<0),dim=-1)
    stance[(sine.abs()<.1)|~moving]=True
    return stance


def reference_positions(default_legs,sine,moving):
    reference=default_legs.clone()
    # G1 both hips/ankles have the same axes, unlike XBot's mirrored legs.
    swing=torch.stack((-sine.clamp_max(0),sine.clamp_min(0)),dim=-1)
    swing *= ((sine.abs()>=.1)&moving)[:,None]
    for foot,base in enumerate((0,6)):
        reference[:,base+2]-=REFERENCE_SCALE*swing[:,foot]
        reference[:,base+3]+=2*REFERENCE_SCALE*swing[:,foot]
        reference[:,base+4]-=REFERENCE_SCALE*swing[:,foot]
    return reference


def distance_reward(distance,minimum,maximum):
    d_min=(distance-minimum).clamp(-.5,0.)
    d_max=(distance-maximum).clamp(0.,.5)
    return ((-d_min.abs()*100).exp()+(-d_max.abs()*100).exp())/2


def euler_xyzw(quat):
    x,y,z,w=quat.unbind(-1)
    roll=torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch=torch.asin((2*(w*y-z*x)).clamp(-1.,1.))
    return torch.stack((roll,pitch),dim=-1)


class RewardState:
    """Only this cache mutates air/height/previous-state histories each tick."""
    def __init__(self,env):
        self.tick=None
        self.dirty=True
        self.fresh=torch.ones(env.num_envs,device=env.device,dtype=torch.bool)
        self.updated=torch.full((env.num_envs,),-1,device=env.device,dtype=torch.long)
        self.air=torch.zeros(env.num_envs,2,device=env.device)
        self.height=torch.zeros_like(self.air)
        self.last_contact=torch.zeros_like(self.air,dtype=torch.bool)
        self.last_feet_z=torch.zeros_like(self.air)
        self.previous_root=torch.zeros(env.num_envs,6,device=env.device)
        self.previous_dof=torch.zeros_like(tensor(env.scene['robot'].data.joint_vel))
        self.previous_action=torch.zeros_like(env.action_manager.action)
        self.previous_previous_action=torch.zeros_like(self.previous_action)
        self.cache={}
        robot=env.scene['robot']
        self.legs=robot.find_joints(list(LEG_NAMES),preserve_order=True)[0]
        if len(self.legs)!=12 or len(set(self.legs))!=12:
            raise ValueError('G1 reward reference requires all 12 unique named leg joints')
        self.feet=robot.find_bodies(['left_ankle_roll_link','right_ankle_roll_link'],preserve_order=True)[0]
        self.knees=robot.find_bodies(['left_knee_link','right_knee_link'],preserve_order=True)[0]
        sensor=env.scene.sensors['contact_forces']
        self.contact_ids=sensor.find_bodies(['left_ankle_roll_link','right_ankle_roll_link'],preserve_order=True)[0]
        self.collision_ids=sensor.find_bodies(['pelvis'],preserve_order=True)[0]
        if len(self.feet)!=2 or len(self.knees)!=2 or len(self.contact_ids)!=2 or len(self.collision_ids)!=1:
            raise ValueError('G1 feet, knees and pelvis contact mappings must resolve exactly')

    def reset(self,ids):
        self.fresh[ids]=True
        self.updated[ids]=-1
        self.air[ids]=0.
        self.height[ids]=0.
        self.last_contact[ids]=False
        self.last_feet_z[ids]=0.
        self.previous_root[ids]=0.
        self.previous_dof[ids]=0.
        self.previous_action[ids]=0.
        self.previous_previous_action[ids]=0.
        self.dirty=True

    def prepare(self,env):
        if self.tick==env.common_step_counter and not self.dirty:
            return self.cache
        # Every environment advances on a new physics step. Only a same-tick
        # partial reset needs dynamic indices; avoid CUDA nonzero synchronization
        # and gather/scatter for the ordinary all-environment path.
        ids = (slice(None) if self.tick != env.common_step_counter else
               ((self.updated != env.common_step_counter) | self.fresh).nonzero().flatten())
        data=env.scene['robot'].data
        force=tensor(env.scene.sensors['contact_forces'].data.net_forces_w)
        q=tensor(data.joint_pos)
        q0=tensor(data.default_joint_pos)
        qd=tensor(data.joint_vel)
        pos=tensor(data.body_pos_w)[:,self.feet]
        foot_quat=tensor(data.body_quat_w)[:,self.feet]
        # Full foot rotation of the local ankle-to-sole (0,0,-.035) offset.
        # Isaac Sim 6 quaternion order is XYZW.
        feet_z=pos[...,2]-SOLE_OFFSET*(1-2*(foot_quat[...,0].square()+foot_quat[...,1].square()))
        contact=force[:,self.contact_ids,2]>5.
        command=env.command_manager.get_command('base_velocity')
        moving=command_moving(command)
        sine=phase(env)[:,0]
        stance=stance_mask(sine,moving)
        root_velocity=torch.cat((tensor(data.root_lin_vel_w),tensor(data.root_ang_vel_w)),dim=-1)
        action=env.action_manager.action
        # Fresh episodes use the first observed post-reset sole height/velocity
        # as the baseline, avoiding stale body transforms and reset acceleration.
        prior_root=torch.where(self.fresh[:,None],root_velocity,self.previous_root)
        prior_dof=torch.where(self.fresh[:,None],qd,self.previous_dof)
        prior_z=torch.where(self.fresh[:,None],feet_z,self.last_feet_z)
        filt=contact|stance|self.last_contact
        first=(self.air>0)&filt
        elapsed=self.air+env.step_dt
        air_reward=(elapsed.clamp(0,.5)*first).sum(-1)*moving
        height=self.height+feet_z-prior_z
        clearance=((height-TARGET_FEET_HEIGHT).abs()<.01).logical_and(~stance).sum(-1).float()*moving
        reference=reference_positions(q0[:,self.legs],sine,moving)
        reference_all=q0.clone()
        reference_all[:,self.legs]=reference
        values={
            'q':q,'q0':q0,'qd':qd,'legs':q[:,self.legs],
            'default_legs':q0[:,self.legs],
            'reference':reference,'reference_all':reference_all,
            'stance':stance,'contact':contact,'feet':pos,'sole_z':feet_z,
            'knees':tensor(data.body_pos_w)[:,self.knees],
            'foot_velocity':tensor(data.body_link_lin_vel_w)[:,self.feet],
            'foot_force':force[:,self.contact_ids],
            'collision_force':force[:,self.collision_ids],
            'command':command,'linear':tensor(data.root_lin_vel_b),
            'angular':tensor(data.root_ang_vel_b),'gravity':tensor(data.projected_gravity_b),
            'euler':euler_xyzw(tensor(data.root_quat_w)),
            'root_z':tensor(data.root_pos_w)[:,2],
            'root_delta':prior_root-root_velocity,'dof_delta':(prior_dof-qd)/env.step_dt,
            'torques':tensor(data.applied_torque),'air_reward':air_reward,'clearance':clearance,
            'smoothness':((self.previous_action-action).square().sum(-1)
                         +(action+self.previous_previous_action-2*self.previous_action).square().sum(-1)
                         +.05*action.abs().sum(-1)),
        }
        for key,value in values.items():
            if key not in self.cache:
                self.cache[key]=value.clone()
            else:
                if isinstance(ids, slice):
                    self.cache[key].copy_(value)
                else:
                    self.cache[key][ids]=value[ids]
        self.air[ids]=(elapsed*~filt)[ids]
        self.height[ids]=(height*~contact)[ids]
        self.last_contact[ids]=contact[ids]
        self.last_feet_z[ids]=feet_z[ids]
        self.previous_root[ids]=root_velocity[ids]
        self.previous_dof[ids]=qd[ids]
        self.previous_previous_action[ids]=self.previous_action[ids]
        self.previous_action[ids]=action[ids]
        self.fresh[ids]=False
        self.updated[ids]=env.common_step_counter
        self.tick=env.common_step_counter
        self.dirty=False
        return self.cache


def state(env):
    if not hasattr(env,'_repo_reward_state'):
        env._repo_reward_state=RewardState(env)
    return env._repo_reward_state.prepare(env)


def reset_state(env,ids):
    if hasattr(env,'_repo_reward_state'):
        env._repo_reward_state.reset(ids)


def joint_pos(env):
    s=state(env)
    diff=(s['q']-s['reference_all'] if upstream_aligned(env)
          else s['legs']-s['reference']).norm(dim=-1)
    return (-2*diff).exp()-.2*diff.clamp(0,.5)


def feet_distance(env):
    p=state(env)['feet'][...,:2]
    return distance_reward((p[:,0]-p[:,1]).norm(dim=-1),MIN_FEET_DISTANCE,MAX_FEET_DISTANCE)


def knee_distance(env):
    p=state(env)['knees'][...,:2]
    return distance_reward((p[:,0]-p[:,1]).norm(dim=-1),MIN_KNEE_DISTANCE,MAX_KNEE_DISTANCE)


def foot_slip(env):
    s=state(env)
    return (s['foot_velocity'][...,:2].norm(dim=-1).sqrt()*s['contact']).sum(-1)


def feet_air_time(env): return state(env)['air_reward']


def feet_contact_number(env):
    s=state(env)
    return torch.where(s['contact']==s['stance'],1.,-.3).mean(-1)


def orientation(env):
    s=state(env)
    return ((-s['euler'].abs().sum(-1)*10).exp()+(-s['gravity'][:,:2].norm(dim=-1)*20).exp())/2


def feet_contact_forces(env):
    return (state(env)['foot_force'].norm(dim=-1)-MAX_CONTACT_FORCE).clamp(0,400).sum(-1)


def default_joint_pos(env):
    s=state(env); diff=s['legs']-s['default_legs']
    yaw_roll=(diff[:,:2].norm(dim=-1)+diff[:,6:8].norm(dim=-1)-.1).clamp(0,50)
    total_diff=s['q']-s['q0'] if upstream_aligned(env) else diff
    return (-yaw_roll*100).exp()-.01*total_diff.norm(dim=-1)


def base_height(env):
    s=state(env); stance=s['stance'].float()
    ground=(s['sole_z']*stance).sum(-1)/stance.sum(-1)
    height=s['root_z']-ground
    return (-(height-BASE_HEIGHT_TARGET).abs()*100).exp()


def base_acc(env): return (-state(env)['root_delta'].norm(dim=-1)*3).exp()


def vel_mismatch_exp(env):
    s=state(env)
    return ((-s['linear'][:,2].square()*10).exp()+(-s['angular'][:,:2].norm(dim=-1)*5).exp())/2


def track_vel_hard(env):
    s=state(env)
    linear=(s['command'][:,:2]-s['linear'][:,:2]).norm(dim=-1)
    angular=(s['command'][:,2]-s['angular'][:,2]).abs()
    return ((-linear*10).exp()+(-angular*10).exp())/2-.2*(linear+angular)


def tracking_lin_vel(env):
    s=state(env)
    return (-(s['command'][:,:2]-s['linear'][:,:2]).square().sum(-1)*5).exp()


def tracking_ang_vel(env):
    s=state(env)
    return (-(s['command'][:,2]-s['angular'][:,2]).square()*5).exp()


def feet_clearance(env): return state(env)['clearance']


def low_speed(env):
    s=state(env); velocity=s['linear'][:,0]; command=s['command'][:,0]
    low=velocity.abs()<.5*command.abs(); high=velocity.abs()>1.2*command.abs()
    reward=torch.where(low,-1.,torch.where(high,0.,1.2))
    reward=torch.where(velocity.sign()!=command.sign(),-2.,reward)
    return reward*(command.abs()>.1)


def torques(env): return state(env)['torques'].square().sum(-1)


def dof_vel(env): return state(env)['qd'].square().sum(-1)


def dof_acc(env): return state(env)['dof_delta'].square().sum(-1)


def collision(env): return (state(env)['collision_force'].norm(dim=-1)>.1).float().sum(-1)


def action_smoothness(env): return state(env)['smoothness']


def foothold(env):
    from .geometry import g1_sole_points,transform_sole_points
    from .paper_mdp import surface_height
    state(env)
    data=env.scene['robot'].data; ids=env._repo_reward_state.feet
    if not hasattr(env, '_repo_sole_points'):
        env._repo_sole_points = g1_sole_points(device=env.device)
    soles=transform_sole_points(tensor(data.body_pos_w)[:,ids],
        tensor(data.body_quat_w)[:,ids][...,[3,0,1,2]],env._repo_sole_points)
    heights,_=surface_height(env,soles)
    unsafe=heights-tensor(env.scene.env_origins)[:,None,2:3]<-.1
    return -(unsafe&state(env)['contact'][...,None]).sum(dim=(-1,-2)).float()
