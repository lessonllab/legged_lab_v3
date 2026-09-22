"""Opt-in turning practice and a larger non-stair rehearsal allocation."""
import math
import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, yaw_quat, wrap_to_pi
from .style_state import tensor


def agility_layout(n, names, device):
    # 25% flat (one third pure turns), 25% ascent, 20% descent, 30% other.
    q = (torch.arange(n, device=device) + .5) / n
    groups = torch.bucketize(q, torch.tensor([.25,.50,.70], device=device))
    columns = torch.zeros(n,device=device,dtype=torch.long)
    for group, allowed in ((0,set(names)), (1,{'pyramid_stairs_inv'}), (2,{'pyramid_stairs'})):
        choices = [i for i,name in enumerate(names) if name in allowed]
        ids = torch.where(groups == group)[0]
        columns[ids] = torch.tensor(choices,device=device)[torch.arange(len(ids),device=device)%len(choices)]
    ids = torch.where(groups == 3)[0]
    rough = [i for i,name in enumerate(names) if name == 'random_rough']
    others = [i for i,name in enumerate(names) if name not in ('pyramid_stairs','pyramid_stairs_inv','random_rough')]
    if not rough or not others:
        raise ValueError('Agility course requires rough terrain and other terrain columns')
    # Half of the 30% other pool is rough, the remainder covers boxes/slopes.
    for selected, choices in ((ids[::2],rough),(ids[1::2],others)):
        columns[selected] = torch.tensor(choices,device=device)[torch.arange(len(selected),device=device)%len(choices)]
    return groups, columns


def rough_outcomes(xy, yaw, progress, opportunity, arrivals, failed):
    finite = torch.isfinite(xy+yaw+progress+opportunity+arrivals)
    passed = finite & ~failed & (xy>.6) & (yaw>.5) & (progress>=1.) & (opportunity>=2.) & (arrivals>=1.)
    bad = failed | ~finite | ((opportunity>=2.) & ((xy<.3) | (yaw<.3)))
    return passed, bad


def flat_reverse_command(position, quat, target, speed_cap):
    """Flat shuttle with a fixed world +X heading: back up, then return forward.

    A rear target must produce negative vx without commanding a 180-degree turn.
    Small lateral correction keeps the shuttle near its center line. Heading
    alignment gates translation so large orientation errors are corrected first.
    """
    delta = target-position
    heading = wrap_to_pi(-euler_xyz_from_quat(quat)[2])
    world_velocity = torch.zeros_like(delta)
    world_velocity[:,0] = torch.maximum(torch.minimum(2.*delta[:,0],speed_cap),-speed_cap)
    world_velocity[:,1] = (delta[:,1]*.5).clamp(-.15,.15)
    velocity = quat_apply_inverse(yaw_quat(quat),world_velocity)
    gate = ((torch.pi*70/180-heading.abs())/(torch.pi*50/180)).clamp(0.,1.)
    velocity[:,:2] *= gate[:,None]
    velocity[:,2] = (heading*2.).clamp(-1.,1.)
    reached = delta[:,:2].norm(dim=-1) <= .4
    velocity[reached] = 0.
    return velocity,heading,reached


class AgilityCommandMixin:
    initial_layout = staticmethod(agility_layout)
    layout_profile = 'agility_v10'
    guarded_rough_curriculum = True

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.turn_pool = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        flat_ids = torch.where(self.flat_pool)[0]
        self.turn_pool[flat_ids[::3]] = True
        self.reverse_pool = torch.zeros_like(self.turn_pool)
        self.reverse_pool[flat_ids[1::3]] = True
        self.reverse_goal_sign = -torch.ones(self.num_envs,device=self.device)
        self.reverse_speed = torch.zeros(self.num_envs,device=self.device)
        self.turn_rate = torch.zeros(self.num_envs,device=self.device)
        for name in ('turn_seconds','turn_tracking','turn_drift_mps',
                     'reverse_seconds','reverse_tracking','reverse_actual_vx','reverse_command_vx'):
            self.metrics[name] = torch.zeros(self.num_envs,device=self.device)
        print('[Agility] 25/25/20/30% flat/up/down/other; 15% rough; ~8.3% pure turns; '
              '~8.3% flat reverse/forward shuttles at 0.2-0.6 m/s; '
              'flat goals include 90/180 degree changes; two safe arrivals-based episodes to promote rough terrain',flush=True)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if not hasattr(self,'turn_pool'):
            return
        ids = self._ids(env_ids)
        ids = ids[self.flat_pool[ids] & ~self.manual_target[ids]]
        if not len(ids): return
        self.is_standing_env[ids] = False
        turning = ids[self.turn_pool[ids]]
        self.turn_rate[turning] = (.3+.7*torch.rand(len(turning),device=self.device)) * torch.where(
            torch.rand(len(turning),device=self.device)<.5,-1.,1.)
        reverse = ids[self.reverse_pool[ids]]
        local_x = tensor(self.robot.data.root_pos_w)[reverse,0]-tensor(self.terrain.env_origins)[reverse,0]
        sign = self.reverse_goal_sign[reverse]
        at_end = (local_x-2.5*sign).abs()<.45
        sign = torch.where(at_end,-sign,sign)
        # Every new physical episode begins by backing up from the center.
        sign = torch.where(tensor(self._env.episode_length_buf)[reverse]==0,-1.,sign)
        self.reverse_goal_sign[reverse] = sign
        self.reverse_speed[reverse] = .2+.4*torch.rand(len(reverse),device=self.device)
        self.speed_cap[reverse] = self.reverse_speed[reverse]
        self.pos_command_w[reverse] = tensor(self.terrain.env_origins)[reverse]
        self.pos_command_w[reverse,0] += 2.5*sign
        walking = ids[~self.turn_pool[ids] & ~self.reverse_pool[ids]]
        # Deliberately exercise front, left/right and rear goals on flat ground.
        yaw = euler_xyz_from_quat(tensor(self.robot.data.root_quat_w)[walking])[2]
        angles = torch.tensor([0.,math.pi/2,-math.pi/2,math.pi],device=self.device)
        angle = yaw + angles[torch.randint(4,(len(walking),),device=self.device)]
        origin = tensor(self.terrain.env_origins)[walking]
        current = tensor(self.robot.data.root_pos_w)[walking]
        xy = current[:,:2]+2.5*torch.stack((angle.cos(),angle.sin()),-1)
        self.pos_command_w[walking] = origin
        self.pos_command_w[walking,:2] = origin[:,:2]+(xy-origin[:,:2]).clamp(-3.,3.)
        # Never count a clipped, already-reached target as a successful maneuver.
        close = (self.pos_command_w[walking,:2]-current[:,:2]).norm(dim=-1)<.8
        self.pos_command_w[walking[close],:2] = origin[close,:2]
        self._begin_target(ids)
        if len(reverse) and not getattr(self,'_reverse_reported',False):
            self._reverse_reported = True
            vx = self.vel_command_b[reverse,0]
            print(f'[FlatReverse] {len(reverse)} flat agents; initial negative-vx commands '
                  f'{int((vx<0).sum())}; vx range {float(vx.min()):.3f}..{float(vx.max()):.3f} m/s; '
                  'stair height adaptation remains enabled',flush=True)

    def _update_selected(self, ids):
        super()._update_selected(ids)
        if not hasattr(self,'turn_pool'): return
        ids = self._ids(ids)
        turning = ids[self.turn_pool[ids] & ~self.manual_target[ids]]
        self.vel_command_b[turning] = 0.
        self.vel_command_b[turning,2] = self.turn_rate[turning]
        self.heading_command_w[turning] = 0. # pure rate command has no heading target
        self.reached_target[turning] = False
        if hasattr(self,'reverse_pool'):
            reverse = ids[self.reverse_pool[ids] & ~self.manual_target[ids]]
            data = self.robot.data
            velocity,heading,reached = flat_reverse_command(tensor(data.root_pos_w)[reverse],
                tensor(data.root_quat_w)[reverse],self.pos_command_w[reverse],self.reverse_speed[reverse])
            self.vel_command_b[reverse] = velocity
            self.heading_command_w[reverse] = heading
            self.reached_target[reverse] = reached

    def _update_metrics(self):
        if hasattr(self,'turn_pool'):
            active = self.turn_pool & ~self.manual_target & ~self._skip_progress_transition
            dt = active.float()*self._env.step_dt
            old = self.metrics['turn_seconds']
            total = old+dt
            yaw = tensor(self.robot.data.root_ang_vel_w)[:,2]
            drift = tensor(self.robot.data.root_link_lin_vel_w)[:,:2].norm(dim=-1)
            for name,value in (('turn_tracking',torch.exp(-(yaw-self.turn_rate).square()/.25)),('turn_drift_mps',drift)):
                self.metrics[name][:] = (self.metrics[name]*old+torch.where(active,value,0.)*dt)/total.clamp_min(self._env.step_dt)
            old.copy_(total)
            if hasattr(self,'reverse_pool'):
                from .locomotion_progress import yaw_frame_velocity
                active_reverse = (self.reverse_pool & ~self.manual_target & ~self._skip_progress_transition
                                  & (self.vel_command_b[:,0]<-.05))
                dt_reverse = active_reverse.float()*self._env.step_dt
                old_reverse = self.metrics['reverse_seconds']
                total_reverse = old_reverse+dt_reverse
                actual = yaw_frame_velocity(self.robot.data)
                score = torch.exp(-(actual[:,:2]-self.vel_command_b[:,:2]).square().sum(-1)/.25)
                for name,value in (('reverse_tracking',score),('reverse_actual_vx',actual[:,0]),
                                   ('reverse_command_vx',self.vel_command_b[:,0])):
                    self.metrics[name][:] = (self.metrics[name]*old_reverse+
                        torch.where(active_reverse,value,0.)*dt_reverse)/total_reverse.clamp_min(self._env.step_dt)
                old_reverse.copy_(total_reverse)
                self._skip_progress_transition |= self.reverse_pool & ~self.manual_target
            # Pure turns must not earn translation/arrival curriculum evidence.
            self._skip_progress_transition |= self.turn_pool & ~self.manual_target
        super()._update_metrics()
