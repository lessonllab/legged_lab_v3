"""Target-patch commands adapted from InstinctLab parkour (CC BY-NC 4.0).

Source: https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/mdp/commands/pose_velocity_command.py
Adaptation: local terrain names, deferred imports, Isaac Lab 3 ProxyArrays,
reset-time command refresh, and goal metrics. No InstinctLab runtime dependency.
"""
import numpy as np
import torch
from isaaclab.managers import CommandTerm
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, wrap_to_pi, euler_xyz_from_quat
from legged_lab.tasks.locomotion.amp.mdp.style_state import tensor


def terrain_column_names(generator):
    """Use the exact ordered-column assignment used by Isaac Lab terrain generation."""
    if not generator.curriculum:
        raise ValueError('Target commands require ordered terrain columns (generator.curriculum=True).')
    names = list(generator.sub_terrains)
    weights = np.array([x.proportion for x in generator.sub_terrains.values()], dtype=float)
    edges = np.cumsum(weights / weights.sum())
    return [names[np.flatnonzero(i / generator.num_cols + .001 < edges)[0]]
            for i in range(generator.num_cols)]


def target_velocity(position, quat, target, speed_cap, standing, distance_threshold=.4,
                    velocity_gain=2., heading_gain=2., yaw_limit=1., heading_slowdown=False):
    """Instinct's forward-only target controller, with zero lateral velocity."""
    delta = target - position
    local = quat_apply_inverse(yaw_quat(quat), delta)
    heading = wrap_to_pi(torch.atan2(delta[:, 1], delta[:, 0]) - euler_xyz_from_quat(quat)[2])
    velocity = torch.zeros_like(delta)
    velocity[:, 0] = torch.minimum((local[:, 0] * velocity_gain).clamp_min(0.), speed_cap)
    if heading_slowdown:
        # Full speed below 20 degrees; smooth slowdown; turn first above 70.
        velocity[:, 0] *= ((torch.pi*70/180-heading.abs())/(torch.pi*50/180)).clamp(0.,1.)
    velocity[:, 2] = (heading * heading_gain).clamp(-yaw_limit, yaw_limit)
    distance = delta[:, :2].norm(dim=-1)
    reached = distance <= distance_threshold
    velocity[reached | standing] = 0.
    return velocity, heading, distance, reached


class TargetVelocityCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.terrain = env.scene.terrain
        self.column_names = terrain_column_names(self.terrain.cfg.terrain_generator)
        missing = set(self.column_names) - cfg.speed_ranges.keys()
        if missing:
            raise ValueError(f'Missing terrain speed ranges: {missing}')
        self.speed_ranges = torch.tensor([cfg.speed_ranges[n] for n in self.column_names], device=self.device)
        self.valid_targets = tensor(self.terrain.flat_patches['target'])
        if not torch.isfinite(self.valid_targets).all():
            raise ValueError('Target patches contain non-finite coordinates')
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.speed_cap = torch.zeros(self.num_envs, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Opt-in playback control; training leaves every entry false.
        self.manual_target = torch.zeros_like(self.is_standing_env)
        self.reached_target = torch.zeros_like(self.is_standing_env)
        self.target_initial_distance = torch.zeros(self.num_envs, device=self.device)
        self.target_best_progress = torch.zeros_like(self.target_initial_distance)
        self._target_valid = torch.zeros_like(self.is_standing_env)
        self._target_valid_reached = torch.zeros_like(self.is_standing_env)
        self._skip_progress_transition = torch.ones_like(self.is_standing_env)
        for name in ('error_vel_xy', 'error_vel_yaw', 'tracking_exp_vel_xy', 'tracking_exp_vel_yaw',
                     'target_distance', 'targets_reached', 'stopped_fraction', 'valid_targets_reached',
                     'target_progress_m', 'moving_opportunity_s', 'moving_tracking_exp_vel_xy',
                     'moving_tracking_exp_vel_yaw'):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        print('[TargetCommand] resample=8-12 s; gains=(2,2); yaw=+-1 rad/s; stop_radius=0.4 m; '
              f'standing={cfg.rel_standing_envs:.0%}; speed_caps={cfg.speed_ranges}', flush=True)

    @property
    def command(self):
        return self.vel_command_b

    def _ids(self, env_ids):
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def reset(self, env_ids=None):
        # The installed CommandTerm.reset(None) forwards a slice to _resample,
        # which requires len(env_ids); normalize before entering the base method.
        ids = self._ids(env_ids)
        # The environment has already reset/teleported the robot before this
        # call. Its first subsequent compute observes no physics transition.
        self._skip_progress_transition[ids] = True
        self.manual_target[ids] = False
        return super().reset(ids)

    def set_manual_target(self, env_id, position_w):
        """Pin one playback goal until released or this environment resets.

        Call on the simulation thread, never from a viewer callback thread.
        """
        if not 0 <= env_id < self.num_envs:
            raise ValueError('Invalid target environment index')
        point = torch.as_tensor(position_w, device=self.device, dtype=self.pos_command_w.dtype)
        if point.shape != (3,) or not torch.isfinite(point).all():
            raise ValueError('Manual target must be one finite world XYZ point')
        ids = self._ids([env_id])
        self.pos_command_w[ids] = point
        self.manual_target[ids] = True
        self.is_standing_env[ids] = False
        columns = tensor(self.terrain.terrain_types)[ids].long()
        limits = self.speed_ranges[columns]
        self.speed_cap[ids] = self.speed_cap[ids].clamp(limits[:, 0], limits[:, 1])
        self._begin_target(ids)

    def release_manual_target(self, env_id):
        """Return only the manually controlled robot to normal random sampling."""
        ids = self._ids([env_id])
        ids = ids[self.manual_target[ids]]
        if len(ids):
            self.manual_target[ids] = False
            self._resample(ids)

    def _resample_command(self, env_ids):
        ids = self._ids(env_ids)
        ids = ids[~self.manual_target[ids]]
        if not len(ids):
            return
        columns = tensor(self.terrain.terrain_types)[ids].long()
        levels = tensor(self.terrain.terrain_levels)[ids].long()
        patches = torch.randint(self.valid_targets.shape[2], (len(ids),), device=self.device)
        # Patches already contain world coordinates; do not add env_origins again.
        self.pos_command_w[ids] = self.valid_targets[levels, columns, patches]
        limits = self.speed_ranges[columns]
        self.speed_cap[ids] = limits[:, 0] + torch.rand(len(ids), device=self.device) * (limits[:, 1] - limits[:, 0])
        self.is_standing_env[ids] = torch.rand(len(ids), device=self.device) < self.cfg.rel_standing_envs
        self._begin_target(ids)

    def _begin_target(self, ids):
        """Establish progress baselines and refresh only the retargeted commands."""
        self.reached_target[ids] = False
        initial_distance = (self.pos_command_w[ids, :2] - tensor(self.robot.data.root_pos_w)[ids, :2]).norm(dim=-1)
        self.target_initial_distance[ids] = torch.nan_to_num(initial_distance, nan=0., posinf=0., neginf=0.)
        self.target_best_progress[ids] = 0.
        # A target that is effectively at the spawn position offers no evidence
        # of locomotion. Ordinary retargeting establishes a fresh baseline while
        # retaining the completed episode's progress metrics.
        margin = getattr(self.cfg, 'target_initial_distance_margin', .2)
        self._target_valid[ids] = (
            torch.isfinite(initial_distance)
            & (initial_distance > self.cfg.target_dis_threshold + margin)
            & ~self.is_standing_env[ids])
        self._target_valid_reached[ids] = False
        # CommandManager.reset() does not call _update_command before observations.
        self._update_selected(ids)

    def _update_selected(self, ids):
        data = self.robot.data
        velocity, heading, _, reached = target_velocity(
            tensor(data.root_pos_w)[ids], tensor(data.root_quat_w)[ids], self.pos_command_w[ids],
            self.speed_cap[ids], self.is_standing_env[ids], self.cfg.target_dis_threshold,
            self.cfg.velocity_control_stiffness, self.cfg.heading_control_stiffness, self.cfg.yaw_limit,
            getattr(self.cfg,'heading_slowdown',False))
        self.vel_command_b[ids] = velocity
        self.heading_command_w[ids] = heading
        first_reach = reached & ~self.reached_target[ids] & ~self.is_standing_env[ids]
        self.metrics['targets_reached'][ids] += first_reach.float()
        self.reached_target[ids] |= reached

    def _update_command(self):
        self._update_selected(slice(None))

    def _update_metrics(self):
        # Measure the command applied during the transition, before retargeting it.
        actual = tensor(self.robot.data.root_lin_vel_b)
        angular = tensor(self.robot.data.root_ang_vel_b)
        error_xy = (self.vel_command_b[:, :2] - actual[:, :2]).square().sum(-1)
        error_yaw = (self.vel_command_b[:, 2] - angular[:, 2]).square()
        horizon = self._env.max_episode_length
        self.metrics['error_vel_xy'] += error_xy.sqrt() / horizon
        self.metrics['error_vel_yaw'] += error_yaw.sqrt() / horizon
        # Full-horizon normalization is intentional: short-lived episodes cannot
        # earn a high curriculum score merely by tracking well before a fall.
        self.metrics['tracking_exp_vel_xy'] += torch.exp(-error_xy / .25) / horizon
        self.metrics['tracking_exp_vel_yaw'] += torch.exp(-error_yaw / .25) / horizon
        distance = (self.pos_command_w[:, :2] - tensor(self.robot.data.root_pos_w)[:, :2]).norm(dim=-1)
        self.metrics['target_distance'] += distance / horizon
        self.metrics['stopped_fraction'] += (self.vel_command_b.norm(dim=-1) == 0).float() / horizon
        self._update_progress_metrics(distance, error_xy, error_yaw)

    def _update_progress_metrics(self, distance, error_xy, error_yaw):
        """Track real approach and command tracking without arrival/standing dwell.

        Unlike the legacy metrics above, moving tracking scores are conditional
        means over actual moving-target opportunities. Curriculum separately
        requires sufficient opportunity, physical progress, and a timeout. Max
        net approach to a single target is used instead of summing path length,
        so oscillating or repeatedly resampling cannot fabricate progress.
        """
        finite = torch.isfinite(distance) & torch.isfinite(error_xy) & torch.isfinite(error_yaw)
        transition = ~self._skip_progress_transition
        episode_steps = getattr(self._env, 'episode_length_buf', None)
        if episode_steps is not None:
            transition &= tensor(episode_steps) > 0
        self._skip_progress_transition[:] = False
        active = (transition & finite & self._target_valid & ~self.is_standing_env
                  & ~self._target_valid_reached & (self.vel_command_b.norm(dim=-1) > 0.))
        dt = float(self._env.step_dt)
        old_time = self.metrics['moving_opportunity_s']
        new_time = old_time + active.float() * dt
        for name, error in (('moving_tracking_exp_vel_xy', error_xy),
                            ('moving_tracking_exp_vel_yaw', error_yaw)):
            score = torch.exp(-torch.nan_to_num(error, nan=float('inf'), posinf=float('inf')) / .25)
            mean = (self.metrics[name] * old_time + score * active.float() * dt) / new_time.clamp_min(dt)
            self.metrics[name].copy_(mean)
        old_time.copy_(new_time)

        progress = (self.target_initial_distance - torch.nan_to_num(
            distance, nan=float('inf'), posinf=float('inf'))).clamp_min(0.)
        progress = torch.where(active, progress, 0.)
        self.target_best_progress.copy_(torch.maximum(self.target_best_progress, progress))
        self.metrics['target_progress_m'].copy_(torch.maximum(
            self.metrics['target_progress_m'], self.target_best_progress))
        valid_arrival = (active & (distance <= self.cfg.target_dis_threshold)
                         & (self.target_best_progress >= getattr(self.cfg, 'target_min_arrival_progress', .2)))
        self.metrics['valid_targets_reached'] += valid_arrival.float()
        self._target_valid_reached |= valid_arrival


def target_tracking_curriculum(env, env_ids):
    """Instinct thresholds: promote xy>0.6 and yaw>0; demote xy<0.3."""
    command = env.command_manager.get_term('base_velocity')
    xy = command.metrics['tracking_exp_vel_xy'][env_ids]
    yaw = command.metrics['tracking_exp_vel_yaw'][env_ids]
    up = (xy > .6) & (yaw > 0.)
    down = (xy < .3) & ~up
    env.scene.terrain.update_env_origins(env_ids, up, down)
    return tensor(env.scene.terrain.terrain_levels).float().mean()


def target_progress_curriculum(env, env_ids, command_name='base_velocity', xy_threshold=.6,
                               yaw_threshold=.5, demote_xy_threshold=.3, demote_yaw_threshold=.3,
                               min_opportunity_s=2., min_progress_m=1.):
    """Promote successful movement; standing and arrival dwell earn no credit.

    This is deliberately stricter than the legacy v4/Instinct score-only rule.
    The environment invokes curriculum before scene reset, so command metrics
    still belong to the completed episode. Reset flags must distinguish a true
    timeout from a fall, including a fall on the final episode step.
    """
    command = env.command_manager.get_term(command_name)
    ids = command._ids(env_ids)
    metrics = command.metrics
    xy = metrics['moving_tracking_exp_vel_xy'][ids]
    yaw = metrics['moving_tracking_exp_vel_yaw'][ids]
    opportunity = metrics['moving_opportunity_s'][ids]
    progress = metrics['target_progress_m'][ids]
    arrivals = metrics['valid_targets_reached'][ids]
    finite = torch.isfinite(xy) & torch.isfinite(yaw) & torch.isfinite(opportunity) & torch.isfinite(progress)
    # Manager buffers exist before the first reset; the env.reset_* aliases
    # are only created by step(), which is too late for initial curriculum.
    terminated = tensor(env.termination_manager.terminated)[ids].bool()
    timeout = tensor(env.termination_manager.time_outs)[ids].bool()
    sufficient_opportunity = opportunity >= min_opportunity_s
    moved = (progress >= min_progress_m) | (arrivals > 0.)
    up = (finite & sufficient_opportunity & moved & (xy > xy_threshold) & (yaw > yaw_threshold)
          & timeout & ~terminated)
    poor_tracking = sufficient_opportunity & ((xy < demote_xy_threshold) | (yaw < demote_yaw_threshold))
    down = (opportunity > 0.) & (terminated | poor_tracking | ~finite) & ~up
    env.scene.terrain.update_env_origins(ids, up, down)
    return tensor(env.scene.terrain.terrain_levels).float().mean()
