"""Stair specialization after balanced locomotion; task rewards stay unchanged.

Unlike Instinct's tracking curriculum, stairs use measured traversal/settling.
Group 0/1/2/3 = flat/ascent/descent/other. Pyramid pits ascend outwards.
"""
import math
import torch
from .balanced_course import BalancedTargetCommand, balanced_curriculum
from .style_state import tensor

STAIR_HEIGHTS = (0., .08, .10, .12, .14, .16, .18, .20, .25, .30)
ENV_STATE = ('group', 'flat_pool', 'flat_speed_tier', 'speed_good', 'speed_bad',
             'success_streak', 'failure_streak', 'stair_phase', 'stair_epoch')
GLOBAL_STATE = ('frontier', 'phase', 'epoch', 'attempts', 'wins', 'last_rate')
STAIR_SPEED_RANGES = ((.35, .55), (.55, .75))


def specialist_layout(n, names, device):
    # Midpoint allocation gives the requested ratios to within one environment.
    q = (torch.arange(n, device=device) + .5) / n
    group = torch.bucketize(q, torch.tensor([.25, .60, .80], device=device))
    columns = torch.zeros(n, device=device, dtype=torch.long)
    for g, kinds in enumerate((set(names), {'pyramid_stairs_inv'}, {'pyramid_stairs'},
                               set(names) - {'pyramid_stairs', 'pyramid_stairs_inv'})):
        choices = torch.tensor([i for i, name in enumerate(names) if name in kinds], device=device)
        ids = torch.where(group == g)[0]
        if not len(choices):
            raise ValueError('Stair specialization requires stairs, inverse stairs and other terrain columns')
        columns[ids] = choices[torch.arange(len(ids), device=device) % len(choices)]
    return group, columns


def stair_speed_ranges(phase, schedule=STAIR_SPEED_RANGES):
    return torch.tensor(schedule, device=phase.device)[phase.long()]


def advance_stage(level, phase, wins, attempts, num_phases=2):
    """A window of >=50 current-stage attempts, with neutral hysteresis."""
    if attempts < 50:
        return level, phase, False
    rate = wins / attempts
    if rate > .8:
        if phase < num_phases - 1:
            phase += 1
        elif level < 9:
            level, phase = level + 1, 0
    elif rate < .5:
        if phase > 0:
            phase -= 1
        else:
            level = max(1, level - 1)
    return level, phase, True


def sample_stair_stages(draw, level, phase, num_phases, challenge=False):
    """20% rehearsal; optionally 10% faster and 10% higher, one axis at a time."""
    rehearsal = draw < .2
    levels = torch.full_like(draw, level, dtype=torch.long)
    phases = torch.full_like(draw, phase, dtype=torch.long)
    levels[rehearsal] = max(1, level - 1)
    phases[rehearsal] = 0
    if challenge:
        phases[(draw >= .2) & (draw < .3)] = min(num_phases - 1, phase + 1)
        levels[(draw >= .3) & (draw < .4)] = min(len(STAIR_HEIGHTS) - 1, level + 1)
    return levels, phases


def exit_geometry(root, feet, goal, gravity, half_size):
    """Both ankles on the outer platform, body upright, low velocity at goal.

    Checking feet against the exit plane prevents horizontal proximity below a
    stair from counting as completion. This is independent of reached_target.
    """
    near = (root[:, :2] - goal[:, :2]).norm(dim=-1) <= .4
    on_exit = (feet[:, :, 0] >= goal[:, None, 0] - .6).all(-1)
    on_exit &= ((feet[:, :, 2] - goal[:, None, 2]).abs() < .12).all(-1)
    in_bounds = (feet[:, :, 0] < goal[:, None, 0] + .3).all(-1)
    in_bounds &= ((feet[:, :, 1] - goal[:, None, 1]).abs() < half_size - .05).all(-1)
    upright = gravity[:, 2] < -.90
    height = root[:, 2] - goal[:, 2]
    return near & on_exit & in_bounds & upright & (height > .4) & (height < 1.1)


def settled_at_exit(root, feet, goal, velocity, angular_velocity, gravity, half_size):
    """Legacy strict stop-quality diagnostic; no longer controls promotion."""
    return (exit_geometry(root, feet, goal, gravity, half_size)
            & (velocity[:, :2].norm(dim=-1) < .2) & (angular_velocity.norm(dim=-1) < .3))


class ExitWindow:
    """One-second rolling window tolerating brief exploration perturbations.

    Average speed is the average of magnitudes, not a vector average that could
    cancel oscillation. A full post-reset window is mandatory. Angular velocity
    is intentionally absent; upright/foot geometry still constrain safety.
    """
    def __init__(self, num_envs, dt, device):
        self.length = max(1, math.ceil(1. / dt))
        self.geometry = torch.zeros(self.length, num_envs, dtype=torch.bool, device=device)
        self.speed = torch.zeros(self.length, num_envs, device=device)
        self.geometry_sum = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.speed_sum = torch.zeros(num_envs, device=device)
        self.samples = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cursor = 0

    def reset(self, ids):
        self.geometry[:, ids] = False
        self.speed[:, ids] = 0.
        self.geometry_sum[ids] = 0
        self.speed_sum[ids] = 0.
        self.samples[ids] = 0

    def update(self, geometry, speed, eligible):
        geometry = geometry & eligible
        # Finite sentinel avoids poisoning later rolling sums with inf-inf.
        speed = torch.where(eligible, torch.nan_to_num(speed, nan=1e4, posinf=1e4, neginf=1e4), 0.)
        i = self.cursor
        self.geometry_sum += geometry.long() - self.geometry[i].long()
        self.speed_sum += speed - self.speed[i]
        self.geometry[i] = geometry
        self.speed[i] = speed
        self.samples[:] = torch.where(eligible, (self.samples + 1).clamp_max(self.length), 0)
        self.cursor = (i + 1) % self.length
        return (geometry & (self.samples >= self.length)
                & (self.geometry_sum >= math.ceil(.9 * self.length))
                & (self.speed_sum / self.length < .2))


def completion_success(completed, failed, terminal_geometry):
    """Earlier completion survives a brief wobble, but not falling/leaving exit."""
    return completed & ~failed & terminal_geometry


class StairTargetCommand(BalancedTargetCommand):
    stair_speed_schedule = STAIR_SPEED_RANGES
    initial_layout = staticmethod(specialist_layout)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.group, columns = self.initial_layout(self.num_envs, self.column_names, self.device)
        tensor(self.terrain.terrain_types)[:] = columns
        self.flat_pool[:] = self.group == 0
        for name in ('stair_phase', 'stair_epoch'):
            setattr(self, name, torch.zeros(self.num_envs, device=self.device, dtype=torch.long))
        self.frontier = torch.full((2,), 2, device=self.device, dtype=torch.long)
        for name in ('phase', 'epoch', 'attempts', 'wins'):
            setattr(self, name, torch.zeros(2, device=self.device, dtype=torch.long))
        self.last_rate = torch.full((2,), -1., device=self.device)
        self.stair_hold = torch.zeros(self.num_envs, device=self.device)
        self.stair_traversed = torch.zeros_like(self.flat_pool)
        self.stair_completed = torch.zeros_like(self.flat_pool)
        self.stair_strict_settled = torch.zeros_like(self.flat_pool)
        self.exit_window = ExitWindow(self.num_envs, env.step_dt, self.device)
        self.stair_started = torch.zeros_like(self.flat_pool)
        self.stair_manual_episode = torch.zeros_like(self.flat_pool)
        self.ankle_ids = self.robot.find_bodies('.*_ankle_roll_link')[0]
        if len(self.ankle_ids) != 2:
            raise ValueError('Stair completion requires exactly two ankle bodies')
        levels = tensor(self.terrain.terrain_levels)
        levels[self.stairs] = 2
        levels[self.flat_pool] = 0
        self.sync_origins()
        print('[StairCourse] flat/ascent/descent/other=25/35/20/20%; start at 10 cm; '
              f'stair speed ranges {self.stair_speed_schedule} m/s; >80%/50 attempts -> speed then height; '
              '<50% -> easier stage; 1 s window: >=90% exit geometry, mean speed <.2 m/s; '
              'angular speed diagnostic only; AMP/references unchanged.', flush=True)

    @property
    def stairs(self):
        return (self.group == 1) | (self.group == 2)

    def sync_origins(self):
        t = self.terrain
        tensor(t.env_origins)[:] = tensor(t.terrain_origins)[tensor(t.terrain_levels), tensor(t.terrain_types)]

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        self.stair_hold[ids] = 0.
        self.stair_traversed[ids] = False
        self.stair_completed[ids] = False
        self.stair_strict_settled[ids] = False
        self.exit_window.reset(ids)
        self.stair_started[ids] = False
        self.stair_manual_episode[ids] = False
        return super().reset(ids)

    def set_manual_target(self, env_id, position_w):
        super().set_manual_target(env_id, position_w)
        self.stair_manual_episode[env_id] = True
        self.stair_hold[env_id] = 0.
        self.stair_traversed[env_id] = False
        self.stair_completed[env_id] = False
        self.stair_strict_settled[env_id] = False
        self.exit_window.reset(self._ids([env_id]))

    def _resample_command(self, env_ids):
        ids = self._ids(env_ids)
        other = ids[~self.stairs[ids]]
        if len(other):
            super()._resample_command(other)
        # Keep one stair traversal goal and speed for the entire episode.
        ids = ids[self.stairs[ids] & ~self.manual_target[ids] & ~self.stair_started[ids]]
        if not len(ids):
            return
        levels, columns = tensor(self.terrain.terrain_levels)[ids], tensor(self.terrain.terrain_types)[ids]
        limits = stair_speed_ranges(self.stair_phase[ids], self.stair_speed_schedule)
        self.speed_cap[ids] = limits[:, 0] + torch.rand(len(ids), device=self.device) * (limits[:, 1] - limits[:, 0])
        self.pos_command_w[ids] = self.valid_targets[levels, columns, 0]
        self.is_standing_env[ids] = False
        self.stair_started[ids] = True
        self._begin_target(ids)

    def release_manual_target(self, env_id):
        if self.manual_target[env_id]:
            self.stair_started[env_id] = False
        super().release_manual_target(env_id)

    def exit_geometry(self):
        d = self.robot.data
        return exit_geometry(tensor(d.root_pos_w), tensor(d.body_pos_w)[:, self.ankle_ids],
                             self.pos_command_w, tensor(d.projected_gravity_b),
                             self.terrain.cfg.terrain_generator.size[0] / 2)

    def _update_metrics(self):
        valid_transition = ~self._skip_progress_transition.clone()
        super()._update_metrics()
        d = self.robot.data
        stable = settled_at_exit(tensor(d.root_pos_w), tensor(d.body_pos_w)[:, self.ankle_ids],
                                 self.pos_command_w, tensor(d.root_lin_vel_w), tensor(d.root_ang_vel_w),
                                 tensor(d.projected_gravity_b), self.terrain.cfg.terrain_generator.size[0] / 2)
        stable &= self.stairs & valid_transition & ~self.stair_manual_episode
        self.stair_hold[:] = torch.where(stable, self.stair_hold + self._env.step_dt, 0.)
        eligible = self.stairs & valid_transition & ~self.stair_manual_episode
        geometry = self.exit_geometry()
        self.stair_traversed |= geometry & eligible
        self.stair_strict_settled |= self.stair_hold >= 1.
        self.stair_completed |= self.exit_window.update(
            geometry, tensor(d.root_lin_vel_w)[:, :2].norm(dim=-1), eligible)


def stair_curriculum(env, env_ids):
    c = env.command_manager.get_term('base_velocity')
    ids = c._ids(env_ids)
    stairs = ids[c.stairs[ids]]
    # Flat speed and other rough terrain retain the existing balanced rules.
    stats = balanced_curriculum(env, ids[~c.stairs[ids]])
    valid = ~c.skip_curriculum_once[stairs] & ~c.stair_manual_episode[stairs]
    failed = tensor(env.termination_manager.terminated)[stairs].bool()
    passed = completion_success(c.stair_completed[stairs], failed, c.exit_geometry()[stairs])
    if hasattr(c, 'speed_success'):
        passed &= c.speed_success()[stairs]
    levels = tensor(c.terrain.terrain_levels)
    diagnostics = c.promotion_diagnostics() if hasattr(c, 'promotion_diagnostics') else {}
    for direction, label in enumerate(('ascent', 'descent')):
        selected = (c.group[stairs] == direction + 1) & valid
        challenge = selected & ((levels[stairs] > c.frontier[direction])
                                | (c.stair_phase[stairs] > c.phase[direction]))
        if getattr(c, 'adjacent_stage_challenges', False):
            stats[f'{label}/challenge_attempts'] = challenge.float().sum()
            stats[f'{label}/challenge_successes'] = (challenge & passed).float().sum()
        current = (selected & (levels[stairs] == c.frontier[direction])
                   & (c.stair_phase[stairs] == c.phase[direction])
                   & (c.stair_epoch[stairs] == c.epoch[direction]))
        if diagnostics:
            # Log counts with a shared denominator; reasons can overlap.
            stats[f'{label}/diagnostic_attempts'] = current.float().sum()
            for name, mask in diagnostics.items():
                stats[f'{label}/diagnostic_{name}'] = (current & mask[stairs]).float().sum()
        c.attempts[direction] += current.sum()
        c.wins[direction] += (current & passed).sum()
        level, phase, completed = advance_stage(int(c.frontier[direction]), int(c.phase[direction]),
                                                int(c.wins[direction]), int(c.attempts[direction]),
                                                len(getattr(c, 'stair_speed_schedule', STAIR_SPEED_RANGES)))
        if completed:
            c.last_rate[direction] = c.wins[direction] / c.attempts[direction]
            stage_changed = (level != int(c.frontier[direction]) or phase != int(c.phase[direction]))
            c.frontier[direction], c.phase[direction] = level, phase
            c.attempts[direction] = c.wins[direction] = 0
            # Long episodes must remain eligible when the task has not changed.
            # Legacy tasks retain their original evidence semantics.
            if stage_changed or not getattr(c, 'preserve_same_stage_evidence', False):
                c.epoch[direction] += 1
        reset = stairs[(c.group[stairs] == direction + 1) & ~c.skip_curriculum_once[stairs]]
        sampled_levels, sampled_phases = sample_stair_stages(
            torch.rand(len(reset), device=env.device), int(c.frontier[direction]),
            int(c.phase[direction]), len(getattr(c, 'stair_speed_schedule', STAIR_SPEED_RANGES)),
            challenge=(getattr(c, 'adjacent_stage_challenges', False)
                       and getattr(c, 'challenge_directions', (True, True))[direction]))
        levels[reset], c.stair_phase[reset] = sampled_levels, sampled_phases
        c.stair_epoch[reset] = c.epoch[direction]
        stats.update({f'{label}/height_cm': 100. * STAIR_HEIGHTS[int(c.frontier[direction])],
                      f'{label}/speed_phase': c.phase[direction].float(),
                      f'{label}/window_pass_rate': c.last_rate[direction],
                      f'{label}/window_attempts': c.attempts[direction].float(),
                      f'{label}/batch_attempts': selected.float().sum(),
                      f'{label}/batch_failures': (selected & failed).float().sum(),
                      f'{label}/batch_traversed': (selected & c.stair_traversed[stairs]).float().sum(),
                      f'{label}/batch_window_settled': (selected & c.stair_completed[stairs]).float().sum(),
                      f'{label}/batch_strict_settled': (selected & c.stair_strict_settled[stairs]).float().sum(),
                      f'{label}/batch_successes': (selected & passed).float().sum()})
    c.skip_curriculum_once[stairs] = False
    c.sync_origins()
    stats['mean_level'] = levels.float().mean()
    return stats
