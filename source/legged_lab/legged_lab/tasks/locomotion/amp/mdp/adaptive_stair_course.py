"""Independent height-at-moderate-speed and 8 cm speed curricula."""
import torch
from .balanced_course import balanced_curriculum
from .stair_course import STAIR_HEIGHTS
from .stair_speed_course import StairSpeedTargetCommand, cruise_success
from .style_state import tensor


HEIGHT, SPEED, REVIEW, CHALLENGE = range(4)
MAX_ADAPTIVE_LEVEL = STAIR_HEIGHTS.index(.30)
ADAPTIVE_FIELDS = ('speed_epoch', 'speed_attempts', 'speed_wins', 'speed_last_rate',
                   'height_age', 'speed_age')


def stair_crossing_reached(env):
    """Successful timeout at the exit: no stop, dwell or later position hold."""
    c = env.command_manager.get_term('base_velocity')
    return (c.stairs & ~c.stair_manual_episode & ~c._skip_progress_transition
            & (tensor(env.episode_length_buf) > 0) & c.exit_geometry())


def crossing_success(crossed, failed):
    return crossed & ~failed


def adaptive_step(value, wins, attempts, age, upper, lower=1, min_age=60.):
    """Wait for mature episodes, then move at most one level per window."""
    if attempts < 100 or age < min_age:
        return value, False
    rate = wins / attempts
    if rate >= .8:
        value = min(upper, value + 1)
    elif rate < .5:
        value = max(lower, value - 1)
    return value, True


def sample_adaptive_stages(draw, height, speed_phase, height_phase=1):
    """50% height, 20% 8 cm speed, 20% lower-height review, 10% higher."""
    height = min(MAX_ADAPTIVE_LEVEL, max(1, height))
    lane = torch.full_like(draw, HEIGHT, dtype=torch.long)
    lane[(draw >= .5) & (draw < .7)] = SPEED
    lane[(draw >= .7) & (draw < .9)] = REVIEW
    lane[draw >= .9] = CHALLENGE
    levels = torch.full_like(lane, height)
    phases = torch.full_like(lane, height_phase)
    levels[lane == SPEED], phases[lane == SPEED] = 1, speed_phase
    levels[lane == REVIEW] = max(1, height - 1)
    levels[lane == CHALLENGE] = min(MAX_ADAPTIVE_LEVEL, height + 1)
    return levels, phases, lane


class AdaptiveStairTargetCommand(StairSpeedTargetCommand):
    promotion_version = 7
    height_speed_phase = 1  # 0.55–0.75 m/s; independent of the fast 8 cm lane.

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.stair_lane = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        for name in ADAPTIVE_FIELDS:
            floating = name.endswith(('rate', 'age'))
            setattr(self, name, torch.zeros(2, dtype=torch.float if floating else torch.long, device=self.device))
        self.speed_last_rate.fill_(-1.)
        self.frontier.fill_(2)
        self.sample_adaptive(self._ids(None)[self.stairs])
        self.sync_origins()

    def sample_adaptive(self, ids):
        for direction in range(2):
            selected = ids[self.group[ids] == direction + 1]
            levels, phases, lane = sample_adaptive_stages(
                torch.rand(len(selected), device=self.device), int(self.frontier[direction]),
                int(self.phase[direction]), self.height_speed_phase)
            tensor(self.terrain.terrain_levels)[selected] = levels
            self.stair_phase[selected], self.stair_lane[selected] = phases, lane
            self.stair_epoch[selected] = torch.where(lane == SPEED, self.speed_epoch[direction], self.epoch[direction])

    def _update_metrics(self):
        super()._update_metrics()
        if hasattr(self, 'height_age'):
            self.height_age.add_(self._env.step_dt)
            self.speed_age.add_(self._env.step_dt)

    def save_adaptive_state(self):
        return {'version': 1, 'height_speed_phase': self.height_speed_phase, 'max_height_cm': 30,
                **{k: getattr(self, k).detach().cpu().clone() for k in ADAPTIVE_FIELDS}}

    def load_adaptive_state(self, state, clear_evidence=False):
        if state is None:
            # Start new height work at 10 cm without erasing the learned speed
            # frontier, policy, AMP, optimizer or flat curriculum.
            self.frontier.clamp_(min=2, max=MAX_ADAPTIVE_LEVEL)
            self.attempts.zero_(); self.wins.zero_(); self.epoch.zero_()
            self.last_rate.fill_(-1.)
            for name in ADAPTIVE_FIELDS:
                getattr(self, name).zero_()
            self.speed_last_rate.fill_(-1.)
        else:
            if (state.get('version') != 1 or state.get('height_speed_phase') != self.height_speed_phase
                    or state.get('max_height_cm') not in (20, 30)):
                raise ValueError('Incompatible adaptive height curriculum')
            if ((self.frontier < 1) | (self.frontier > MAX_ADAPTIVE_LEVEL)).any():
                raise ValueError('Adaptive height frontier must be between 8 and 30 cm')
            for name in ADAPTIVE_FIELDS:
                value = state[name]
                if value.shape != (2,) or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid adaptive curriculum field: {name}')
                getattr(self, name).copy_(value)
        if clear_evidence:
            self.attempts.zero_(); self.wins.zero_(); self.epoch.zero_()
            self.last_rate.fill_(-1.)
            for name in ADAPTIVE_FIELDS:
                getattr(self, name).zero_()
            self.speed_last_rate.fill_(-1.)
        # Loading already resets all physical episodes. Start fresh lane
        # assignments at the restored frontiers, retaining completed evidence.
        self.sample_adaptive(self._ids(None)[self.stairs])
        print('[AdaptiveStairs] height frontiers:', self.frontier.tolist(),
              '8cm speed phases:', self.phase.tolist(),
              '; 50% current height / 20% speed / 20% review / 10% higher; '
              'height speed 0.55–0.75 m/s; maximum height 30 cm; '
              '>=100 attempts and >=one episode before adaptation', flush=True)


def adaptive_stair_curriculum(env, env_ids):
    c = env.command_manager.get_term('base_velocity')
    ids = c._ids(env_ids)
    stairs = ids[c.stairs[ids]]
    stats = balanced_curriculum(env, ids[~c.stairs[ids]])
    valid = ~c.skip_curriculum_once[stairs] & ~c.stair_manual_episode[stairs]
    failed = tensor(env.termination_manager.terminated)[stairs].bool()
    # Terminations are computed before command metrics in Isaac Lab. Include
    # this step's geometry, so crossing is credited without a one-step delay.
    crossed = c.stair_traversed | c.exit_geometry()
    complete = crossing_success(crossed[stairs], failed)
    m = c.metrics
    strict_speed = cruise_success(m['cruise_seconds'], m['cruise_tracking'],
                                  m['cruise_actual_mps'], m['cruise_command_mps'])[stairs]
    passed = complete & torch.where(c.stair_lane[stairs] == SPEED, c.speed_success()[stairs], strict_speed)
    levels = tensor(c.terrain.terrain_levels)
    diagnostics = c.promotion_diagnostics()
    diagnostics.pop('never_settled', None)
    diagnostics.pop('left_exit', None)
    diagnostics['not_traversed'] = ~crossed
    for direction, label in enumerate(('ascent', 'descent')):
        selected = (c.group[stairs] == direction + 1) & valid
        height_current = (selected & (c.stair_lane[stairs] == HEIGHT)
                          & (levels[stairs] == c.frontier[direction])
                          & (c.stair_phase[stairs] == c.height_speed_phase)
                          & (c.stair_epoch[stairs] == c.epoch[direction]))
        speed_current = (selected & (c.stair_lane[stairs] == SPEED) & (levels[stairs] == 1)
                         & (c.stair_phase[stairs] == c.phase[direction])
                         & (c.stair_epoch[stairs] == c.speed_epoch[direction]))
        for lane, mask in (('height', height_current), ('speed', speed_current)):
            attempts = c.attempts if lane == 'height' else c.speed_attempts
            wins = c.wins if lane == 'height' else c.speed_wins
            epoch = c.epoch if lane == 'height' else c.speed_epoch
            rates = c.last_rate if lane == 'height' else c.speed_last_rate
            ages = c.height_age if lane == 'height' else c.speed_age
            values = c.frontier if lane == 'height' else c.phase
            attempts[direction] += mask.sum()
            wins[direction] += (mask & passed).sum()
            old = int(values[direction])
            new, finished = adaptive_step(old, int(wins[direction]), int(attempts[direction]),
                float(ages[direction]), MAX_ADAPTIVE_LEVEL if lane == 'height' else len(c.stair_speed_schedule) - 1,
                lower=1 if lane == 'height' else 0, min_age=env.cfg.episode_length_s)
            if finished:
                rates[direction] = wins[direction] / attempts[direction]
                attempts[direction] = wins[direction] = 0
                ages[direction] = 0.
                values[direction] = new
                if new != old:
                    epoch[direction] += 1
            stats.update({f'{label}/{lane}/attempts': mask.float().sum(),
                          f'{label}/{lane}/successes': (mask & passed).float().sum(),
                          f'{label}/{lane}/window_pass_rate': rates[direction].clone(),
                          f'{label}/{lane}/window_attempts': attempts[direction].float().clone(),
                          f'{label}/{lane}/window_age_s': ages[direction].clone()})
        stats[f'{label}/diagnostic_attempts'] = height_current.float().sum()
        for name, mask in diagnostics.items():
            stats[f'{label}/diagnostic_{name}'] = (height_current & mask[stairs]).float().sum()
        stats.update({f'{label}/height_cm': 100. * STAIR_HEIGHTS[int(c.frontier[direction])],
                      f'{label}/speed_phase': c.phase[direction].float().clone(),
                      f'{label}/height_speed_phase': float(c.height_speed_phase),
                      f'{label}/window_pass_rate': c.last_rate[direction].clone(),
                      f'{label}/batch_attempts': selected.float().sum(),
                      f'{label}/batch_successes': (selected & passed).float().sum(),
                      f'{label}/batch_failures': (selected & failed).float().sum(),
                      f'{label}/batch_traversed': (selected & crossed[stairs]).float().sum(),
                      f'{label}/challenge_attempts': (selected & (c.stair_lane[stairs] == CHALLENGE)).float().sum(),
                      f'{label}/challenge_successes': (selected & (c.stair_lane[stairs] == CHALLENGE) & passed).float().sum()})
    c.sample_adaptive(stairs[~c.skip_curriculum_once[stairs]])
    c.skip_curriculum_once[stairs] = False
    c.sync_origins()
    stats['mean_level'] = levels.float().mean()
    for group, label in ((1, 'ascent'), (2, 'descent')):
        stats[f'{label}/sampled_height_cm_mean'] = torch.tensor(STAIR_HEIGHTS, device=c.device)[levels[c.group == group]].mean() * 100.
    return stats
