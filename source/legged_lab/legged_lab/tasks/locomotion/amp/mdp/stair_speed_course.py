"""Opt-in speed continuation with measured cruise-speed promotion evidence."""
import torch
from .stair_course import StairTargetCommand, STAIR_SPEED_RANGES
from .balanced_course import FLAT_SPEED_RANGES
from .style_state import tensor
from .locomotion_progress import yaw_frame_velocity

FAST_FLAT_RANGES = FLAT_SPEED_RANGES + ((1.8, 2.5), (2.3, 3.0))
FAST_STAIR_RANGES = STAIR_SPEED_RANGES + ((.75, 1.0), (1.0, 1.25), (1.25, 1.5))


def overspeed_cost(actual_speed, commanded_speed, enabled, margin=.15):
    """Bounded excess-speed cost; no penalty within the tracking dead band."""
    excess = (actual_speed - commanded_speed - margin).clamp(0., 2.)
    return torch.where(enabled, excess.square(), torch.zeros_like(excess))


def descent_overspeed_penalty(env, command_name='base_velocity'):
    c = env.command_manager.get_term(command_name)
    actual = yaw_frame_velocity(c.robot.data)[:, :2].norm(dim=-1)
    desired = c.vel_command_b[:, :2].norm(dim=-1)
    enabled = (c.group == 2) & ~c._skip_progress_transition
    return overspeed_cost(actual, desired, enabled)


def cruise_success(seconds, tracking, actual, commanded):
    """Exclude short/no cruise, nonfinite evidence, underspeed and overspeed."""
    finite = torch.isfinite(seconds + tracking + actual + commanded)
    return (finite & (seconds >= .5) & (tracking > .6) & (commanded > 0.)
            & (actual >= .8 * commanded) & (actual <= 1.2 * commanded))


def cruise_failures(seconds, tracking, actual, commanded):
    """Overlapping failure reasons; denominators must be reported separately."""
    finite = torch.isfinite(seconds + tracking + actual + commanded)
    return {'nonfinite': ~finite,
            'short_cruise': finite & (seconds < .5),
            'low_tracking': finite & (tracking <= .6),
            'no_command': finite & (commanded <= 0.),
            'underspeed': finite & (actual < .8 * commanded),
            'overspeed': finite & (actual > 1.2 * commanded)}


def promotion_speed_limit(commanded, phase, schedule=FAST_STAIR_RANGES):
    """Allow readiness for the adjacent band, without claiming low-speed mastery."""
    last = len(schedule) - 1
    next_upper = torch.tensor(schedule, device=phase.device)[(phase.long() + 1).clamp_max(last), 1]
    strict = 1.2 * commanded
    return torch.where(phase < last, torch.maximum(strict, next_upper), strict)


def cruise_ready_for_promotion(seconds, tracking, actual, commanded, phase, schedule=FAST_STAIR_RANGES):
    finite = torch.isfinite(seconds + tracking + actual + commanded)
    return (finite & (seconds >= .5) & (tracking > .6) & (commanded > 0.)
            & (actual >= .8 * commanded)
            & (actual <= promotion_speed_limit(commanded, phase, schedule)))


class StairSpeedTargetCommand(StairTargetCommand):
    course_profile = 'speed_v1'
    promotion_version = 2
    flat_speed_schedule = FAST_FLAT_RANGES
    stair_speed_schedule = FAST_STAIR_RANGES

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        for name in ('cruise_seconds', 'cruise_tracking', 'cruise_actual_mps', 'cruise_command_mps',
                     'warm_cruise_seconds', 'warm_cruise_tracking', 'warm_cruise_actual_mps', 'warm_cruise_command_mps'):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        print('[StairSpeed] flat caps .6/1/1.5/2/2.5/3 m/s; stair caps .55/.75/1/1.25/1.5 m/s; '
              'promotion requires completion plus >=0.5 s cruise, tracking >0.6, actual>=0.8*command; '
              'intermediate bands permit safe speed up to the adjacent band cap; final band requires 0.8–1.2.',
              flush=True)

    def _update_metrics(self):
        # Capture the transition mask before the base method consumes it.
        eligible = self.stairs & ~self._skip_progress_transition & ~self.stair_manual_episode
        eligible &= tensor(self._env.episode_length_buf) > 0
        command = self.vel_command_b[:, 0]
        # Only the near-cap portion counts; arrival braking cannot mask crawling.
        eligible &= (command >= .8 * self.speed_cap) & (command > .1)
        actual = yaw_frame_velocity(self.robot.data)
        score = torch.exp(-(self.vel_command_b[:, :2] - actual[:, :2]).square().sum(-1) / .25)
        for prefix in ('', 'warm_'):
            active = eligible.clone()
            if prefix:
                active &= tensor(self._env.episode_length_buf) * self._env.step_dt > .5
            dt = active.float() * self._env.step_dt
            old = self.metrics[prefix + 'cruise_seconds']
            total = old + dt
            for name, value in (('cruise_tracking', score), ('cruise_actual_mps', actual[:, 0]),
                                ('cruise_command_mps', command)):
                value = torch.where(active, value, 0.)
                key = prefix + name
                self.metrics[key][:] = (self.metrics[key] * old + value * dt) / total.clamp_min(self._env.step_dt)
            old.copy_(total)
        super()._update_metrics()

    def speed_success(self):
        m = self.metrics
        return cruise_ready_for_promotion(m['cruise_seconds'], m['cruise_tracking'],
                                          m['cruise_actual_mps'], m['cruise_command_mps'],
                                          self.stair_phase, self.stair_speed_schedule)

    def promotion_diagnostics(self):
        m = self.metrics
        reasons = cruise_failures(m['cruise_seconds'], m['cruise_tracking'],
                                  m['cruise_actual_mps'], m['cruise_command_mps'])
        reasons.update(not_traversed=~self.stair_traversed,
                       never_settled=~self.stair_completed,
                       left_exit=~self.exit_geometry(),
                       terminated=tensor(self._env.termination_manager.terminated).bool())
        reasons['warm_speed_pass'] = cruise_success(m['warm_cruise_seconds'], m['warm_cruise_tracking'],
                                                    m['warm_cruise_actual_mps'], m['warm_cruise_command_mps'])
        reasons['strict_speed_pass'] = cruise_success(m['cruise_seconds'], m['cruise_tracking'],
                                                      m['cruise_actual_mps'], m['cruise_command_mps'])
        reasons['beyond_next_band'] = m['cruise_actual_mps'] > promotion_speed_limit(
            m['cruise_command_mps'], self.stair_phase, self.stair_speed_schedule)
        reasons['promotion_speed_pass'] = self.speed_success()
        return reasons
