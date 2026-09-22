"""Instinct terrain promotion with an independent local flat-speed curriculum.

Terrain thresholds follow InstinctLab parkour/mdp/curriculums.py and its
parkour_env_cfg.py override (CC BY-NC 4.0).
"""
import torch
from .scratch_curriculum import ScratchTargetCommand
from .style_state import tensor


FLAT_SPEED_RANGES = ((.45, .6), (.65, 1.), (.9, 1.5), (1.2, 2.))


def flat_speed_ranges(tier, schedule=FLAT_SPEED_RANGES):
    table = torch.tensor(schedule, device=tier.device)
    return table[tier.long().clamp(0, len(schedule) - 1)]


def instinct_terrain_decisions(xy, yaw):
    """Exact configured Instinct predicates on full-horizon tracking scores."""
    up = (xy > .6) & (yaw > 0.)
    down = (xy < .3) & ~up
    return up, down


def update_evidence(good, bad, passed, failed, valid):
    """Two successes (neutral episodes allowed), two consecutive failures."""
    passed, failed = passed & valid & ~failed, failed & valid
    good = torch.where(failed, torch.zeros_like(good), good + passed.long())
    bad = torch.where(valid, torch.where(failed, bad + 1, torch.zeros_like(bad)), bad)
    up, down = (good >= 2) & valid, (bad >= 2) & valid
    good = torch.where(up | down, torch.zeros_like(good), good)
    bad = torch.where(up | down, torch.zeros_like(bad), bad)
    return good, bad, up, down


class BalancedTargetCommand(ScratchTargetCommand):
    flat_speed_schedule = FLAT_SPEED_RANGES

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.flat_pool = torch.arange(self.num_envs, device=self.device) % 4 == 0
        self.flat_speed_tier = torch.zeros_like(self.success_streak)
        self.failure_streak = torch.zeros_like(self.success_streak)
        self.speed_good = torch.zeros_like(self.success_streak)
        self.speed_bad = torch.zeros_like(self.success_streak)
        print(f'[BalancedCourse] 25% persistent flat practice; flat speed ranges {self.flat_speed_schedule} m/s; '
              'Instinct terrain: full-horizon xy>.6 and yaw>0 -> up, xy<.3 -> down; '
              'flat speed uses separate two-episode evidence; style/reference data unchanged.', flush=True)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = self._ids(env_ids)
        ids = ids[~self.manual_target[ids]]
        levels = tensor(self.terrain.terrain_levels)[ids]
        flat = ids[levels == 0]
        if not len(flat):
            return
        limits = flat_speed_ranges(self.flat_speed_tier[flat], self.flat_speed_schedule)
        self.speed_cap[flat] = limits[:, 0] + torch.rand(len(flat), device=self.device) * (limits[:, 1] - limits[:, 0])
        # Alternate endpoints inside the 8 m flat tile. Arrival immediately
        # triggers a new target, rather than waiting at the same +3 m goal.
        local_x = tensor(self.robot.data.root_pos_w)[flat, 0] - self._env.scene.env_origins[flat, 0]
        self.pos_command_w[flat] = self._env.scene.env_origins[flat]
        self.pos_command_w[flat, 0] += torch.where(local_x > 1., -3., 3.)
        self._begin_target(flat)

    def _update_command(self):
        super()._update_command()
        ids = torch.where((tensor(self.terrain.terrain_levels) == 0) & self.reached_target
                          & ~self.is_standing_env & ~self.manual_target)[0]
        if len(ids):
            self._resample(ids)


def balanced_curriculum(env, env_ids):
    c = env.command_manager.get_term('base_velocity')
    ids = c._ids(env_ids)
    m, terrain = c.metrics, env.scene.terrain
    xy, yaw = m['moving_tracking_exp_vel_xy'][ids], m['moving_tracking_exp_vel_yaw'][ids]
    progress, opportunity = m['target_progress_m'][ids], m['moving_opportunity_s'][ids]
    finite = torch.isfinite(xy) & torch.isfinite(yaw) & torch.isfinite(progress) & torch.isfinite(opportunity)
    valid = ~c.skip_curriculum_once[ids]
    if hasattr(c,'turn_pool'):
        valid &= ~c.turn_pool[ids]
    if hasattr(c,'reverse_pool'):
        valid &= ~c.reverse_pool[ids]
    terminated = tensor(env.termination_manager.terminated)[ids].bool()
    timeout = tensor(env.termination_manager.time_outs)[ids].bool()
    enough = (progress >= 1.) & (opportunity >= 2.)
    passed = finite & (xy > .6) & (yaw > 0.) & enough & timeout & ~terminated
    failed = terminated | ~finite | ((opportunity >= 2.) & (xy < .3))
    levels = tensor(terrain.terrain_levels)[ids].clone()
    terrain_xy, terrain_yaw = m['tracking_exp_vel_xy'][ids], m['tracking_exp_vel_yaw'][ids]
    up, down = instinct_terrain_decisions(terrain_xy, terrain_yaw)
    # Only initialization and the explicitly reserved flat pool are excluded.
    # No success streak, timeout, fall, distance, or opportunity gate is added.
    terrain_up = up & valid & ~c.flat_pool[ids]
    terrain_down = down & valid & ~c.flat_pool[ids]
    if getattr(c,'guarded_rough_curriculum',False):
        from .agility_course import rough_outcomes
        safe, bad = rough_outcomes(xy,yaw,progress,opportunity,m['valid_targets_reached'][ids],terminated)
        good, failures, promote, demote = update_evidence(c.success_streak[ids],c.failure_streak[ids],
            safe,bad,valid & ~c.flat_pool[ids])
        c.success_streak[ids],c.failure_streak[ids] = good,failures
        terrain_up,terrain_down = promote,demote
        up = safe
    # Keep legacy checkpoint fields, but they no longer control terrain changes.
    if not getattr(c,'guarded_rough_curriculum',False):
        c.success_streak[ids] = 0
        c.failure_streak[ids] = 0
    # Let the importer handle levels above the final row exactly as Instinct:
    # reassign to a random row instead of suppressing further promotions.
    terrain.update_env_origins(ids, terrain_up, terrain_down)
    sg, sb, speed_up, speed_down = update_evidence(c.speed_good[ids], c.speed_bad[ids], passed, failed, valid & (levels == 0))
    c.speed_good[ids], c.speed_bad[ids] = sg, sb
    schedule = getattr(c, 'flat_speed_schedule', FLAT_SPEED_RANGES)
    c.flat_speed_tier[ids] = (c.flat_speed_tier[ids] + speed_up.long() - speed_down.long()).clamp(0, len(schedule) - 1)
    c.skip_curriculum_once[ids] = False
    levels = tensor(terrain.terrain_levels)
    eligible = ~c.flat_pool
    denominator = valid.float().sum().clamp_min(1.)
    rate = lambda mask: (mask & valid).float().sum() / denominator
    return {'mean_level': levels.float().mean(), 'flat_fraction': (levels == 0).float().mean(),
            'rough_fraction': (levels > 0).float().mean(),
            'eligible_rough_fraction': ((levels > 0) & eligible).float().sum() / eligible.float().sum().clamp_min(1.),
            'flat_pool_fraction': c.flat_pool.float().mean(),
            'flat_speed_cap_mean': flat_speed_ranges(c.flat_speed_tier[c.flat_pool], schedule)[:, 1].sum() / c.flat_pool.float().sum().clamp_min(1.),
            'flat_fast_fraction': ((c.flat_speed_tier >= 2) & c.flat_pool).float().sum() / c.flat_pool.float().sum().clamp_min(1.),
            'blocked_tracking': rate(terrain_xy <= .6), 'blocked_yaw': rate(terrain_yaw <= 0.),
            'failed': rate(terminated), 'episode_pass': rate(up),
            'flat_speed_blocked_tracking': rate(xy <= .6),
            'flat_speed_blocked_progress': rate(progress < 1.),
            'flat_speed_blocked_opportunity': rate(opportunity < 2.),
            'promoted': rate(terrain_up), 'demoted': rate(terrain_down)}
