"""Instinct tracking curriculum with a flat entry row and local progress guards.

Reference: InstinctLab parkour/mdp/curriculums.py (CC BY-NC 4.0).
Success streaks, fall/progress guards, and speed staging are local additions.
"""
import torch
from .commands.target_velocity import TargetVelocityCommand
from .style_state import tensor


def stage_speed_ranges(levels, final_ranges):
    """Flat entry at .45-.6; gently restore each terrain's full requested range."""
    fraction = (levels.float() / 3.).clamp(0., 1.)
    low = torch.full_like(fraction, .45)
    high = .6 + fraction * (final_ranges[:, 1] - .6)
    return torch.stack((low, high), -1)


class ScratchTargetCommand(TargetVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.success_streak = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.skip_curriculum_once = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        if type(self) is ScratchTargetCommand:
            print('[ScratchCourse] row 0 = flat .45-.60 m/s; rows 1+ = rough; '
                  'full terrain-specific speeds by row 3; promote after 3 successful episodes; '
                  'AMP=.50 additive, no velocity gate; all references retained.', flush=True)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = self._ids(env_ids)
        ids = ids[~self.manual_target[ids]]
        if not len(ids):
            return
        columns = tensor(self.terrain.terrain_types)[ids].long()
        levels = tensor(self.terrain.terrain_levels)[ids].long()
        limits = stage_speed_ranges(levels, self.speed_ranges[columns])
        self.speed_cap[ids] = limits[:, 0] + torch.rand(len(ids), device=self.device) * (limits[:, 1] - limits[:, 0])
        # Predictable forward target only on the flat entry row. Later rows use
        # Instinct-style terrain patches, including turns and arrival stopping.
        flat = ids[levels == 0]
        self.pos_command_w[flat] = self._env.scene.env_origins[flat]
        self.pos_command_w[flat, 0] += 3.0
        self._begin_target(ids)


def update_streak(streak, xy, yaw, progress, opportunity, timeout, failed, initialized,
                  max_level, levels):
    finite = torch.isfinite(xy) & torch.isfinite(yaw) & torch.isfinite(progress) & torch.isfinite(opportunity)
    # Instinct explicitly configures angular threshold 0, not its helper's .5 default.
    success = finite & (xy > .6) & (yaw > 0.) & (progress >= 1.) & (opportunity >= 2.) & timeout & ~failed & initialized
    streak = torch.where(success, streak + 1, torch.zeros_like(streak))
    up = (streak >= 3) & (levels < max_level - 1)
    down = initialized & (failed | (xy < .3) | ~finite) & ~up
    streak = torch.where(up | down, torch.zeros_like(streak), streak)
    return streak, up, down


def scratch_tracking_curriculum(env, env_ids):
    command = env.command_manager.get_term('base_velocity')
    ids = command._ids(env_ids)
    m = command.metrics
    terrain = env.scene.terrain
    streak, up, down = update_streak(
        command.success_streak[ids], m['tracking_exp_vel_xy'][ids], m['tracking_exp_vel_yaw'][ids],
        m['target_progress_m'][ids], m['moving_opportunity_s'][ids],
        tensor(env.termination_manager.time_outs)[ids].bool(),
        tensor(env.termination_manager.terminated)[ids].bool(),
        ~command.skip_curriculum_once[ids], terrain.max_terrain_level, tensor(terrain.terrain_levels)[ids])
    command.success_streak[ids] = streak
    command.skip_curriculum_once[ids] = False
    terrain.update_env_origins(ids, up, down)
    levels = tensor(terrain.terrain_levels)
    return {'mean_level': levels.float().mean(), 'flat_fraction': (levels == 0).float().mean(),
            'rough_fraction': (levels > 0).float().mean(), 'full_speed_fraction': (levels >= 3).float().mean()}
