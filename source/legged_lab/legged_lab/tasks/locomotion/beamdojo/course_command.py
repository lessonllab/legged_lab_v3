"""Explicit Stage 2 spawn/finish adaptation around official command sampling."""
import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


class CourseVelocityCommand(UniformVelocityCommand):
    """Keep sampling/curriculum intact; choose spawn and hold the episode end."""

    def _enabled(self):
        return (self._env.cfg.beamdojo_stage == 2
                and getattr(self._env.cfg, 'beamdojo_course_integrity', False))

    def _ensure_course_state(self):
        env = self._env
        if not hasattr(env, 'beamdojo_course_direction'):
            env.beamdojo_course_direction = torch.ones(env.num_envs, device=env.device)
        if not hasattr(env, 'beamdojo_finish_requested'):
            env.beamdojo_finish_requested = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def _orient_platforms(self):
        env = self._env
        if getattr(env.cfg, 'beamdojo_physical_aligned', False):
            # Both command directions start on the original central platform.
            env.beamdojo_start_platform = env.beamdojo_geometry[:, 0]
            env.beamdojo_end_platform = env.beamdojo_start_platform
            return
        rows = torch.arange(env.num_envs, device=env.device)
        first = env.beamdojo_geometry[:, 0]
        last = env.beamdojo_geometry[rows, env.beamdojo_rectangle_mask.sum(-1) - 1]
        forward = env.beamdojo_course_direction[:, None] >= 0
        env.beamdojo_start_platform = torch.where(forward, first, last)
        env.beamdojo_end_platform = torch.where(forward, last, first)

    def reset(self, env_ids=None):
        # Native reset clears episode_length_buf AFTER command.reset(), so a
        # nonzero old length is normal here. Preserve the official reset path.
        self._course_reset_context = True
        try:
            return super().reset(env_ids)
        finally:
            self._course_reset_context = False

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if not self._enabled():
            return
        self._ensure_course_state()
        env = self._env
        ids = torch.arange(env.num_envs, device=env.device)[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        reset_ids = ids if getattr(self, '_course_reset_context', False) else ids[env.episode_length_buf[ids] == 0]
        if reset_ids.numel():
            env.beamdojo_finish_requested[reset_ids] = False
            env.beamdojo_course_direction[reset_ids] = torch.where(self.vel_command_b[reset_ids, 0] >= 0, 1., -1.)
            self._orient_platforms()
            data = self.robot.data
            # The native write invalidates cached poses (skip_forward=False).
            # Preserve reset orientation, z, y randomization and all velocity.
            pose = torch.cat((_tensor(data.root_pos_w)[reset_ids], _tensor(data.root_quat_w)[reset_ids]), -1).clone()
            pose[:, 0] = _tensor(env.scene.env_origins)[reset_ids, 0] + env.beamdojo_start_platform[reset_ids, 0]
            self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=reset_ids)
        self.vel_command_b[env.beamdojo_finish_requested] = 0.

    def _update_command(self):
        super()._update_command()
        if not self._enabled():
            return
        self._ensure_course_state()
        if getattr(self._env.cfg, 'beamdojo_physical_aligned', False):
            # The traversal terminates at its progress goal, not a platform latch.
            return
        self._orient_platforms()
        env = self._env
        xy = _tensor(self.robot.data.root_pos_w)[:, :2] - _tensor(env.scene.env_origins)[:, :2]
        end = env.beamdojo_end_platform
        margin = torch.tensor([.25, .15], device=env.device)
        arrived = ((xy - end[:, :2]).abs() <= end[:, 2:4] / 2 - margin).all(-1)
        env.beamdojo_finish_requested |= arrived
        self.vel_command_b[env.beamdojo_finish_requested] = 0.


class CourseTrackingVelocityCommand(CourseVelocityCommand):
    """Official command ranges/curriculum with an explicit straight subset."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        fraction = float(self._env.cfg.tracking_straight_fraction)
        if not 0. <= fraction <= 1.:
            raise ValueError('Straight-command fraction must be within [0, 1]')
        selected = self.vel_command_b[env_ids].clone()
        straight = torch.rand(selected.shape[0], device=selected.device) < fraction
        selected[straight, 1:] = 0.
        self.vel_command_b[env_ids] = selected
