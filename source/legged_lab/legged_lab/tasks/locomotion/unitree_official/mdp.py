"""Unitree terms with explicit ProxyArray-to-Torch and XYZW quaternion access.

Baseline formulas follow Unitree commit 4960b847 and its Isaac Lab helpers.
Explicit opt-in direction-training extensions are defined below; no AMP is used.
"""
import torch
from dataclasses import MISSING
from isaaclab.envs import mdp as native
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def __getattr__(name):
    return getattr(native, name)


def tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


@configclass
class UniformLevelVelocityCommandCfg(native.UniformVelocityCommandCfg):
    limit_ranges: native.UniformVelocityCommandCfg.Ranges = MISSING
    direction_yaw_min: float = .1


def energy(env, asset_cfg=SceneEntityCfg("robot")):
    data = env.scene[asset_cfg.name].data
    return (tensor(data.joint_vel)[:, asset_cfg.joint_ids].abs()
            * tensor(data.applied_torque)[:, asset_cfg.joint_ids].abs()).sum(-1)


def foot_clearance_reward(env, asset_cfg, target_height, std, tanh_mult):
    data = env.scene[asset_cfg.name].data
    error = (tensor(data.body_pos_w)[:, asset_cfg.body_ids, 2] - target_height).square()
    speed = torch.tanh(tanh_mult * torch.norm(tensor(data.body_lin_vel_w)[:, asset_cfg.body_ids, :2], dim=2))
    return torch.exp(-(error * speed).sum(1) / std)


def feet_gait(env, period, offset, sensor_cfg, threshold=.5, command_name=None):
    contact = tensor(env.scene.sensors[sensor_cfg.name].data.current_contact_time)[:, sensor_cfg.body_ids] > 0
    phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = torch.cat([(phase + value) % 1.0 for value in offset], dim=-1)
    reward = torch.zeros(env.num_envs, device=env.device)
    for index in range(len(sensor_cfg.body_ids)):
        reward += ~((phases[:, index] < threshold) ^ contact[:, index])
    if command_name is not None:
        reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > .1
    return reward


def feet_slide(env, sensor_cfg, asset_cfg=SceneEntityCfg("robot")):
    forces = tensor(env.scene.sensors[sensor_cfg.name].data.net_forces_w_history)
    contact = forces[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1)[0] > 1.
    velocity = tensor(env.scene[asset_cfg.name].data.body_lin_vel_w)[:, asset_cfg.body_ids, :2]
    return (velocity.norm(dim=-1) * contact).sum(1)


def feet_gait_directional(env, period, offset, sensor_cfg, threshold=.5,
                          command_name="base_velocity", linear_threshold=.1, yaw_threshold=.05):
    """Keep Unitree's contact-phase reward, activate yaw independently of m/s."""
    command = env.command_manager.get_command(command_name)
    moving = (command[:, :2].norm(dim=1) > linear_threshold) | (command[:, 2].abs() > yaw_threshold)
    return feet_gait(env, period, offset, sensor_cfg, threshold, command_name=None) * moving


def feet_gait_turn_support(env, period, offset, sensor_cfg, threshold=.5,
                           command_name="base_velocity", linear_threshold=.1, yaw_threshold=.05):
    """Only reward phase matching during single support for pure-turn commands.

    Ordinary walking retains its previous reward; flight and double support do
    not earn the turning gait reward. Other stability/clearance terms are kept.
    """
    reward = feet_gait_directional(env, period, offset, sensor_cfg, threshold,
                                   command_name, linear_threshold, yaw_threshold)
    command = env.command_manager.get_command(command_name)
    pure_turn = (command[:, :2].norm(dim=1) <= linear_threshold) & (command[:, 2].abs() > yaw_threshold)
    contact = tensor(env.scene.sensors[sensor_cfg.name].data.current_contact_time)[:, sensor_cfg.body_ids] > 0
    single_support = contact.sum(dim=1) == 1
    return reward * (~pure_turn | single_support)


def track_lin_vel_xy_yaw_frame_exp(env, std, command_name, asset_cfg=SceneEntityCfg("robot")):
    data = env.scene[asset_cfg.name].data
    velocity = quat_apply_inverse(yaw_quat(tensor(data.root_quat_w)), tensor(data.root_lin_vel_w)[:, :3])
    error = (env.command_manager.get_command(command_name)[:, :2] - velocity[:, :2]).square().sum(1)
    return torch.exp(-error / std**2)


def terrain_levels_vel(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(tensor(env.scene[asset_cfg.name].data.root_pos_w)[env_ids, :2]
                          - tensor(env.scene.env_origins)[env_ids, :2], dim=1)
    up = distance > terrain.cfg.terrain_generator.size[0] / 2
    down = (distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * .5) & ~up
    terrain.update_env_origins(env_ids, up, down)
    return terrain.terrain_levels.float().mean()


def lin_vel_cmd_levels(env, env_ids, reward_term_name="track_lin_vel_xy"):
    term = env.command_manager.get_term("base_velocity")
    ranges, limits = term.cfg.ranges, term.cfg.limit_ranges
    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    reward = env.reward_manager._episode_sums[reward_term_name][env_ids].mean() / env.max_episode_length_s
    if env.common_step_counter % env.max_episode_length == 0 and reward > reward_cfg.weight * .8:
        delta = torch.tensor([-.1, .1], device=env.device)
        for name in ("lin_vel_x", "lin_vel_y"):
            bound = getattr(limits, name)
            value = (torch.tensor(getattr(ranges, name), device=env.device) + delta).clamp(*bound).tolist()
            setattr(ranges, name, value)
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env,
    env_ids,
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def sample_direction_commands(count, ranges, device, yaw_min=.1):
    """Explicit maneuver mixture; category probabilities sum to one."""
    kind = torch.bucketize(torch.rand(count, device=device),
                          torch.tensor([.30, .50, .70, .90, .98], device=device))
    values = torch.zeros(count, 3, device=device)
    def uniform(lo, hi):
        return lo + torch.rand(count, device=device) * (hi - lo)
    sign = torch.where(torch.rand(count, device=device) < .5, -1., 1.)
    forward = uniform(min(.2, ranges.lin_vel_x[1]), ranges.lin_vel_x[1])
    back_max = abs(min(0., ranges.lin_vel_x[0]))
    back = -uniform(min(.1, back_max), back_max)
    side_max = min(abs(ranges.lin_vel_y[0]), ranges.lin_vel_y[1])
    side = sign * uniform(min(.15, side_max), side_max)
    yaw_max = min(abs(ranges.ang_vel_z[0]), ranges.ang_vel_z[1])
    yaw = sign * uniform(min(yaw_min, yaw_max), yaw_max)
    values[:, 0] = torch.where(kind == 0, forward, torch.where(kind == 1, back, 0.))
    values[:, 1] = torch.where(kind == 2, side, 0.)
    values[:, 2] = torch.where((kind == 3) | (kind == 4), yaw, 0.)
    values[:, 0] = torch.where(kind == 4, uniform(min(.2, ranges.lin_vel_x[1]),
                                               min(1., ranges.lin_vel_x[1])), values[:, 0])
    return values, kind


class DirectionBalancedVelocityCommand(native.UniformVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        values, kind = sample_direction_commands(len(env_ids), self.cfg.ranges, self.device,
                                                 yaw_min=self.cfg.direction_yaw_min)
        self.vel_command_b[env_ids] = values
        self.is_standing_env[env_ids] = kind == 5
        self.is_heading_env[env_ids] = False


def track_lin_vel_xy_precision(env, command_name, std=.5, low_speed_std=.2, low_speed_limit=.6,
                               asset_cfg=SceneEntityCfg("robot")):
    data = env.scene[asset_cfg.name].data
    velocity = quat_apply_inverse(yaw_quat(tensor(data.root_quat_w)), tensor(data.root_lin_vel_w)[:, :3])
    command = env.command_manager.get_command(command_name)[:, :2]
    width = torch.where(command.norm(dim=1) <= low_speed_limit, low_speed_std, std)
    return torch.exp(-(command - velocity[:, :2]).square().sum(1) / width.square())
