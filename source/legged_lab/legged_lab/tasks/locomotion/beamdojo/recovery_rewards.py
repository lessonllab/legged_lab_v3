"""Opt-in movement gates for the unchanged official clearance/gait formulas.

Simulator base velocity is reward-only truth; this module adds no observations.
The caller must explicitly select these wrappers in its recovery profile.
"""
import torch

from ..unitree_official import mdp as official


def _command_progress(env, command_name, asset_name, moving_threshold):
    command = official.tensor(env.command_manager.get_command(command_name))[:, :2]
    data = env.scene[asset_name].data
    velocity = official.quat_apply_inverse(
        official.yaw_quat(official.tensor(data.root_quat_w)),
        official.tensor(data.root_lin_vel_w)[:, :3])[:, :2]
    norm_squared = command.square().sum(-1)
    moving = command.norm(dim=-1) > moving_threshold
    ratio = ((velocity * command).sum(-1) / norm_squared.clamp_min(1e-8)).clamp(0., 1.)
    return moving, ratio


def motion_gate(env, command_name="base_velocity", asset_name="robot", moving_threshold=.2):
    """Smoothly enable a translation reward at half commanded aligned speed."""
    moving, ratio = _command_progress(env, command_name, asset_name, moving_threshold)
    fraction = (ratio / .5).clamp(0., 1.)
    return fraction.square() * (3. - 2. * fraction) * moving


def feet_clearance_moving(env, asset_cfg, target_height, std, tanh_mult,
                          command_name="base_velocity", moving_threshold=.2):
    """Preserve official clearance value, but do not reward commanded standing."""
    reward = official.foot_clearance_reward(env, asset_cfg, target_height, std, tanh_mult)
    return reward * motion_gate(env, command_name, asset_cfg.name, moving_threshold)


def gait_moving(env, period, offset, sensor_cfg, threshold=.5,
                command_name="base_velocity", asset_name="robot", moving_threshold=.2):
    """Gate translation gait by base progress; retain official pure-turn gait."""
    reward = official.feet_gait(env, period, offset, sensor_cfg, threshold, command_name)
    moving, _ = _command_progress(env, command_name, asset_name, moving_threshold)
    gate = motion_gate(env, command_name, asset_name, moving_threshold)
    return torch.where(moving, reward * gate, reward)


def command_stall(env, command_name="base_velocity", asset_name="robot", moving_threshold=.2):
    """Aligned-speed deficit in [0, 1]; use a negative reward weight."""
    moving, ratio = _command_progress(env, command_name, asset_name, moving_threshold)
    return (1. - ratio) * moving
