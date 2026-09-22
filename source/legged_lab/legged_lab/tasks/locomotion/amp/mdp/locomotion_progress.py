"""Command-conditioned progress for v7; preserve standing and turn-in-place."""
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from .style_state import tensor


def command_progress(command, velocity_xy, yaw_rate, threshold=.15):
    """Signed progress capped at the command; no credit for backwards motion.

    Translation takes precedence when requested. Pure turns require actual
    rotation of the requested sign. Only a genuine zero command bypasses gating.
    """
    speed = command[:, :2].norm(dim=-1)
    projected = (velocity_xy * command[:, :2]).sum(-1) / speed.square().clamp_min(1e-6)
    turn = yaw_rate * command[:, 2] / command[:, 2].square().clamp_min(1e-6)
    return torch.where(speed > threshold, projected.clamp(0., 1.),
                       torch.where(command[:, 2].abs() > threshold, turn.clamp(0., 1.),
                                   torch.ones_like(speed)))


def yaw_frame_velocity(robot):
    """Root-link velocity in heading coordinates, without pitch/roll leakage."""
    return quat_apply_inverse(yaw_quat(tensor(robot.root_quat_w)), tensor(robot.root_link_lin_vel_w))


def motion_state(env, command_name='base_velocity'):
    robot = env.scene['robot'].data
    velocity = yaw_frame_velocity(robot)
    return env.command_manager.get_command(command_name), velocity[:, :2], tensor(robot.root_ang_vel_w)[:, 2]


def style_tracking_gate(command, velocity_xy, yaw_rate, linear_std=.35, angular_std=.5,
                        moving_tracking_floor=.5):
    """Retain moving style feedback despite tracking errors; still enforce stopping.

    The floor applies only to speed matching, not signed command progress.
    A commanded stationary or wrong-direction agent therefore still gets zero.
    """
    if not 0. <= moving_tracking_floor <= 1.:
        raise ValueError('moving_tracking_floor must be in [0, 1]')
    xy_error = (command[:, :2] - velocity_xy).square().sum(-1)
    yaw_error = (command[:, 2] - yaw_rate).square()
    tracking = torch.exp(-xy_error / linear_std**2 - yaw_error / angular_std**2)
    moving_command = (command[:, :2].norm(dim=-1) > .15) | (command[:, 2].abs() > .15)
    matching = torch.where(moving_command,
                           moving_tracking_floor + (1. - moving_tracking_floor) * tracking,
                           tracking)
    return command_progress(command, velocity_xy, yaw_rate) * matching


def style_progress_gate(env):
    return style_tracking_gate(*motion_state(env)).unsqueeze(-1)


def track_linear_progress(env, command_name='base_velocity', std=.5):
    command, velocity, yaw = motion_state(env, command_name)
    error = (command[:, :2] - velocity).square().sum(-1)
    return torch.exp(-error / std**2) * command_progress(command, velocity, yaw)


def track_angular_progress(env, command_name='base_velocity', std=.5):
    command, velocity, yaw = motion_state(env, command_name)
    return torch.exp(-(command[:, 2] - yaw).square() / std**2) * command_progress(command, velocity, yaw)


def low_pelvis_height(env, target_height=.55, asset_cfg=SceneEntityCfg('robot')):
    """Soft squat cost above the lower ankle; permits terrain-dependent bending."""
    data = env.scene[asset_cfg.name].data
    support_z = tensor(data.body_pos_w)[:, asset_cfg.body_ids, 2].min(dim=1).values
    clearance = tensor(data.root_pos_w)[:, 2] - support_z
    return (target_height - clearance).clamp_min(0.).square()
