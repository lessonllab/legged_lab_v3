from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject  # runtime class, guarded per v3 pattern
    from isaaclab.sensors import ContactSensor  # runtime class, guarded per v3 pattern
    from isaaclab.envs import ManagerBasedRLEnv


def feet_orientation_l2(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet orientation not parallel to the ground when in contact.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # shape: (N, M)

    num_feet = len(sensor_cfg.body_ids)

    feet_quat = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]  # shape: (N, M, 4)
    feet_proj_g = math_utils.quat_apply_inverse(
        feet_quat, asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, num_feet, -1)  # shape: (N, M, 3)
    )
    feet_proj_g_xy_square = torch.sum(torch.square(feet_proj_g[:, :, :2]), dim=-1)  # shape: (N, M)

    return torch.sum(feet_proj_g_xy_square * in_contact, dim=-1)  # shape: (N, )


def _as_torch(value) -> torch.Tensor:
    """Accept both Isaac Lab 3 ProxyArray buffers and ordinary tensor test data."""
    return value if isinstance(value, torch.Tensor) else value.torch


def instinct_feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    vel_threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward single-support steps under translation *or* turning commands.

    Adapted from InstinctLab parkour's ``feet_air_time`` (CC BY-NC 4.0):
    https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py

    Return the shorter of the stance and swing durations when exactly one foot
    is grounded. The reward manager applies the configured weight and step_dt;
    this function does not add another time-step factor. Tracking air time must
    be enabled on the contact sensor and sensor_cfg must select the two feet.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    air_time = _as_torch(contact_sensor.data.current_air_time)[:, sensor_cfg.body_ids]
    contact_time = _as_torch(contact_sensor.data.current_contact_time)[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact, dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1).values
    command = _as_torch(env.command_manager.get_command(command_name))
    moving = (torch.norm(command[:, :2], dim=1) > vel_threshold) | (torch.abs(command[:, 2]) > vel_threshold)
    return reward * moving


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15,
    offset: float = 4.0,
) -> torch.Tensor:
    """Return joint deviation minus an offset only for a standing command.

    Adapted from InstinctLab parkour's ``stand_still`` (CC BY-NC 4.0):
    https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py

    With a negative reward weight this both discourages joint deviation and
    provides a standing bonus near the default pose. Do not clamp the offset:
    it is intentionally a bonus, not a dead band. Joint IDs are honored so the
    caller can explicitly select the whole body or a particular joint set.
    """
    data = env.scene[asset_cfg.name].data
    joint_pos = _as_torch(data.joint_pos)[:, asset_cfg.joint_ids]
    default_pos = _as_torch(data.default_joint_pos)[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(joint_pos - default_pos), dim=1)
    command = _as_torch(env.command_manager.get_command(command_name))
    standing = (torch.norm(command[:, :2], dim=1) < threshold) & (torch.abs(command[:, 2]) < threshold)
    return (deviation - offset) * standing


def body_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize tilt of the selected rigid bodies using projected gravity.

    Select explicit pelvis/torso body names in the reward configuration; this
    uses each selected body's world quaternion, not the articulation root pose.
    Selection/order is resolved by SceneEntityCfg. Following InstinctLab's
    ``link_orientation``, sum the squared horizontal gravity components; when
    selecting multiple bodies their costs are added. This does not penalize yaw.
    """
    data = env.scene[asset_cfg.name].data
    body_quat = _as_torch(data.body_quat_w)[:, asset_cfg.body_ids, :]
    gravity = _as_torch(data.GRAVITY_VEC_W).unsqueeze(1).expand(-1, body_quat.shape[1], -1)
    body_gravity = math_utils.quat_apply_inverse(body_quat, gravity)
    return torch.sum(torch.square(body_gravity[..., :2]), dim=(-1, -2))


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    angular_threshold: float = 0.15,
) -> torch.Tensor:
    """Penalize joint offsets only when both translation and turning are small.

    The separate thresholds have units of m/s and rad/s respectively. A pure
    turning command must allow stepping, even though its XY component is zero.
    This follows the two-component standing gate in InstinctLab's ``stand_still``:
    https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py
    """
    command = env.command_manager.get_command(command_name)
    standing = (torch.norm(command[:, :2], dim=1) < command_threshold) & (
        torch.abs(command[:, 2]) < angular_threshold
    )
    return mdp.joint_deviation_l1(env, asset_cfg) * standing


def target_heading_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize target misalignment through the target controller's yaw command.

    Use a negative weight and a target-derived velocity command. The controller
    recomputes its yaw command from heading error, so facing the target reduces
    this cost. For arbitrary velocity commands this would instead be a constant
    command-dependent cost and would not measure the robot's heading error.
    This matches InstinctLab parkour's ``heading_error`` definition (CC BY-NC 4.0):
    https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py
    """
    return torch.abs(env.command_manager.get_command(command_name)[:, 2])


def dont_wait(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.3,
    speed_threshold: float = 0.15,
    startup_grace_s: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_yaw_frame: bool = False,
) -> torch.Tensor:
    """Penalize failing to move under a forward command; use a negative weight.

    Based on InstinctLab parkour's dont_wait, with an episode-start grace period:
    https://github.com/project-instinct/InstinctLab/blob/main/source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py
    """
    command = env.command_manager.get_command(command_name)
    velocity = env.scene[asset_cfg.name].data.root_lin_vel_b
    if use_yaw_frame:
        from .locomotion_progress import yaw_frame_velocity
        velocity = yaw_frame_velocity(env.scene[asset_cfg.name].data)
    if not isinstance(velocity, torch.Tensor):
        velocity = velocity.torch
    vx = velocity[:, 0]
    penalty = (vx < speed_threshold).float() + (vx < 0.0).float() + (vx < -speed_threshold).float()
    active = (command[:, 0] > command_threshold) & (env.episode_length_buf * env.step_dt >= startup_grace_s)
    return penalty * active
