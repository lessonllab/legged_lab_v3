"""Small optional yaw/parking terms; importable with only Torch on the CPU.

Keep the complete Unitree motion reward configuration, including its gait and
clearance formulas/weights. These terms add no observations or gait targets.
For segment heading, reuse ``course_tracking.heading_error_l2`` only when an
active course-heading term is absent; it already handles command transitions,
reset references and intentional turns. Recommended initial heading weight is
-.20 with its existing .5 rad scale. No additional cross-track term is needed.

All values are reward rates; the native RewardManager applies control dt.
Parking rewards require a near-zero command, safe complete soles, and double
contact on either a flat rehearsal tile or the latched terminal ring. They do
not encourage a nominal standing posture while the robot is walking.
"""
import math

import torch


INITIAL_WEIGHTS = {
    'straight_yaw_rate': -.10,
    'course_heading': -.20,
    'parking_stability': .40,
    'parking_pose': -.05,
}


def tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def _positive(value, name):
    if not math.isfinite(value) or value <= 0.:
        raise ValueError(f'{name} must be finite and positive')


def straight_command_mask(command, moving_threshold=.1, zero_tolerance=1e-4):
    """Match course_tracking: forward/reverse, with zero lateral/yaw demand."""
    _positive(moving_threshold, 'moving_threshold')
    _positive(zero_tolerance, 'zero_tolerance')
    return (torch.isfinite(command).all(-1)
            & (command[:, 0].abs() >= moving_threshold)
            & (command[:, 1].abs() <= zero_tolerance)
            & (command[:, 2].abs() <= zero_tolerance))


def stand_command_mask(command, command_threshold=.05):
    """A planar command norm below .05 m/s and yaw below .05 rad/s."""
    _positive(command_threshold, 'command_threshold')
    return (torch.isfinite(command).all(-1)
            & (command[:, :2].norm(dim=-1) < command_threshold)
            & (command[:, 2].abs() < command_threshold))


def imu_yaw_rate(angular_velocity_b, projected_gravity_b):
    """Euler yaw rate from gyro/gravity, using course_tracking's same formula.

    The signed rate comes from signals already available to the policy. It does
    not use absolute world heading. Clamp the tilt denominator as in tracking.
    """
    gravity = projected_gravity_b
    return -(angular_velocity_b[:, 1] * gravity[:, 1]
             + angular_velocity_b[:, 2] * gravity[:, 2]) / (
                 gravity[:, 1].square() + gravity[:, 2].square()).clamp_min(.25)


def straight_yaw_rate_values(command, angular_velocity_b, projected_gravity_b,
                             scale=.35, moving_threshold=.1, zero_tolerance=1e-4):
    """Nonnegative squared unintended yaw rate in [0, 4]; use negative weight."""
    _positive(scale, 'scale')
    active = straight_command_mask(command, moving_threshold, zero_tolerance)
    rate = imu_yaw_rate(angular_velocity_b, projected_gravity_b)
    # Invalid active measurements receive the bounded maximum penalty, while
    # inactive turns/standing cannot contaminate the batch with NaNs.
    normalized = torch.nan_to_num(rate / scale, nan=2., posinf=2., neginf=-2.)
    return torch.where(active, normalized.clamp(-2., 2.).square(), 0.)


def straight_yaw_rate_l2(env, command_name='base_velocity', asset_name='robot',
                         scale=.35, moving_threshold=.1, zero_tolerance=1e-4):
    data = env.scene[asset_name].data
    return straight_yaw_rate_values(
        tensor(env.command_manager.get_command(command_name)),
        tensor(data.root_ang_vel_b), tensor(data.projected_gravity_b),
        scale, moving_threshold, zero_tolerance)


def parking_gate_values(command, contact_fz, sole_supported, corner_supported,
                        eligible, command_threshold=.05, contact_threshold=5.):
    """Require both full soles/corners and both contacts on a valid stop area."""
    _positive(contact_threshold, 'contact_threshold')
    if contact_fz.shape != (command.shape[0], 2):
        raise ValueError('Parking requires exactly two ordered foot contacts')
    if sole_supported.shape != contact_fz.shape:
        raise ValueError('Parking requires complete support for each foot')
    if (corner_supported.ndim != 3 or corner_supported.shape[:2] != contact_fz.shape
            or corner_supported.shape[-1] != 4):
        raise ValueError('Parking requires explicit support for all four sole corners')
    return (stand_command_mask(command, command_threshold) & eligible
            & torch.isfinite(contact_fz).all(-1)
            & (contact_fz > contact_threshold).all(-1)
            & sole_supported.bool().all(-1)
            & corner_supported.bool().all((-1, -2)))


def parking_stability_values(command, linear_velocity, angular_velocity,
                             projected_gravity, contact_fz, sole_supported,
                             corner_supported, eligible, linear_scale=.15,
                             angular_scale=.3, tilt_scale=.35,
                             command_threshold=.05, contact_threshold=5.):
    """Support-gated exp stability in [0, 1]; use a small positive weight."""
    for name, value in (('linear_scale', linear_scale), ('angular_scale', angular_scale),
                        ('tilt_scale', tilt_scale)):
        _positive(value, name)
    active = parking_gate_values(command, contact_fz, sole_supported, corner_supported,
                                 eligible, command_threshold, contact_threshold)
    finite = (torch.isfinite(linear_velocity).all(-1)
              & torch.isfinite(angular_velocity).all(-1)
              & torch.isfinite(projected_gravity).all(-1))
    tilt = torch.acos((-projected_gravity[:, 2]).clamp(-1., 1.))
    error = (linear_velocity.square().sum(-1) / linear_scale ** 2
             + angular_velocity.square().sum(-1) / angular_scale ** 2
             + tilt.square() / tilt_scale ** 2)
    quality = torch.exp(-torch.nan_to_num(error, nan=float('inf')))
    return torch.where(active & finite, quality, 0.)


def _parking_inputs(env, command_name):
    # Delay runtime/native access so this module and its tensor math are usable
    # without Isaac Sim. The selected profile installs extended hooks here.
    import beamdojo_square_mixed_runtime as runtime
    command = tensor(env.command_manager.get_command(command_name))
    measurements = runtime.read_measurements(env)
    state = runtime.ensure_state(env)
    flat = getattr(env, 'flat_rehearsal_mask', torch.zeros_like(command[:, 0], dtype=torch.bool))
    _, _, full_ring = runtime._geometry(env, measurements)
    terminal = state.get('terminal_entered', state['parked'])
    eligible = flat | (terminal & full_ring)
    # A failed environment must never collect standing credit, even on flat.
    eligible &= ~getattr(env, 'beamdojo_course_failed', torch.zeros_like(flat))
    return command, measurements, eligible


def parking_stability(env, command_name='base_velocity', linear_scale=.15,
                      angular_scale=.3, tilt_scale=.35,
                      command_threshold=.05, contact_threshold=5.):
    command, m, eligible = _parking_inputs(env, command_name)
    return parking_stability_values(
        command, m['lin'], m['ang'], m['gravity'], m['contact_fz'],
        m['supported'], m['corner_supported'], eligible, linear_scale,
        angular_scale, tilt_scale, command_threshold, contact_threshold)


def parking_pose_values(command, joint_position, default_joint_position,
                        contact_fz, sole_supported, corner_supported, eligible,
                        scale=.35, command_threshold=.05, contact_threshold=5.):
    """Mean normalized pose deviation in [0, 4], exclusively during safe stop."""
    _positive(scale, 'scale')
    active = parking_gate_values(command, contact_fz, sole_supported, corner_supported,
                                 eligible, command_threshold, contact_threshold)
    error = (joint_position - default_joint_position) / scale
    error = torch.nan_to_num(error, nan=2., posinf=2., neginf=-2.)
    penalty = error.clamp(-2., 2.).square().mean(-1)
    return torch.where(active, penalty, 0.)


def parking_pose_l2(env, command_name='base_velocity', asset_cfg=None,
                    scale=.35, command_threshold=.05, contact_threshold=5.):
    command, m, eligible = _parking_inputs(env, command_name)
    asset_name = 'robot' if asset_cfg is None else asset_cfg.name
    joint_ids = slice(None) if asset_cfg is None else asset_cfg.joint_ids
    data = env.scene[asset_name].data
    return parking_pose_values(
        command, tensor(data.joint_pos)[:, joint_ids],
        tensor(data.default_joint_pos)[:, joint_ids], m['contact_fz'],
        m['supported'], m['corner_supported'], eligible,
        scale, command_threshold, contact_threshold)


def diagnostics(env, command_name='base_velocity'):
    """Current yaw/parking quality, without adding privileged policy inputs."""
    command, m, eligible = _parking_inputs(env, command_name)
    straight = straight_command_mask(command)
    standing = stand_command_mask(command)
    parking = parking_gate_values(command, m['contact_fz'], m['supported'],
                                  m['corner_supported'], eligible)
    rate = imu_yaw_rate(m['ang'], m['gravity'])
    quality = parking_stability_values(command, m['lin'], m['ang'], m['gravity'],
                                      m['contact_fz'], m['supported'],
                                      m['corner_supported'], eligible)

    def mean(value, mask):
        valid = mask & torch.isfinite(value)
        return torch.where(valid, torch.nan_to_num(value, posinf=0., neginf=0.), 0.).sum() / valid.sum().clamp_min(1)

    return {
        'GaitPreservation/straight_fraction': straight.float().mean(),
        'GaitPreservation/straight_yaw_rate_signed_rad_s': mean(rate, straight),
        'GaitPreservation/straight_yaw_rate_abs_rad_s': mean(rate.abs(), straight),
        'GaitPreservation/stand_command_fraction': standing.float().mean(),
        'GaitPreservation/safe_double_support_fraction': mean(parking.float(), standing),
        'GaitPreservation/parking_linear_speed_m_s': mean(m['lin'].norm(dim=-1), parking),
        'GaitPreservation/parking_angular_speed_rad_s': mean(m['ang'].norm(dim=-1), parking),
        'GaitPreservation/parking_stability': mean(quality, parking),
    }
