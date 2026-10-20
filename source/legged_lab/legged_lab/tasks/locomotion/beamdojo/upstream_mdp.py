"""ezoicoder/humanoid-gym observation/command/reset adapters for Unitree G1.

Source: ezoicoder/humanoid-gym @ 9a47b7dfe85fdcc737c3c8963d80d05997376575,
humanoid/envs/custom/humanoid_env.py and base/legged_robot.py.
G1 default pose, joint count, PD and physics clock remain Unitree's. No new
reward coefficients or hand-designed motion targets are introduced here.
"""
import math
import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from ..unitree_official import mdp as native
from . import repo_mdp

tensor = native.tensor


class UpstreamVelocityCommand(UniformVelocityCommand):
    """Source _resample_commands and _post_physics_step_callback, including deadband."""
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        xy = self.vel_command_b[env_ids, :2]
        self.vel_command_b[env_ids, :2] = xy * (xy.norm(dim=-1) > .2)[:, None]

    def _update_command(self):
        if not self.cfg.heading_command:
            return
        error = self.heading_target - tensor(self.robot.data.heading_w)
        error = (error + math.pi) % (2 * math.pi) - math.pi
        self.vel_command_b[:, 2] = (.5 * error).clamp(-1., 1.)


class UpstreamResetRoot(native.reset_root_state_uniform):
    """Sim6 stateful reset API; ranges supplied by source profile config."""
    def __call__(self, env, env_ids, pose_range, velocity_range):
        from . import upstream_terrain, recovery_curriculum
        upstream_terrain.ensure_state(env)
        upstream_terrain.refresh_geometry(env)
        super().__call__(env, env_ids, recovery_curriculum.yaw_pose_range(env, pose_range), velocity_range)


def base_contact(env):
    # Preserve XBot's root-contact rule through G1's pelvis mapping. This is
    # supplementary: separate torso/head colliders can support a fallen G1
    # with no pelvis contact; Unitree height/orientation terms cover that case.
    sensor = env.scene.sensors['contact_forces']
    # Sim6 enumerates PhysX prim paths during name resolution (~90 ms per
    # step for 4096 robots). Body IDs stay fixed through resets and curricula.
    if not hasattr(env, '_upstream_base_contact_ids'):
        env._upstream_base_contact_ids = sensor.find_bodies('pelvis')[0]
        if len(env._upstream_base_contact_ids) != 1:
            raise ValueError('G1 base contact requires exactly one pelvis body')
    ids = env._upstream_base_contact_ids
    return (tensor(sensor.data.net_forces_w)[:, ids].norm(dim=-1) > 1.).any(dim=-1)


def time_out(env):
    # Preserve the source strict > comparison.
    return env.episode_length_buf > env.max_episode_length


def push_robot(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, '_upstream_push'):
        env._upstream_push = torch.zeros(env.num_envs, 5, device=env.device)
    pushes = torch.empty(len(env_ids), 5, device=env.device)
    pushes[:, :2].uniform_(-.2, .2)
    pushes[:, 2:].uniform_(-.4, .4)
    env._upstream_push[env_ids] = pushes
    robot = env.scene['robot']
    velocity = tensor(robot.data.root_vel_w)[env_ids].clone()
    velocity[:, :2] = pushes[:, :2]
    velocity[:, 3:] = pushes[:, 2:]
    robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)


def _parts(env):
    data = env.scene['robot'].data
    phase = repo_mdp.phase(env)
    if not hasattr(env, '_upstream_command_scale'):
        env._upstream_command_scale = torch.tensor([2., 2., 1.], device=env.device)
    cmd = env.command_manager.get_command('base_velocity') * env._upstream_command_scale
    return data, phase, cmd


def actor_frame(env):
    """Source actor ordering; 29-DoF expansion gives 98 values per frame."""
    d, phase, command = _parts(env)
    q = tensor(d.joint_pos) - tensor(d.default_joint_pos)
    dq = tensor(d.joint_vel) * .05
    angular = tensor(d.root_ang_vel_b)
    gravity = tensor(d.projected_gravity_b)
    if env.cfg.observations.policy.enable_corruption:
        # Actual upstream compute_observations uses Gaussian noise, despite
        # its stale _get_noise_scale_vec docstring calling this uniform.
        q = q + torch.randn_like(q) * (.05 * .6)
        dq = dq + torch.randn_like(dq) * (.5 * .05 * .6)
        angular = angular + torch.randn_like(angular) * (.1 * .6)
        gravity = gravity + torch.randn_like(gravity) * (.03 * .6)
    return torch.cat((phase, command, q, dq, env.action_manager.action, angular, gravity), -1).clamp(-18., 18.)


def critic_frame(env):
    """Source privileged ordering, expanded from 12 to 29 joints (141 values)."""
    d, phase, command = _parts(env)
    robot = env.scene['robot']
    q = tensor(d.joint_pos)
    default = tensor(d.default_joint_pos)
    if not hasattr(env, '_upstream_observation_indices'):
        env._upstream_observation_indices = (
            robot.find_joints(list(repo_mdp.LEG_NAMES), preserve_order=True)[0],
            env.scene.sensors['contact_forces'].find_bodies(
                ['left_ankle_roll_link', 'right_ankle_roll_link'], preserve_order=True)[0],
            robot.find_bodies('pelvis')[0][0])
    legs, feet, root_id = env._upstream_observation_indices
    moving = repo_mdp.command_moving(env.command_manager.get_command('base_velocity'))
    reference = default.clone()
    reference[:, legs] = repo_mdp.reference_positions(default[:, legs], phase[:, 0], moving)
    stance = repo_mdp.stance_mask(phase[:, 0], moving)
    sensor = env.scene.sensors['contact_forces']
    contact = tensor(sensor.data.net_forces_w)[:, feet, 2] > 5.
    if not hasattr(env, '_upstream_push'):
        env._upstream_push = torch.zeros(env.num_envs, 5, device=env.device)
    # Observation managers probe dimensions before startup events. Read the
    # post-event coefficient cache, never freeze that pre-startup probe value.
    friction = getattr(env, '_upstream_friction', None)
    if friction is None:
        import warp as wp
        friction = wp.to_torch(robot.root_view.get_material_properties())[:, 0, 0:1].to(env.device)
    # Source privileged mass is the root body's mass (G1 randomization acts
    # on the torso instead, as specified by the Unitree body-mass event).
    mass = getattr(env, '_upstream_root_mass_observation', None)
    if mass is None:
        mass = tensor(d.body_mass)[:, root_id:root_id+1] / 30.
        # The friction startup event proves startup has begun; root mass is
        # unchanged by the subsequent Unitree torso-only mass event. Never
        # freeze values during observation-manager dimension probing.
        if hasattr(env, '_upstream_friction'):
            env._upstream_root_mass_observation = mass.clone()
    return torch.cat((phase, command, q-default, tensor(d.joint_vel)*.05,
        env.action_manager.action, q-reference, tensor(d.root_lin_vel_b)*2.,
        tensor(d.root_ang_vel_b), euler_xyz(tensor(d.root_quat_w)),
        env._upstream_push, friction, mass, stance.float(), contact.float()), -1).clamp(-18., 18.)


def euler_xyz(quaternion):
    x, y, z, w = quaternion.unbind(-1)
    roll_pitch = repo_mdp.euler_xyzw(quaternion)
    yaw = torch.atan2(2 * (w*z+x*y), 1-2*(y*y+z*z))
    return torch.cat((roll_pitch, yaw[:, None]), -1)


def randomize_friction(env, env_ids):
    """Source _process_rigid_shape_props: one of 256 coefficients per robot."""
    import warp as wp
    ids = (torch.arange(env.num_envs, dtype=torch.int32) if env_ids is None
           else env_ids.to(device='cpu', dtype=torch.int32))
    if not hasattr(env, '_upstream_friction_buckets'):
        env._upstream_friction_buckets = torch.empty(256).uniform_(.1, 2.)
    coefficient = env._upstream_friction_buckets[torch.randint(256, (len(ids),))]
    if not hasattr(env, '_upstream_friction'):
        env._upstream_friction = torch.zeros(env.num_envs, 1, device=env.device)
    env._upstream_friction[ids.long().to(env.device), 0] = coefficient.to(env.device)
    view = env.scene['robot'].root_view
    materials = wp.to_torch(view.get_material_properties())
    materials[ids.long(), :, :2] = coefficient[:, None, None]
    materials[ids.long(), :, 2] = 0.
    view.set_material_properties(wp.from_torch(materials, dtype=wp.float32),
                                 wp.from_torch(ids, dtype=wp.int32))


def observation_history(env, name, frame, length):
    """Source deque semantics: zero prefix, oldest -> newest, partial resets.

    Native CircularBuffer repeats the first append through its empty history;
    humanoid_env.reset_idx instead clears history slots to zero. One update per
    environment per transition also avoids double appends during final-observation
    probes and repeated observation reads in the same physics tick.
    """
    if not hasattr(env, '_upstream_observation_histories'):
        env._upstream_observation_histories = {}
    histories = env._upstream_observation_histories
    if name not in histories:
        histories[name] = (
            frame.new_zeros(env.num_envs, length, frame.shape[-1]),
            torch.full((env.num_envs,), -1, dtype=torch.long, device=frame.device))
    history, updated = histories[name]
    tick = env.common_step_counter
    if not hasattr(env, '_upstream_history_ticks'):
        env._upstream_history_ticks = {}
        env._upstream_history_pending = {}
    previous_tick = env._upstream_history_ticks.get(name)
    if previous_tick != tick:
        history[:, :-1] = history[:, 1:].clone()
        history[:, -1] = frame
        updated.fill_(tick)
    else:
        pending = env._upstream_history_pending.get(name, [])
        if pending:
            ids = torch.cat(pending)
            history[ids, :-1] = history[ids, 1:].clone()
            history[ids, -1] = frame[ids]
            updated[ids] = tick
    env._upstream_history_ticks[name] = tick
    env._upstream_history_pending[name] = []
    return history.flatten(1)


def reset_observation_history(env, env_ids):
    for name, (history, updated) in getattr(env, '_upstream_observation_histories', {}).items():
        history[env_ids] = 0.
        updated[env_ids] = -1
        if hasattr(env, '_upstream_history_pending'):
            env._upstream_history_pending.setdefault(name, []).append(env_ids.clone())
    env._upstream_critic_frame_cache = None
    env._upstream_heightmap_cache = None


def actor_observation(env):
    return observation_history(env, 'actor', actor_frame(env), 15)


def critic_observation(env):
    cached = getattr(env, '_upstream_critic_frame_cache', None)
    if cached is None or cached[0] != env.common_step_counter:
        cached = (env.common_step_counter, critic_frame(env))
        env._upstream_critic_frame_cache = cached
    return observation_history(env, 'critic', cached[1], 3)
