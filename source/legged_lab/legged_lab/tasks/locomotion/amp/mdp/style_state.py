"""Shared physical state encoding for agent and reference AMP observations."""
import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_inverse


def tensor(x):
    return x if isinstance(x, torch.Tensor) else x.torch


def encode_style(quat, joint_pos, joint_vel, lin_vel_w, ang_vel_w, default_pos):
    """XYZW; [gravity(3), q-relative(29), .05*qdot(29), v-body(3), w-body(3)]."""
    gravity = torch.zeros_like(lin_vel_w)
    gravity[..., 2] = -1.
    return torch.cat((quat_apply_inverse(quat, gravity), joint_pos - default_pos,
                      .05 * joint_vel, quat_apply_inverse(quat, lin_vel_w),
                      quat_apply_inverse(quat, ang_vel_w)), dim=-1)


def joint_mirror_map(names, device):
    """Derive bilateral joint permutation/signs from named G1 axes, not indices."""
    indices, signs = [], []
    for name in names:
        other = ('right_' + name[5:]) if name.startswith('left_') else (
            'left_' + name[6:] if name.startswith('right_') else name)
        indices.append(names.index(other))
        signs.append(-1. if ('_roll_' in name or '_yaw_' in name) else 1.)
    return torch.tensor(indices, device=device), torch.tensor(signs, device=device)


def mirror_style(state, indices, signs):
    """Reflect an entire encoded trajectory through body Y=0, preserving time."""
    n = len(indices)
    polar = state.new_tensor([1., -1., 1.])
    axial = state.new_tensor([-1., 1., -1.])
    return torch.cat((state[..., :3] * polar,
                      state[..., 3:3+n][..., indices] * signs,
                      state[..., 3+n:3+2*n][..., indices] * signs,
                      state[..., 3+2*n:6+2*n] * polar,
                      state[..., 6+2*n:9+2*n] * axial), -1)


def agent_style_state(env):
    data = env.scene['robot'].data
    # Reference velocity is the derivative of the root-link position, not COM.
    return encode_style(tensor(data.root_quat_w), tensor(data.joint_pos), tensor(data.joint_vel),
                        tensor(data.root_link_lin_vel_w), tensor(data.root_ang_vel_w), tensor(data.default_joint_pos))


class ReferenceStyleState(ManagerTermBase):
    """50% trajectory reflection, fixed per episode; no camera-image mirroring."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        robot = env.scene['robot']
        self.indices, self.signs = joint_mirror_map(robot.joint_names, env.device)
        default = tensor(robot.data.default_joint_pos)
        # Reflection of relative q is valid only for a symmetric default pose.
        torch.testing.assert_close(default[:, self.indices] * self.signs, default)
        self.mirrored = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.reset()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        self.mirrored[env_ids] = torch.rand(len(env_ids), device=self._env.device) < .5

    def __call__(self, env, animation='animation'):
        ref = env.animation_manager.get_term(animation)
        default = tensor(env.scene['robot'].data.default_joint_pos)[:, None]
        state = encode_style(ref.get_root_quat(), ref.get_dof_pos(), ref.get_dof_vel(),
                             ref.get_root_vel_w(), ref.get_root_ang_vel_w(), default)
        mirrored = mirror_style(state, self.indices, self.signs)
        return torch.where(self.mirrored[:, None, None], mirrored, state)
