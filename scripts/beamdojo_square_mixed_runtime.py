"""Per-environment mixed square routes and reset-time curriculum (CPU import safe)."""
import math

import torch

import beamdojo_square_course_runtime as base


tensor = base.tensor
_COUNT_NAMES = ('episodes', 'successes', 'failures', 'timeouts', 'promotions', 'demotions')


def ensure_state(env):
    initialize = globals().get('initialize_geometry')
    if initialize is not None and not hasattr(env, '_square_mixed_grids'):
        initialize(env)
    state = base.ensure_state(env)
    if hasattr(env, '_square_mixed_state'):
        return state
    n, device = env.num_envs, env.device
    ids = torch.arange(n, device=device)
    defaults = {
        'beamdojo_levels': torch.full_like(ids, int(getattr(env.cfg, 'beamdojo_initial_level', 0))),
        'square_family': (ids % 8) // 4,
        'square_route': ids % 4,
        'square_variant': torch.zeros_like(ids),
        'square_win_streak': torch.zeros_like(ids),
        'square_fail_streak': torch.zeros_like(ids),
    }
    for name, value in defaults.items():
        if not hasattr(env, name):
            setattr(env, name, value)
    env._square_mixed_state = {
        'last_reset_tick': torch.full_like(ids, -1),
        **{name: torch.zeros(8, 11, dtype=torch.long, device=device) for name in _COUNT_NAMES},
    }
    refresh_angles(env)
    return state


def refresh_angles(env):
    angles = torch.tensor([[0., math.pi / 2, math.pi, -math.pi / 2],
                           [math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4, -math.pi / 4]],
                          device=env.device)
    env.square_angles = angles[env.square_family, env.square_route]


def _ids(env, ids):
    if ids is None or isinstance(ids, slice):
        return torch.arange(env.num_envs, device=env.device)[slice(None) if ids is None else ids]
    return torch.as_tensor(ids, device=env.device, dtype=torch.long).flatten()


def goal_start(env):
    return torch.where(env.square_family == 0, 3., 5.5 / math.sqrt(2))


def route_frame(env, value):
    """Rotate tile-local vectors into each environment's assigned route frame."""
    shape = (env.num_envs,) + (1,) * (value.ndim - 2)
    cosine = env.square_angles.cos().reshape(shape)
    sine = env.square_angles.sin().reshape(shape)
    result = value.clone()
    result[..., 0] = cosine * value[..., 0] + sine * value[..., 1]
    result[..., 1] = -sine * value[..., 0] + cosine * value[..., 1]
    return result


def read_measurements(env):
    m = base.read_measurements(env)
    callback = getattr(env, '_square_course_surface_height', None)
    if callback is None:
        raise ValueError('Mixed square routes require exact physical surface queries')
    origin = tensor(env.scene.env_origins)
    world_corners = m['corners'] + origin[:, None, None, :]
    height, supported = callback(world_corners)
    m['corner_supported'] = supported & ((world_corners[..., 2] - height).abs() <= .035)
    return m


def _geometry(env, m):
    root = route_frame(env, m['root'])
    corners = route_frame(env, m['corners'])
    goal = goal_start(env)
    full = ((corners[..., 0] >= goal[:, None, None])
            & (corners[..., 1].abs() <= .6)
            & (corners[..., 2].abs() <= .035)
            & (m['corners'][..., :2].abs() < 4.).all(-1)).all((-1, -2))
    # CPU measurement callers must also supply explicit physical support truth.
    full &= m['supported'].all(-1) & m['corner_supported'].all((-1, -2))
    return root, corners, full


def update_course(env, measurements=None):
    import sys
    from beamdojo_square_terminal import update_course_impl
    state=update_course_impl(env,measurements,sys.modules[__name__])
    from beamdojo_flat_rehearsal import apply_flat_course_state,flat_mask
    if bool(flat_mask(env).any()):apply_flat_course_state(env,state,read_measurements(env) if measurements is None else measurements,tensor)
    return state


def parking_command(env, cruise, root_s, full_ring):
    from beamdojo_square_terminal import parking_command_impl
    return parking_command_impl(env,cruise,root_s,full_ring,ensure_state,goal_start)


def make_command_class(_base_class=None):
    if _base_class is None:
        from isaaclab.envs.mdp.commands import UniformVelocityCommand
        _base_class = UniformVelocityCommand

    class MixedSquareVelocityCommand(_base_class):
        def _resample_command(self, env_ids):
            env_ids=_ids(self._env,env_ids)
            super()._resample_command(env_ids)
            state = ensure_state(self._env)
            if not hasattr(self, '_square_cruise'):
                self._square_cruise = torch.full((self._env.num_envs,), .75, device=self.vel_command_b.device)
            self._square_cruise[env_ids] = torch.empty_like(self._square_cruise[env_ids]).uniform_(.5, 1.)
            self.vel_command_b[env_ids] = 0.
            self.vel_command_b[env_ids, 0] = torch.where(state['parked'][env_ids], 0., self._square_cruise[env_ids])
            from beamdojo_flat_rehearsal import flat_mask,sample_commands
            if not hasattr(self,'_flat_command'):self._flat_command=torch.zeros_like(self.vel_command_b)
            flat_ids=env_ids[flat_mask(self._env)[env_ids]]
            if flat_ids.numel():
                self._flat_command[flat_ids]=sample_commands(self._env.flat_rehearsal_mode[flat_ids])
                if hasattr(self,'time_left'):self.time_left[flat_ids]=torch.empty(len(flat_ids),device=self.vel_command_b.device).uniform_(4.,8.)
            self.vel_command_b[flat_ids]=self._flat_command[flat_ids]

        def _update_command(self):
            ensure_state(self._env)
            m = read_measurements(self._env)
            root, _, full = _geometry(self._env, m)
            if not hasattr(self, '_square_cruise'):
                self._square_cruise = self.vel_command_b[:, 0].clone().clamp(.5, 1.)
            self.vel_command_b[:] = parking_command(self._env, self._square_cruise, root[:, 0], full)
            from beamdojo_flat_rehearsal import flat_mask
            flat=flat_mask(self._env)
            if hasattr(self,'_flat_command'):self.vel_command_b[flat]=self._flat_command[flat]
            self._env._square_course_state['parked'][flat]=False

    return MixedSquareVelocityCommand


def before_reset(env, ids):
    ensure_state(env)
    ids = _ids(env, ids).unique()
    state = env._square_mixed_state
    tick = int(env.common_step_counter)
    ids = ids[state['last_reset_tick'][ids] != tick]
    if ids.numel() == 0:
        return {}
    levels = env.beamdojo_levels[ids].clone()
    bins = env.square_family[ids] * 4 + env.square_route[ids]
    experienced = env._upstream_started[ids]
    expired = tensor(env.episode_length_buf)[ids] >= env.max_episode_length
    failed = env.beamdojo_course_failed[ids]
    wins = env.beamdojo_qualified_success[ids] & ~failed & ~expired
    eligible = experienced & (env.beamdojo_qualified_success[ids] | failed | expired)
    from beamdojo_flat_rehearsal import flat_mask
    eligible &= ~flat_mask(env)[ids]
    skip = getattr(env, '_square_skip_curriculum_once', None)
    if skip is not None:
        eligible &= ~skip[ids]
        skip[ids] = False
    losses = eligible & ~wins
    win_streak = torch.where(wins, env.square_win_streak[ids] + 1, 0)
    fail_streak = torch.where(wins, 0, env.square_fail_streak[ids] + 1)
    win_streak = torch.where(eligible, win_streak, env.square_win_streak[ids])
    fail_streak = torch.where(eligible, fail_streak, env.square_fail_streak[ids])
    promote = eligible & (win_streak >= 3) & (levels < 10)
    demote = losses & (fail_streak >= 2) & (levels > 0)
    env.beamdojo_levels[ids] = (levels + promote.long() - demote.long()).clamp(0, 10)
    # Cap streaks at their trigger, including at the difficulty endpoints.
    env.square_win_streak[ids] = torch.where(promote | demote, 0, win_streak.clamp_max(3))
    env.square_fail_streak[ids] = torch.where(promote | demote, 0, fail_streak.clamp_max(2))
    count_values = dict(episodes=eligible, successes=eligible & wins, failures=eligible & failed,
                        timeouts=eligible & expired & ~failed, promotions=promote, demotions=demote)
    for name, value in count_values.items():
        state[name].view(-1).scatter_add_(0, bins * 11 + levels, value.long())
    state['last_reset_tick'][ids] = tick
    choose = getattr(env, 'square_choose_variants', None)
    if choose is not None:
        choose(ids)
    refresh_angles(env)
    env.square_refresh_geometry(env, ids)
    base.reset_course(env, ids)
    return {
        'Metrics/course_success_rate': (eligible & wins).float().sum() / eligible.sum().clamp_min(1),
        'Curriculum/square_promotions': promote.float().sum(),
        'Curriculum/square_demotions': demote.float().sum(),
        'Curriculum/square_mean_level': env.beamdojo_levels.float().mean(),
    }


def reset_root_state(env, env_ids, pose_range=None, velocity_range=None, asset_cfg=None):
    """Native Sim6 XYZW reset; selected route yaw is evaluated at every reset."""
    from isaaclab.utils import math as math_utils
    ensure_state(env)
    ids = _ids(env, env_ids)
    asset = env.scene['robot' if asset_cfg is None else asset_cfg.name]
    default_pose = tensor(asset.data.default_root_pose)[ids]
    positions = default_pose[:, :3] + tensor(env.scene.env_origins)[ids]
    zero = torch.zeros(ids.numel(), device=env.device)
    rotation = math_utils.quat_from_euler_xyz(zero, zero, env.square_angles[ids])
    orientations = math_utils.quat_mul(default_pose[:, 3:7], rotation)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat((positions, orientations), -1), env_ids=ids)
    asset.write_root_velocity_to_sim_index(root_velocity=tensor(asset.data.default_root_vel)[ids].clone(), env_ids=ids)


def course_success(env):
    update_course(env)
    return env._course_finished


def course_boundary(env):
    update_course(env)
    return env._course_boundary


def time_out(env):
    update_course(env)
    from beamdojo_flat_rehearsal import neutral_boundary
    return ((tensor(env.episode_length_buf) >= env.max_episode_length)|neutral_boundary(env)) & ~env.beamdojo_course_failed


def course_metrics(env):
    ensure_state(env)
    metrics = base.course_metrics(env)
    from beamdojo_flat_rehearsal import flat_mask
    stone=~flat_mask(env)
    metrics['Curriculum/square_mean_level'] = (env.beamdojo_levels.float()*stone).sum()/stone.sum().clamp_min(1)
    metrics['Curriculum/mean_level']=metrics['Curriculum/square_mean_level']
    counts = env._square_mixed_state
    for family in range(2):
        for route in range(4):
            from beamdojo_flat_rehearsal import flat_mask
            selected = (env.square_family == family) & (env.square_route == route) & ~flat_mask(env)
            prefix = f'Square/family_{family}_route_{route}'
            metrics[prefix + '_mean_level'] = (env.beamdojo_levels * selected).sum().float() / selected.sum().clamp_min(1)
            episodes = counts['episodes'][family * 4 + route].sum()
            metrics[prefix + '_success_rate'] = counts['successes'][family * 4 + route].sum().float() / episodes.clamp_min(1)
    from beamdojo_flat_rehearsal import flat_mask,MODE_NAMES
    flat=flat_mask(env)
    if bool(flat.any()):
        command=tensor(env.command_manager.get_command('base_velocity'));robot=env.scene['robot'].data
        lin=tensor(robot.root_lin_vel_b);ang=tensor(robot.root_ang_vel_b)
        for i,name in enumerate(MODE_NAMES):
            ids=flat&(env.flat_rehearsal_mode==i)
            if bool(ids.any()):
                metrics['Flat/'+name+'_xy_velocity_error']=(lin[ids,:2]-command[ids,:2]).norm(dim=-1).mean()
                metrics['Flat/'+name+'_yaw_rate_error']=(ang[ids,2]-command[ids,2]).abs().mean()
    return metrics


def snapshot(env):
    ensure_state(env)
    names = ('beamdojo_levels', 'square_family', 'square_route', 'square_variant',
             'square_win_streak', 'square_fail_streak')
    return dict(version=1, num_envs=env.num_envs,
                assignments={name: getattr(env, name).detach().cpu().tolist() for name in names},
                counters={name: env._square_mixed_state[name].detach().cpu().tolist() for name in _COUNT_NAMES})


def restore(env, saved):
    """Restore curriculum before a neutral env.reset; do not restore live episodes."""
    ensure_state(env)
    if saved.get('version') != 1 or saved.get('num_envs') != env.num_envs:
        raise ValueError('Mixed curriculum checkpoint version/environment count mismatch')
    bounds = dict(beamdojo_levels=10, square_family=1, square_route=3, square_variant=3,
                  square_win_streak=3, square_fail_streak=2)
    staged = {}
    for name, maximum in bounds.items():
        value = torch.as_tensor(saved['assignments'][name], dtype=torch.long, device=env.device)
        if value.shape != (env.num_envs,) or bool(((value < 0) | (value > maximum)).any()):
            raise ValueError(f'Invalid mixed curriculum checkpoint field: {name}')
        staged[name] = value
    counters = {}
    for name in _COUNT_NAMES:
        value = torch.as_tensor(saved['counters'][name], dtype=torch.long, device=env.device)
        if value.shape != (8, 11) or bool((value < 0).any()):
            raise ValueError(f'Invalid mixed curriculum counter: {name}')
        counters[name] = value
    for name, value in staged.items():
        getattr(env, name).copy_(value)
    for name, value in counters.items():
        env._square_mixed_state[name].copy_(value)
    env._square_mixed_state['last_reset_tick'].fill_(-1)
    env._square_skip_curriculum_once = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    refresh_angles(env)
    env.square_refresh_geometry(env, None)
    base.reset_course(env, None)
