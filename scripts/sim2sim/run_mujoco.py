"""Run LeggedLab G1 visual CNN checkpoints in the hiking MuJoCo scenes, without Isaac Sim."""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
TRAINING_SCENE = ROOT / 'scripts/sim2sim/scenes/training_stairs_12cm.xml'
sys.path.insert(0, str(ROOT / '.runtime/rsl_rl_5_4_1'))
from rsl_rl.models import CNNModel
from tensordict import TensorDict


class ConfigLoader(yaml.SafeLoader):
    pass


# Config dumps contain tuples, but must never instantiate arbitrary Python objects.
ConfigLoader.add_constructor('tag:yaml.org,2002:python/tuple', lambda loader, node: loader.construct_sequence(node))
ConfigLoader.add_constructor('tag:yaml.org,2002:python/object/apply:builtins.slice', lambda loader, node: loader.construct_sequence(node))


def read_config(path):
    return yaml.load(Path(path).read_text(), Loader=ConfigLoader)


def resolve(value, name, default=None):
    if not isinstance(value, dict):
        return value
    matches = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
    if len(matches) > 1:
        raise ValueError(f'Ambiguous configuration for {name}')
    if not matches and default is None:
        raise ValueError(f'Missing configuration for {name}')
    return matches[0] if matches else default


def rotation(quat_wxyz):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(quat_wxyz, dtype=np.float64))
    return result.reshape(3, 3)


def orientation_features(matrix):
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ matrix
    return np.concatenate((local[:, 0], local[:, 2]))


def target_command(position, matrix, target, speed, config):
    """Match the training forward-only target controller (zero lateral speed)."""
    delta = np.asarray(target) - position[:2]
    distance = np.linalg.norm(delta)
    if distance <= config['target_dis_threshold']:
        return np.zeros(3)
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    error = np.arctan2(delta[1], delta[0]) - yaw
    error = np.arctan2(np.sin(error), np.cos(error))
    forward = np.cos(yaw) * delta[0] + np.sin(yaw) * delta[1]
    vx = min(max(forward * config['velocity_control_stiffness'], 0.), speed)
    if config.get('heading_slowdown', False):
        vx *= np.clip((np.deg2rad(70) - abs(error)) / np.deg2rad(50), 0., 1.)
    return np.array([vx, 0., np.clip(error * config['heading_control_stiffness'], -config['yaw_limit'], config['yaw_limit'])])


def calibrated_model(scene, profile):
    """Keep the terrain, replacing robot geometry with the training USD shapes."""
    root = ET.parse(scene).getroot()
    for include in list(root.findall('include')):
        source = (scene.parent / include.attrib['file']).resolve()
        robot = ET.parse(source).getroot()
        compiler = robot.find('compiler')
        meshdir = (source.parent / compiler.attrib.get('meshdir', '')).resolve()
        for asset in robot.findall('asset/mesh'):
            asset.set('file', str(meshdir / asset.attrib['file']))
        root.remove(include)
        for child in list(robot):
            root.append(child)
    bodies = {body.attrib['name']: body for body in root.iter('body')}
    for name in profile['bodies']:
        for geom in bodies[name].findall('geom'):
            if profile.get('visuals') or geom.attrib.get('contype') != '0':
                bodies[name].remove(geom)
    asset = ET.SubElement(root, 'asset')
    for visual in profile.get('visuals', []):
        name = visual['mesh']
        ET.SubElement(asset, 'mesh', name=name, file=str(Path(__file__).parent / 'assets' / f'{name}.stl'))
        ET.SubElement(bodies[visual['body']], 'geom', name=name, type='mesh', mesh=name,
                      contype='0', conaffinity='0', density='0', group=str(visual['group']), rgba='.65 .67 .7 1')
    for i, shape in enumerate(profile['collisions']):
        attributes = {'name': f'training_collision_{i}', 'type': shape['type'], 'group': '2'}
        for key in ('pos', 'quat', 'size'):
            if key in shape:
                attributes[key] = ' '.join(map(str, shape[key]))
        if 'mesh' in shape:
            name = shape['mesh']
            ET.SubElement(asset, 'mesh', name=name, file=str(Path(__file__).parent / 'assets' / f'{name}.obj'))
            attributes['mesh'] = name
        ET.SubElement(bodies[shape['body']], 'geom', attributes)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def preprocess_depth(raw):
    x = torch.as_tensor(raw, dtype=torch.float32)[18:, 16:-16]
    x = torch.nan_to_num(x, nan=2.5, posinf=2.5, neginf=0.)
    x = torch.where((x > 0) & (x < .1), 2.5, x).clamp(0, 2.5)[None, None]
    axis = torch.arange(-1, 2, dtype=x.dtype)
    k = torch.exp(-.5 * axis.square()); k /= k.sum()
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), (k[:, None] * k[None, :])[None, None])[0, 0] / 2.5


def configure_geometry_groups(model):
    """0: terrain, 1: sensed visuals, 2: collisions, 3: display-only visuals."""
    model.geom_group[(model.geom_bodyid != 0) & ~np.isin(model.geom_group, [1, 3])] = 2
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            if model.mesh(mesh).name in {'head_link', 'waist_support_link'}:
                model.geom_group[geom] = 3


class Histories:
    def __init__(self, delay=0):
        self.policy = None
        self.depth = None
        self.delay = delay

    def update(self, terms, frame):
        if self.policy is None:
            self.policy = [np.repeat(np.asarray(t)[None], 5, axis=0) for t in terms]
            self.depth = frame.repeat(37, 1, 1)
        else:
            for history, term in zip(self.policy, terms, strict=True):
                history[:-1] = history[1:]; history[-1] = term
            self.depth = torch.roll(self.depth, -1, 0)
            self.depth[-1] = frame
        policy = torch.tensor(np.concatenate([h.ravel() for h in self.policy]), dtype=torch.float32)[None]
        depth = self.depth[36 - torch.arange(35, -1, -5) - self.delay][None]
        return TensorDict({'policy': policy, 'depth': depth}, batch_size=[1])


class Runner:
    def __init__(self, checkpoint, scene, spawn, yaw, delay):
        cfg = read_config(checkpoint.parent / 'params/env.yaml')
        agent = read_config(checkpoint.parent / 'params/agent.yaml')
        self.command_config = cfg['commands']['base_velocity']
        terms = cfg['observations']['policy']
        expected = ['base_ang_vel', 'root_local_rot_tan_norm', 'velocity_commands', 'joint_pos', 'joint_vel', 'actions']
        active = [k for k, v in terms.items() if isinstance(v, dict) and 'func' in v]
        if active != expected or any(terms[k]['history_length'] != 5 or terms[k]['scale'] is not None or terms[k]['clip'] is not None for k in active):
            raise ValueError('Unsupported policy observations; expected the unscaled 5-frame G1 visual policy.')
        if not cfg['observations']['depth']['image']['func'].endswith(':InstinctDepthHistory'):
            raise ValueError('This runner requires InstinctDepthHistory (8 x 18 x 32).')
        action_cfg = cfg['actions']['joint_pos']
        if action_cfg['joint_names'] != ['.*'] or not action_cfg['use_default_offset'] or action_cfg['clip'] is not None or agent['clip_actions'] is not None:
            raise ValueError('Unsupported action configuration')
        self.scale = float(action_cfg['scale'])
        self.names = read_config(ROOT / 'scripts/tools/retarget/config/g1_29dof.yaml')['lab_dof_names']
        profile = json.loads(Path(__file__).with_name('g1_training_profile.json').read_text())
        self.model = m = calibrated_model(scene, profile)
        self.data = d = mujoco.MjData(m)
        if Path(cfg['scene']['robot']['spawn']['usd_path']).name != Path(profile['source']).name:
            raise ValueError('Robot USD does not match the calibrated G1 profile')
        for name, body in profile['bodies'].items():
            bid = m.body(name).id
            m.body_mass[bid] = body['mass']
            m.body_ipos[bid] = body['ipos']
            m.body_iquat[bid] = body['iquat']
            m.body_inertia[bid] = body['inertia']
        for name, joint in profile['joints'].items():
            bid = m.jnt_bodyid[m.joint(name).id]
            m.body_pos[bid] = joint['pos']
            m.body_quat[bid] = joint['quat']
        m.opt.timestep = cfg['sim']['dt']
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.decimation = cfg['decimation']
        self.dt = m.opt.timestep * self.decimation
        ids = np.array([m.joint(n).id for n in self.names])
        self.qadr, self.vadr = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
        self.actids = np.array([np.flatnonzero(m.actuator_trnid[:, 0] == j).item() for j in ids])
        robot = cfg['scene']['robot']
        self.default = np.array([resolve(robot['init_state']['joint_pos'], n, 0.) for n in self.names])
        self.kp, self.kd, self.limits = [], [], []
        for name, jid, aid, vid in zip(self.names, ids, self.actids, self.vadr, strict=True):
            groups = [a for a in robot['actuators'].values() if any(re.fullmatch(p, name) for p in a['joint_names_expr'])]
            if len(groups) != 1:
                raise ValueError(f'Expected one actuator configuration: {name}')
            a = groups[0]
            self.kp.append(resolve(a['stiffness'], name)); self.kd.append(resolve(a['damping'], name))
            limit = resolve(a['effort_limit_sim'], name)
            self.limits.append(limit)
            m.actuator_ctrlrange[aid] = [-limit, limit]
            m.jnt_actfrcrange[jid] = [-limit, limit]
            # Native affine PD lets MuJoCo integrate damping implicitly, like
            # the training ImplicitActuator; explicit PD destabilizes light wrists.
            m.actuator_gaintype[aid] = mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_AFFINE
            m.actuator_gainprm[aid, 0] = self.kp[-1]
            m.actuator_biasprm[aid, :3] = [0, -self.kp[-1], -self.kd[-1]]
            m.actuator_ctrllimited[aid] = False
            m.actuator_forcelimited[aid] = True
            m.actuator_forcerange[aid] = [-limit, limit]
            m.dof_armature[vid] = resolve(a['armature'], name)
            m.dof_damping[vid] = 0.; m.dof_frictionloss[vid] = 0.
        self.kp, self.kd, self.limits = map(np.asarray, (self.kp, self.kd, self.limits))
        mujoco.mj_setConst(m, d)
        d.qpos[:3] = spawn
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[self.qadr] = self.default
        mujoco.mj_forward(m, d)
        self.base = m.body('pelvis').id
        self.torso = m.body('torso_link').id
        camera = cfg['scene']['depth_camera']
        self.camera = camera
        if camera['offset']['convention'] != 'world' or camera['pattern_cfg']['width'] != 64 or camera['pattern_cfg']['height'] != 36:
            raise ValueError('Unsupported camera convention or resolution')
        self.camera_period = max(1, round(camera['update_period'] / self.dt))
        offset = camera['offset']; q = offset['rot']
        self.camera_rot = rotation([q[3], *q[:3]])  # Isaac Lab 3 uses xyzw.
        pat = camera['pattern_cfg']
        u, v = np.meshgrid(np.arange(64) + .5, np.arange(36) + .5)
        fx = 64 * pat['focal_length'] / pat['horizontal_aperture']
        fy = 36 * pat['focal_length'] / pat['vertical_aperture']
        rays = np.stack((np.ones_like(u), -(u - 32) / fx, -(v - 18) / fy), -1).reshape(-1, 3)
        self.rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        # Head housing is visible in group 3, but absent from the depth ray mask.
        configure_geometry_groups(m)
        self.history = Histories(delay)
        self.action = np.zeros(29)
        self.frame = None
        self.raw_depth = None
        self.policy_depth = None
        actor_cfg = dict(agent['actor']); kind = actor_cfg.pop('class_name')
        if kind != 'CNNModel' or agent['obs_groups']['actor'] != ['policy', 'depth']:
            raise ValueError('Unsupported network architecture')
        sample = TensorDict({'policy': torch.zeros(1, 495), 'depth': torch.zeros(1, 8, 18, 32)}, batch_size=[1])
        self.actor = CNNModel(sample, agent['obs_groups'], 'actor', 29, **actor_cfg)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.actor.load_state_dict(saved['actor_state_dict'], strict=True)
        self.actor.eval()

    def depth_frame(self):
        m, d = self.model, self.data
        torso_rot = d.xmat[self.torso].reshape(3, 3)
        origin = d.xpos[self.torso] + torso_rot @ self.camera['offset']['pos']
        rays = np.ascontiguousarray(self.rays @ (torso_rot @ self.camera_rot).T)
        distances = np.empty(len(rays)); ids = np.empty(len(rays), dtype=np.int32)
        # Camera range is optical depth, not Euclidean ray length. Isaac's
        # camera also raycasts beyond max_distance before depth preprocessing.
        mujoco.mj_multiRay(m, d, origin, rays.ravel(), np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8), 1, -1, ids, distances, None, len(rays), 1.e6)
        optical = distances * self.rays[:, 0]
        optical[distances < 0] = np.inf
        self.raw_depth = optical.reshape(36, 64).copy()
        return preprocess_depth(self.raw_depth)

    @torch.inference_mode()
    def step(self, command, tick):
        m, d = self.model, self.data
        if tick % self.camera_period == 0:
            self.frame = self.depth_frame()
        rot = d.xmat[self.base].reshape(3, 3)
        # Free-joint angular velocity is in the root link frame. BODY velocity
        # from mj_objectVelocity uses its principal-inertia frame instead.
        obs = self.history.update([d.qvel[3:6], orientation_features(rot), command, d.qpos[self.qadr], d.qvel[self.vadr], self.action], self.frame)
        self.policy_depth = obs['depth'][0, -1].numpy().copy()
        self.action = self.actor(obs).numpy()[0]
        if not np.isfinite(self.action).all():
            raise RuntimeError('Policy produced non-finite actions')
        target = self.default + self.scale * self.action
        d.ctrl[self.actids] = target
        for _ in range(self.decimation):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        if not np.isfinite(d.qpos).all():
            raise RuntimeError('Simulation produced non-finite state')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--scene', type=Path, default=TRAINING_SCENE)
    p.add_argument('--command', nargs=3, type=float, default=[.4, 0., 0.], metavar=('VX', 'VY', 'WZ'))
    p.add_argument('--control', choices=['target', 'velocity'], default='target')
    p.add_argument('--target', nargs=2, type=float, metavar=('X', 'Y'), help='Initial world target; GUI waits for a click if omitted')
    p.add_argument('--speed', type=float, default=.65, help='Target controller speed cap (m/s); training height-stage range is .55-.75')
    p.add_argument('--spawn', nargs=3, type=float, default=[0., 0., .8])
    p.add_argument('--yaw', type=float, default=0.)
    p.add_argument('--depth-delay', type=int, choices=[0, 1], default=0)
    p.add_argument('--duration', type=float, help='Seconds; GUI runs until closed, headless defaults to 60')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--real-time', action='store_true')
    p.add_argument('--report', type=Path)
    p.add_argument('--camera-distance', type=float, default=None)
    args = p.parse_args()
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        p.error('--duration must be positive')
    if not np.isfinite([*args.command, *args.spawn, *(args.target or []), args.speed, args.yaw]).all() or args.speed < 0:
        p.error('Commands and initial pose must be finite')
    torch.set_num_threads(1)
    runner = Runner(args.checkpoint.resolve(), args.scene.resolve(), args.spawn, args.yaw, args.depth_delay)
    scene_info = args.scene.with_suffix('.json')
    courses = json.loads(scene_info.read_text()).get('lanes', []) if scene_info.exists() else []
    current_course = None
    viewer = None
    keys = deque()
    command = np.asarray(args.command, dtype=float)
    default_target = [6.7, 0.] if args.scene.resolve() == TRAINING_SCENE else [20., 0.]
    target = np.asarray(args.target if args.target is not None else (default_target if args.headless else args.spawn[:2]), dtype=float)
    if not args.headless:
        from interactive_viewer import InteractiveViewer
        viewer = InteractiveViewer(runner.model, runner.data, keys.append)
        import glfw
        glfw.set_window_title(viewer.window, f'G1 sim2sim | {args.scene.stem} | {args.checkpoint.stem}')
        if args.scene.resolve() == TRAINING_SCENE:
            viewer.cam.distance = 10.
        if args.camera_distance is not None:
            viewer.cam.distance = args.camera_distance
        if courses:
            viewer.course_help = '1-5: stairs 8/12/16/20/23 cm | 6/7: slopes\n8: rough | 9: hurdles | 0: blocks | R: reset'
        runner.frame = runner.depth_frame()
        runner.policy_depth = runner.frame.numpy().copy()
        if args.target is None and args.control == 'target':
            viewer.message = 'Click a terrain target to start (simulation paused)'
            command[:] = 0
        viewer.draw(runner, target, command, args.control)
        print('Left click terrain: set target. Left drag: orbit. Right drag: pan. Wheel: zoom. F: follow camera. Space: stop. Esc: exit.', flush=True)
    start = runner.data.qpos[:3].copy(); min_height = start[2]; fell = False
    wall = time.monotonic()
    duration = args.duration if args.duration is not None else (60. if args.headless else None)
    started = args.headless or args.target is not None or args.control == 'velocity'
    tick = 0
    try:
        while duration is None or tick < max(1, int(duration / runner.dt)):
            if viewer is not None:
                viewer.poll()
                if not viewer.is_running():
                    break
                while viewer.targets:
                    point = viewer.targets.popleft()
                    target[:] = point[:2]
                    args.control = 'target'
                    started = True
                    print('Clicked target:', point.round(3).tolist(), flush=True)
            while keys:
                key = keys.popleft()
                course_index = (key - 49 if key != 48 else 9) if 48 <= key <= 57 else None
                if viewer is not None and (key == 82 or (courses and course_index is not None and course_index < len(courses))):
                    if course_index is not None:
                        current_course = courses[course_index]
                    spawn = current_course['spawn'] if current_course else args.spawn
                    mujoco.mj_resetData(runner.model, runner.data)
                    runner.data.qpos[:3] = spawn
                    runner.data.qpos[3:7] = [np.cos(args.yaw/2), 0, 0, np.sin(args.yaw/2)]
                    runner.data.qpos[runner.qadr] = runner.default
                    mujoco.mj_forward(runner.model, runner.data)
                    runner.history = Histories(args.depth_delay)
                    runner.action[:] = 0
                    runner.frame = runner.depth_frame()
                    runner.policy_depth = runner.frame.numpy().copy()
                    target[:] = spawn[:2]
                    command[:] = 0
                    tick = 0
                    fell = False
                    started = False
                    start = np.array(spawn, dtype=float)
                    min_height = start[2]
                    args.control = 'target'
                    viewer.target_point = None
                    viewer.cam.lookat[:] = spawn
                    viewer.message = (current_course['name'] if current_course else 'Reset') + ' | Click terrain to start'
                    continue
                if args.control == 'target':
                    for code, axis, increment in [(87, 0, 1.), (83, 0, -1.), (65, 1, 1.), (68, 1, -1.)]:
                        if key == code:
                            target[axis] += increment
                            started = True
                    if key == 32:
                        target[:] = runner.data.qpos[:2]
                    print('Target:', target.round(2).tolist(), flush=True)
                    continue
                for code, axis, increment in [(87, 0, .1), (83, 0, -.1), (65, 1, .1), (68, 1, -.1), (81, 2, .1), (69, 2, -.1)]:
                    if key == code:
                        command[axis] += increment
                if key == 32:
                    command[:] = 0
                command[:] = np.clip(command, [-1., -.5, -1.], [1.5, .5, 1.])
                print('Command:', command.round(2).tolist(), flush=True)
            if not started:
                viewer.draw(runner, target, command, args.control)
                time.sleep(.02)
                continue
            if tick == 0:
                # Wall-clock waiting for the initial click is not simulated time.
                wall = time.monotonic()
                if viewer is not None:
                    viewer.message = 'Target selected'
            if args.control == 'target':
                command = target_command(runner.data.qpos[:3], runner.data.xmat[runner.base].reshape(3, 3), target, args.speed, runner.command_config)
            runner.step(command, tick)
            min_height = min(min_height, float(runner.data.qpos[2]))
            if runner.data.qpos[2] < .35 or runner.data.xmat[runner.base].reshape(3, 3)[2, 2] < .3:
                fell = True
                if viewer is not None and courses:
                    started = False
                    viewer.message = 'Fallen | R: reset | 1-9 / 0: choose terrain'
                    continue
                break
            if viewer is not None:
                viewer.draw(runner, target, command, args.control)
            if args.real_time or viewer is not None:
                time.sleep(max(0., wall + (tick + 1) * runner.dt - time.monotonic()))
            tick += 1
    finally:
        if viewer is not None:
            viewer.close()
    report = {'checkpoint': str(args.checkpoint.resolve()), 'scene': str(args.scene.resolve()), 'control': args.control,
              'target': target.tolist() if args.control == 'target' else None, 'speed_cap': args.speed,
              'final_command': command.tolist(),
              'sim_seconds': float(runner.data.time), 'wall_seconds': time.monotonic() - wall,
              'fell': fell, 'min_base_height': min_height, 'final_position': runner.data.qpos[:3].tolist(),
              'displacement': (runner.data.qpos[:3] - start).tolist(), 'policy_shape': [1, 495], 'depth_shape': [1, 8, 18, 32]}
    if args.control == 'target':
        report['target_distance'] = float(np.linalg.norm(target - runner.data.qpos[:2]))
        report['target_reached'] = report['target_distance'] <= runner.command_config['target_dis_threshold']
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    return 2 if fell else 0


if __name__ == '__main__':
    raise SystemExit(main())
