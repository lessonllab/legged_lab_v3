"""Controlled wall/terrain ablation. Never modifies the interactive scene.

The optional hidden/visual-only walls separate camera effects from collisions.
These sensor/physics ablations are diagnostics, not deployment settings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from run_mujoco import ROOT, TRAINING_SCENE, Runner, target_command

CHECKPOINT = ROOT / 'logs/rsl_rl/g1_amp_stairs_long/2026-09-16_20-26-52_reverse_height_v11/model_50100.pt'
SHORT_SCENE = ROOT / 'hiking-in-the-wild-sim2sim/unitree_mujoco/unitree_robots/g1/scene_29dof_terrain_with_camera.xml'


def make_scene(family, condition, output):
    source = SHORT_SCENE if family == 'short' else TRAINING_SCENE
    root = ET.parse(source).getroot()
    include = root.find('include')
    include.set('file', str((source.parent / include.get('file')).resolve()))
    world = root.find('worldbody')
    walls = []
    for geom in list(world.findall('geom')):
        size = np.fromstring(geom.get('size', ''), sep=' ')
        if family == 'short' and geom.get('type') == 'box' and np.isclose(size[1], .025):
            world.remove(geom)
            walls.append(geom)
    if family == 'long':
        for sign in (-1, 1):
            walls.append(ET.Element('geom', type='box', pos=f'4.5 {sign * .775} 1.5', size='3.5 .025 1.5'))
    assert len(walls) == 2
    if condition.startswith('flat'):
        for geom in list(world.findall('geom')):
            world.remove(geom)
        ET.SubElement(world, 'geom', name='diagnostic_floor', type='plane', size='0 0 .05')
    if condition not in ('flat_open', 'stairs_open'):
        for i, geom in enumerate(walls):
            geom.set('name', f'diagnostic_wall_{i}')
            world.append(geom)
    ET.indent(root)
    path = output / f'{family}_{condition}.xml'
    ET.ElementTree(root).write(path, encoding='unicode')
    y = 2. if family == 'short' else 0.
    return path, [0., y, .8], np.array([3. if family == 'short' else 6.7, y])


def run_case(args, family, condition):
    scene, spawn, goal = make_scene(family, condition, args.output)
    r = Runner(args.checkpoint, scene, spawn, args.yaw, 0)
    wall_ids = {r.model.geom(f'diagnostic_wall_{i}').id for i in range(2)} if condition not in ('flat_open', 'stairs_open') else set()
    if condition.endswith(('hidden_walls', 'disabled_walls')):
        for gid in wall_ids:
            r.model.geom_group[gid] = 4  # keep collision, omit from depth
    if condition.endswith(('visual_walls', 'disabled_walls')):
        for gid in wall_ids:
            r.model.geom_contype[gid] = r.model.geom_conaffinity[gid] = 0
    r.depth_frame()
    np.savez_compressed(args.output / f'{family}_{condition}_initial_depth.npz', raw=r.raw_depth)
    max_wall_force = max_riser_force = 0.
    wall_contact_steps = 0
    real_step = mujoco.mj_step

    def audited_step(model, data):
        nonlocal max_wall_force, max_riser_force, wall_contact_steps
        real_step(model, data)
        touched = False
        for ci, contact in enumerate(data.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            world = a if model.geom_bodyid[a] == 0 else b if model.geom_bodyid[b] == 0 else -1
            if world < 0:
                continue
            is_wall = world in wall_ids
            is_riser = model.geom_type[world] == mujoco.mjtGeom.mjGEOM_BOX and abs(contact.frame[2]) < .5
            if not is_wall and not is_riser:
                continue
            force = np.empty(6)
            mujoco.mj_contactForce(model, data, ci, force)
            if is_wall:
                max_wall_force = max(max_wall_force, float(force[0]))
                touched |= force[0] > 1.
            elif is_riser:
                max_riser_force = max(max_riser_force, float(force[0]))
        wall_contact_steps += int(touched)

    trace = []
    fell = False
    start = time.monotonic()
    mujoco.mj_step = audited_step
    try:
        for tick in range(round(args.duration / r.dt)):
            command = target_command(r.data.qpos[:3], r.data.xmat[r.base].reshape(3, 3), goal, args.speed, r.command_config)
            r.step(command, tick)
            trace.append([r.data.time, *r.data.qpos[:3], *r.data.qvel[:3], *command,
                          r.data.xmat[r.base].reshape(3, 3)[2, 2]])
            if r.data.qpos[2] < .35 or trace[-1][-1] < .3:
                fell = True
                break
    finally:
        mujoco.mj_step = real_step
    trace = np.asarray(trace)
    np.save(args.output / f'{family}_{condition}_trace.npy', trace)
    tail = trace[-min(len(trace), round(5. / r.dt)):]
    distance = float(np.linalg.norm(r.data.qpos[:2] - goal))
    reached = distance <= r.command_config['target_dis_threshold']
    avg_v = float(np.linalg.norm(tail[:, 4:6], axis=1).mean())
    avg_cmd = float(tail[:, 7].mean())
    result = dict(family=family, condition=condition, yaw=args.yaw, speed=args.speed,
                  duration=float(r.data.time), target=goal.tolist(), final_position=r.data.qpos[:3].tolist(),
                  max_x=float(trace[:, 1].max()), max_base_z=float(trace[:, 3].max()),
                  target_distance=distance, target_reached=reached, fell=fell,
                  stalled=bool(not fell and not reached and avg_v < .05 and avg_cmd > .1),
                  final5s_mean_planar_speed=avg_v, final5s_mean_forward_command=avg_cmd,
                  peak_wall_normal_force_N=max_wall_force, wall_contact_physics_steps=wall_contact_steps,
                  peak_riser_normal_force_N=max_riser_force, wall_seconds=time.monotonic() - start)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    p.add_argument('--family', nargs='+', choices=['short', 'long'], default=['short', 'long'])
    p.add_argument('--conditions', nargs='+', choices=['flat_open', 'flat_walls', 'stairs_open', 'stairs_walls',
                   'stairs_hidden_walls', 'stairs_visual_walls', 'stairs_disabled_walls', 'flat_hidden_walls', 'flat_visual_walls'],
                   default=['flat_open', 'flat_walls', 'stairs_open', 'stairs_walls'])
    p.add_argument('--speed', type=float, default=.65)
    p.add_argument('--yaw', type=float, default=0.)
    p.add_argument('--duration', type=float, default=30.)
    p.add_argument('--output', type=Path, default=ROOT / '.runtime/sim2sim/wall_ablation')
    args = p.parse_args()
    if args.duration <= 0 or args.speed <= 0:
        p.error('duration and speed must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    results = []
    metadata = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    duration=args.duration, speed=args.speed, yaw=args.yaw, depth_delay=0,
                    wall_geometry='3 m tall, 1.5 m clear width; short x=[1,5], long x=[1,8]',
                    trace_columns=['time','x','y','z','vx','vy','vz','cmd_vx','cmd_vy','cmd_wz','upright_zz'],
                    note='Deterministic paired cases, immediate target command from the same initial state. Physics contacts sampled every .005 s.')
    for family in args.family:
        for condition in args.conditions:
            results.append(run_case(args, family, condition))
            (args.output / 'results.json').write_text(json.dumps({'metadata': metadata, 'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
