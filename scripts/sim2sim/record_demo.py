"""Record genuine policy rollouts with a following camera and control telemetry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from run_mujoco import Runner, TRAINING_SCENE, target_command


def flat_scene(destination):
    root = ET.parse(TRAINING_SCENE).getroot()
    for include in root.findall('include'):
        include.set('file', str((TRAINING_SCENE.parent / include.get('file')).resolve()))
    world = root.find('worldbody')
    for geom in list(world.findall('geom')):
        world.remove(geom)
    ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 .05', material='groundplane')
    ET.ElementTree(root).write(destination, encoding='unicode')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['flat', 'turn', 'stairs-up', 'stairs-down'], required=True)
    parser.add_argument('--scene', type=Path, default=TRAINING_SCENE)
    parser.add_argument('--step-height', type=float, default=.12)
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--speed', type=float, default=.65)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--probe', action='store_true', help='Simulate and save telemetry without rendering.')
    args = parser.parse_args()
    if not np.isfinite([args.duration, args.speed]).all() or args.duration <= 0 or args.speed < 0:
        parser.error('Duration must be positive and speed non-negative; both must be finite.')
    if min(args.width, args.height) < 240 or args.width % 2 or args.height % 2:
        parser.error('Video dimensions must be even and at least 240 pixels.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    scene = args.scene.resolve()
    spawn, yaw, target = [0., 0., .8], 0., np.array([6.7, 0.])
    if args.mode in ('flat', 'turn'):
        scene = flat_scene(args.output.with_suffix('.scene.xml'))
    elif args.mode == 'stairs-down':
        spawn, yaw, target = [6.6, 0., 2.84], np.pi, np.array([0., 0.])
    runner = Runner(args.checkpoint.resolve(), scene, spawn, yaw, 0)
    m, d = runner.model, runner.data
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 3.8, 125., -18.
    if args.mode == 'stairs-down':
        camera.azimuth = 55.
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 0, 1, 0, 0]
    renderer = writer = None
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 19)
    title_font = ImageFont.truetype(font_path, 24)
    # Render every second 50 Hz policy step: playback time equals simulated time.
    stride, fps = 2, 1. / (runner.dt * 2)
    if not args.probe:
        m.vis.global_.offwidth, m.vis.global_.offheight = args.width, args.height
        renderer = mujoco.Renderer(m, height=args.height, width=args.width)
        writer = imageio.get_writer(str(args.output), fps=fps, codec='libx264', quality=8,
                                   macro_block_size=1, ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    rows, frames, fell = [], 0, False
    titles = {'flat': 'Forward walking', 'turn': 'Walking with left / right steering',
              'stairs-up': f'Ascending 17 steps / {args.step_height*100:g} cm', 'stairs-down': f'Descending 17 steps / {args.step_height*100:g} cm'}
    try:
        for tick in range(round(args.duration / runner.dt)):
            seconds = tick * runner.dt
            command = np.array([args.speed, 0., 0.])
            if args.mode == 'turn':
                command[2] = .3 if 4 <= seconds < 10 else (-.3 if 12 <= seconds < 18 else 0.)
            elif args.mode.startswith('stairs'):
                command = target_command(d.qpos[:3], d.xmat[runner.base].reshape(3, 3),
                                         target, args.speed, runner.command_config)
            runner.step(command, tick)
            tilt = float(d.xmat[runner.base].reshape(3, 3)[2, 2])
            fell = bool(d.qpos[2] < .35 or tilt < .3)
            if tick % stride == stride - 1:
                rows.append({'time': float(d.time), 'position': d.qpos[:3].tolist(),
                             'command': command.tolist(), 'tilt_cosine': tilt})
                if renderer is not None:
                    camera.lookat[:] = d.qpos[:3] + np.array([0., 0., .05])
                    renderer.update_scene(d, camera=camera, scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, args.width, 76), fill=(20, 24, 29))
                    draw.text((24, 10), 'G1  |  ' + titles[args.mode], font=title_font, fill=(245, 247, 250))
                    draw.text((24, 44), f'MuJoCo simulation  |  {args.checkpoint.stem}  |  1x playback', font=font, fill=(192, 201, 211))
                    draw.rectangle((0, args.height - 42, args.width, args.height), fill=(20, 24, 29))
                    draw.text((24, args.height - 31),
                              f't = {d.time:05.2f} s     command: vx {command[0]:.2f} m/s   yaw {command[2]:+.2f} rad/s'
                              + ('     FALL DETECTED' if fell else ''), font=font, fill=(230, 235, 240))
                    writer.append_data(np.asarray(frame))
                    if frames == 0:
                        frame.save(args.output.with_suffix('.preview.jpg'))
                    frames += 1
            if fell:
                break
            if (tick + 1) % 250 == 0:
                print(f'{args.mode}: {d.time:.1f}s, position {d.qpos[:3].round(3)}', flush=True)
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    report = {'checkpoint': str(args.checkpoint.resolve()), 'mode': args.mode, 'scene': str(scene), 'step_height_m':args.step_height if args.mode.startswith('stairs') else None,
              'duration_seconds': float(d.time), 'fell': fell, 'fps': fps, 'frames': frames,
              'final_position': d.qpos[:3].tolist(), 'trajectory': rows}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'trajectory'}), flush=True)


if __name__ == '__main__':
    main()
