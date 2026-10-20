"""Render paired, recorded Isaac Sim poses without advancing a simulation.

This reads compare_square_policy_effects.py's first-episode traces. MuJoCo is
used only for forward kinematics and rasterization: there is no policy loading,
physics integration, or Isaac Sim initialization in this process.

Example::

    python scripts/tools/render_square_effects_replay.py \
        --input-dir .runtime/policy_effects_11700_vs18200
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

# Must precede the MuJoCo import, including imports made by another helper.
os.environ['MUJOCO_GL'] = 'egl'

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
# The same G1 scene used by scripts/sim2sim/run_mujoco.py and record_demo.py.
TRAINING_SCENE = ROOT / 'scripts/sim2sim/scenes/training_stairs_12cm.xml'
PROFILE = ROOT / 'scripts/sim2sim/g1_training_profile.json'
DEFAULT_INPUT = ROOT / '.runtime/policy_effects_11700_vs18200'
DEFAULT_REFERENCE = ROOT / (
    'logs/rsl_rl/g1_beamdojo_fullbody_sim6/'
    '2026-10-11_11-43-11-667266_square_flat_rehearsal_3000_4096/model_11700.pt'
)
LABELS = ('11700', '18200')
PANEL_WIDTH, PANEL_HEIGHT, FPS = 640, 480, 25
REPLAY_LABEL = 'Recorded Isaac Sim state replay; rendering only'
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def read_json(path):
    return json.loads(Path(path).read_text())


def numbers(values):
    return ' '.join(f'{float(v):.12g}' for v in values)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archived_geometry(params):
    """Load only the four CPU geometry modules, always from the archive."""
    loaded = {}
    names = ['beamdojo_feasibility_geometry', 'beamdojo_square_course_geometry',
             'beamdojo_square_diagram_playback', 'beamdojo_square_mixed_geometry']
    if read_json(params.parent/'contract.json')['square_mixed']['version'] == 2:
        names.append('beamdojo_square_extended_geometry')
    for name in names:
        path = (params / f'{name}.py').resolve(strict=True)
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load archived geometry: {path}')
        module = importlib.util.module_from_spec(spec)
        # These archived modules import each other by their original names.
        # Explicitly replacing them prevents an accidental live-source import.
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def stone_terrain(reference, episode, modules):
    """Exact archived 2 cm collision boxes, translated into tile-local space."""
    contract = read_json(reference.parent / 'contract.json')
    meta = contract['square_mixed']
    column = int(episode['column'])
    seed, family, variant = meta['specs'][column]
    if int(family) != int(episode['family']) or int(variant) != int(episode['variant']):
        raise ValueError(f'Archived tile column {column} does not match case metadata')
    family_name = meta['families'][int(family)]
    extended = meta['version'] == 2
    mixed = modules['beamdojo_square_extended_geometry' if extended else 'beamdojo_square_mixed_geometry']
    course = mixed.mixed_layout(int(seed), int(episode['level']) / 10.,
                               family_name, int(variant))
    raster = (mixed if extended else modules['beamdojo_square_course_geometry']).rasterize(course)
    grid = raster['collision_height_m']
    meshes, origin = mixed.grid_mesh(grid)
    boxes = [(mesh.bounds - origin) for mesh in meshes]
    provenance = dict(column=column, seed=int(seed), family=family_name,
                      level=int(episode['level']), variant=int(variant),
                      resolution_m=float(raster['resolution_m']),
                      raster_sha256=hashlib.sha256(grid.tobytes()).hexdigest(),
                      origin_removed_m=np.asarray(origin).tolist(),
                      top_height_m=0., hole_height_m=-1.,
                      terrain_box_count=len(boxes))
    return boxes, provenance


def rendering_model(boxes, destination, profile):
    """Reuse the training scene and its USD-calibrated G1 visual geometry.

    This performs the XML geometry replacement used by calibrated_model in
    run_mujoco.py, without importing its policy/Runner dependencies. Joint-body
    transforms are applied before model compilation so recorded angles have
    the same link transforms as the training USD asset.
    """
    root = ET.parse(TRAINING_SCENE).getroot()
    world = root.find('worldbody')
    if world is None:
        raise ValueError('Training scene has no worldbody')
    for geom in list(world.findall('geom')):
        world.remove(geom)
    for light in list(world.findall('light')):
        world.remove(light)
    ET.SubElement(world, 'light', pos='0 0 5', dir='0 0 -1', directional='true',
                  diffuse='.45 .45 .45', ambient='.1 .1 .1',
                  specular='0 0 0', castshadow='false')
    if boxes is None:
        ET.SubElement(world, 'geom', name='replay_flat_floor', type='plane',
                      size='0 0 .05', material='groundplane', group='0')
    else:
        for index, bounds in enumerate(boxes):
            low, high = bounds
            center = (low + high) / 2
            platform = bool((np.abs(center[:2]) >= 3.).any() or
                            np.max(np.abs(np.r_[low[:2], high[:2]])) <= .71)
            color = '.16 .19 .22 1' if index == 0 else (
                '.54 .61 .67 1' if platform else '.73 .65 .49 1')
            ET.SubElement(world, 'geom', name=f'replay_terrain_{index}', type='box',
                          pos=numbers(center), size=numbers((high-low)/2),
                          rgba=color, group='0')
    # Resolve original robot meshes before flattening the scene include.
    for include in list(root.findall('include')):
        source = (TRAINING_SCENE.parent / include.attrib['file']).resolve(strict=True)
        robot = ET.parse(source).getroot()
        compiler = robot.find('compiler')
        meshdir = (source.parent / (compiler.get('meshdir', '') if compiler is not None else '')).resolve()
        for mesh in robot.findall('asset/mesh'):
            mesh.set('file', str((meshdir / mesh.attrib['file']).resolve(strict=True)))
        root.remove(include)
        for child in list(robot):
            root.append(child)
    bodies = {body.attrib['name']: body for body in root.iter('body')}
    for name in profile['bodies']:
        for geom in list(bodies[name].findall('geom')):
            bodies[name].remove(geom)
    assets = ET.SubElement(root, 'asset')
    for visual in profile['visuals']:
        mesh_name = visual['mesh']
        mesh_file = ROOT / 'scripts/sim2sim/assets' / f'{mesh_name}.stl'
        ET.SubElement(assets, 'mesh', name=mesh_name, file=str(mesh_file.resolve(strict=True)))
        ET.SubElement(bodies[visual['body']], 'geom', name=mesh_name, type='mesh',
                      mesh=mesh_name, contype='0', conaffinity='0', density='0',
                      group=str(visual['group']), rgba='.68 .71 .75 1')
    # Collision shapes are retained in group 2, hidden in the replay renderer.
    for index, shape in enumerate(profile['collisions']):
        attrs = dict(name=f'training_collision_{index}', type=shape['type'], group='2')
        for key in ('pos', 'quat', 'size'):
            if key in shape:
                attrs[key] = numbers(shape[key])
        if 'mesh' in shape:
            name = shape['mesh']
            ET.SubElement(assets, 'mesh', name=name,
                          file=str((ROOT / 'scripts/sim2sim/assets' / f'{name}.obj').resolve(strict=True)))
            attrs['mesh'] = name
        ET.SubElement(bodies[shape['body']], 'geom', attrs)
    for body in bodies.values():
        for joint in body.findall('joint'):
            transform = profile['joints'].get(joint.get('name'))
            if transform is not None:
                body.set('pos', numbers(transform['pos']))
                body.set('quat', numbers(transform['quat']))
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    global_settings = visual.find('global')
    if global_settings is None:
        global_settings = ET.SubElement(visual, 'global')
    global_settings.set('offwidth', str(PANEL_WIDTH))
    global_settings.set('offheight', str(PANEL_HEIGHT))
    xml = ET.tostring(root, encoding='unicode')
    destination.write_text(xml)
    return mujoco.MjModel.from_xml_string(xml)


def episode_id(episode):
    if 'case_id' in episode:
        return int(episode['case_id'])
    return int(episode['caseid'])


def select_cases(episodes):
    selected = []
    for mode, slug, title in (
            ('forward_1.0', 'flat_forward_1p0', 'Flat: forward 1.0 m/s'),
            ('backward', 'flat_backward', 'Flat: backward'),
            ('turn_left', 'flat_turn_left', 'Flat: turn left')):
        matches = [row for row in episodes if row['kind'] == 'flat' and row['mode'] == mode]
        if not matches:
            raise ValueError(f'Missing flat case: {mode}')
        row = min(matches, key=lambda item: (int(item.get('replica', 0)), episode_id(item)))
        selected.append((slug, title, row))
    matches = [row for row in episodes if row['kind'] == 'stone' and
               all(int(row[key]) == value for key, value in
                   (('family', 0), ('route', 0), ('level', 10), ('variant', 0)))]
    if not matches:
        raise ValueError('Missing stone family 0 / route 0 / level 10 / variant 0 case')
    row = min(matches, key=lambda item: (int(item.get('replica', 0)), episode_id(item)))
    selected.append(('stone_f0_r0_l10_v0', 'Stones: cardinal / level 10 / variant 0', row))
    return selected


class RecordedCase:
    def __init__(self, trace, episode, label):
        self.trace, self.episode, self.label = trace, episode, label
        self.case_id = episode_id(episode)
        active = np.asarray(trace['active'][:, self.case_id], dtype=bool)
        self.indices = np.flatnonzero(active)
        if not len(self.indices):
            raise ValueError(f'{label} case {self.case_id} has no recorded active states')
        if np.any(np.diff(self.indices) != 1):
            raise ValueError(f'{label} case {self.case_id} active states are not contiguous')
        self.times = np.asarray(trace['time'][self.indices, self.case_id], dtype=float)
        if not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError(f'{label} case {self.case_id} timestamps are not increasing')
        self.first_time, self.last_time = float(self.times[0]), float(self.times[-1])
        for name in ('root_local', 'root_quat', 'joint_pos', 'heading_error_deg',
                     'body_velocity', 'command'):
            if not np.isfinite(trace[name][self.indices, self.case_id]).all():
                raise ValueError(f'{label} case {self.case_id} has invalid {name}')

    def frame(self, replay_time):
        # Zero-order hold uses an actual saved state; no pose interpolation.
        position = int(np.searchsorted(self.times, replay_time + 1e-7, side='right') - 1)
        position = max(0, min(position, len(self.indices)-1))
        return int(self.indices[position]), replay_time > self.last_time + 1e-7


class PoseRenderer:
    def __init__(self, model, names):
        self.model = model
        self.data = mujoco.MjData(model)
        joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                              for name in names], dtype=int)
        if (joint_ids < 0).any():
            missing = [name for name, jid in zip(names, joint_ids) if jid < 0]
            raise ValueError('Recorded joints missing from G1 model: ' + ', '.join(missing))
        if len(set(joint_ids.tolist())) != len(joint_ids):
            raise ValueError('Recorded joint_names contains duplicates')
        if not np.isin(model.jnt_type[joint_ids], [mujoco.mjtJoint.mjJNT_HINGE,
                                                  mujoco.mjtJoint.mjJNT_SLIDE]).all():
            raise ValueError('Expected scalar recorded joints')
        self.qadr = model.jnt_qposadr[joint_ids]
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError('Expected exactly one floating G1 root')
        self.root_adr = int(model.jnt_qposadr[free[0]])
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 4.
        self.camera.azimuth = 125.
        self.camera.elevation = -24.
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = [1, 1, 0, 1, 0, 0]
        self.renderer = mujoco.Renderer(model, height=PANEL_HEIGHT, width=PANEL_WIDTH)

    def render(self, case, frame_index):
        trace, cid = case.trace, case.case_id
        self.data.qpos[:] = self.model.qpos0
        root = np.asarray(trace['root_local'][frame_index, cid], dtype=float)
        xyzw = np.asarray(trace['root_quat'][frame_index, cid], dtype=float)
        norm = np.linalg.norm(xyzw)
        if norm < .5 or norm > 1.5:
            raise ValueError(f'Invalid root quaternion in {case.label} case {cid}')
        self.data.qpos[self.root_adr:self.root_adr+3] = root
        self.data.qpos[self.root_adr+3:self.root_adr+7] = xyzw[[3, 0, 1, 2]] / norm
        self.data.qpos[self.qadr] = trace['joint_pos'][frame_index, cid]
        self.data.qvel[:] = 0.
        self.data.time = float(trace['time'][frame_index, cid])
        # The sole state update: forward kinematics, never physics integration.
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = [root[0], root[1], .72]
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        return Image.fromarray(self.renderer.render())

    def close(self):
        self.renderer.close()


def caption(frame, case, index, replay_time, terminated, title, fonts):
    draw = ImageDraw.Draw(frame)
    color = (100, 190, 255) if case.label == LABELS[0] else (255, 196, 94)
    draw.rectangle((0, 0, PANEL_WIDTH, 65), fill=(18, 24, 31))
    draw.rectangle((0, 0, 6, 65), fill=color)
    draw.text((18, 8), f'{case.label}  |  {title}', font=fonts['title'], fill=(242, 246, 250))
    draw.text((18, 39), REPLAY_LABEL, font=fonts['small'], fill=(187, 201, 215))
    draw.rectangle((0, PANEL_HEIGHT-61, PANEL_WIDTH, PANEL_HEIGHT), fill=(18, 24, 31))
    trace, cid = case.trace, case.case_id
    command = trace['command'][index, cid]
    velocity = trace['body_velocity'][index, cid]
    sample_time = float(trace['time'][index, cid])
    telemetry = (f't = {replay_time:04.2f} s   cmd vx {command[0]:+.2f} m/s'
                 f'   yaw {command[2]:+.2f} rad/s')
    draw.text((18, PANEL_HEIGHT-52), telemetry, font=fonts['body'], fill=(231, 237, 243))
    state_text = (f'vx {velocity[0]:+.2f} m/s   heading error '
                  f"{float(trace['heading_error_deg'][index, cid]):+.1f} deg")
    draw.text((18, PANEL_HEIGHT-28), state_text, font=fonts['body'], fill=(196, 209, 221))
    if terminated:
        text = f'TERMINATED at {sample_time:.2f} s | last active frame frozen'
        draw.rectangle((14, 76, PANEL_WIDTH-14, 109), fill=(91, 38, 37))
        draw.text((24, 82), text, font=fonts['body'], fill=(255, 220, 214))
    return frame


def render_video(model, traces, episodes, selection, output_dir, duration, fonts):
    slug, title, reference_episode = selection
    cid = episode_id(reference_episode)
    cases = {label: RecordedCase(traces[label], episodes[label][cid], label) for label in LABELS}
    if not math.isclose(cases[LABELS[0]].first_time, cases[LABELS[1]].first_time, abs_tol=1e-6):
        raise ValueError(f'Paired initial trace timestamp mismatch for case {cid}')
    frame_count = max(1, int(round(duration * FPS)))
    start = cases[LABELS[0]].first_time
    times = start + np.arange(frame_count) / FPS
    preview_index = int(np.argmin(np.abs(times-2.5)))
    video_path = output_dir / f'{slug}.mp4'
    preview_path = output_dir / f'{slug}.preview.jpg'
    renderers = {}
    writer = None
    preview = None
    try:
        for label in LABELS:
            names = [str(name) if not isinstance(name, bytes) else name.decode()
                     for name in traces[label]['joint_names']]
            renderers[label] = PoseRenderer(model, names)
        writer = imageio.get_writer(str(video_path), fps=FPS, codec='libx264', quality=8,
                                   macro_block_size=1,
                                   ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
        for frame_number, seconds in enumerate(times):
            combined = Image.new('RGB', (PANEL_WIDTH*2, PANEL_HEIGHT))
            for panel, label in enumerate(LABELS):
                case = cases[label]
                index, terminated = case.frame(float(seconds))
                frame = renderers[label].render(case, index)
                caption(frame, case, index, float(seconds), terminated, title, fonts)
                combined.paste(frame, (panel*PANEL_WIDTH, 0))
            ImageDraw.Draw(combined).line((PANEL_WIDTH, 0, PANEL_WIDTH, PANEL_HEIGHT),
                                         fill=(75, 90, 105), width=2)
            writer.append_data(np.asarray(combined))
            if frame_number == preview_index:
                combined.save(preview_path, quality=94)
                preview = combined.copy()
    finally:
        if writer is not None:
            writer.close()
        for renderer in renderers.values():
            renderer.close()
    report = dict(case_id=cid, case=reference_episode, video=str(video_path),
                  preview=str(preview_path), frames=frame_count, fps=FPS,
                  output_duration_s=frame_count/FPS, first_replay_time_s=float(times[0]),
                  last_replay_time_s=float(times[-1]), preview_time_s=float(times[preview_index]),
                  last_active_time_s={label: cases[label].last_time for label in LABELS},
                  frozen_frames={label: int(np.sum(times > cases[label].last_time + 1e-7))
                                 for label in LABELS})
    print(f'{slug}: {frame_count} frames, preview t={times[preview_index]:.2f}s -> {video_path}', flush=True)
    return report, preview


def main():
    global LABELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path,
                        help='Defaults to INPUT_DIR/replays.')
    parser.add_argument('--reference', type=Path,
                        help='11700 checkpoint, used only to locate the archived terrain/contract.')
    parser.add_argument('--duration', type=float, default=6.,
                        help='Playback duration, at most 6 seconds (default: 6).')
    parser.add_argument('--course-duration', type=float,
                        help='Optional longer course replay, at most 20 seconds, to show terminal braking.')
    args = parser.parse_args()
    if not np.isfinite(args.duration) or not .04 <= args.duration <= 6.:
        parser.error('--duration must be finite and between 0.04 and 6 seconds')
    if args.course_duration is not None and (not np.isfinite(args.course_duration) or not .04 <= args.course_duration <= 20.):
        parser.error('--course-duration must be between0.04 and20seconds')
    input_dir = args.input_dir.resolve(strict=True)
    output_dir = (args.output_dir or input_dir/'replays').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = args.reference
    results_path = input_dir/'results.json'
    if results_path.is_file():
        LABELS = tuple(read_json(results_path)['results'])
        if len(LABELS) != 2:
            raise ValueError('Replay expects exactly two policy results')
    if reference is None and results_path.is_file():
        reference = Path(read_json(results_path)['common_environment_checkpoint'])
    reference = (reference or DEFAULT_REFERENCE).resolve(strict=True)
    contract = read_json(reference.parent/'contract.json')
    profile = read_json(PROFILE)
    if profile['sha256'] != contract['asset_sha256']:
        raise ValueError('Training G1 visual profile does not match reference asset hash')
    traces, episodes = {}, {}
    required = ('active', 'time', 'root_local', 'root_quat', 'joint_pos',
                'heading_error_deg', 'body_velocity', 'command', 'joint_names')
    for label in LABELS:
        path = input_dir/f'traces_{label}.npz'
        with np.load(path, allow_pickle=False) as loaded:
            for name in required:
                if name not in loaded:
                    raise ValueError(f'{path} missing {name}')
            traces[label] = {name: loaded[name].copy() for name in required}
        rows = read_json(input_dir/f'episodes_{label}.json')
        episodes[label] = {episode_id(row): row for row in rows}
        expected = traces[label]['active'].shape
        if len(expected) != 2:
            raise ValueError(f'{path}: expected active [time, cases]')
        for name in required[:-1]:
            if traces[label][name].shape[:2] != expected:
                raise ValueError(f'{path}: inconsistent {name} time/case dimensions')
        if traces[label]['joint_pos'].shape[-1] != len(traces[label]['joint_names']):
            raise ValueError(f'{path}: joint position/name dimensions disagree')
        if traces[label]['root_local'].shape[-1] != 3 or traces[label]['root_quat'].shape[-1] != 4:
            raise ValueError(f'{path}: expected XYZ root and XYZW quaternion')
    selected = select_cases(list(episodes[LABELS[0]].values()))
    identity_keys = ('kind', 'mode', 'family', 'route', 'level', 'variant', 'replica', 'column')
    for _, _, row in selected:
        cid = episode_id(row)
        paired = episodes[LABELS[1]].get(cid)
        if paired is None or any(row.get(key) != paired.get(key) for key in identity_keys):
            raise ValueError(f'Case {cid} metadata differs between the paired traces')
    modules = archived_geometry(reference.parent/'params')
    boxes, terrain_provenance = stone_terrain(reference, selected[-1][2], modules)
    flat_scene = output_dir/'flat_replay.scene.xml'
    stone_scene = output_dir/'stone_replay.scene.xml'
    flat_model = rendering_model(None, flat_scene, profile)
    stone_model = rendering_model(boxes, stone_scene, profile)
    fonts = {name: ImageFont.truetype(FONT_PATH, size) for name, size in
             (('title', 21), ('body', 16), ('small', 15))}
    reports, previews = [], []
    for selection in selected:
        model = flat_model if selection[2]['kind'] == 'flat' else stone_model
        report, preview = render_video(model, traces, episodes, selection,
                                      output_dir, (args.course_duration if selection[2]['kind']=='stone' and args.course_duration is not None else args.duration), fonts)
        reports.append(report)
        previews.append(preview)
    sheet = Image.new('RGB', (PANEL_WIDTH*2, PANEL_HEIGHT*len(previews)), (18, 24, 31))
    for index, preview in enumerate(previews):
        sheet.paste(preview, (0, index*PANEL_HEIGHT))
    contactsheet = output_dir/'preview_contactsheet.jpg'
    sheet.save(contactsheet, quality=94)
    provenance = dict(label=REPLAY_LABEL, source='Recorded real Isaac Sim first-episode states',
                      physics_steps=0, policy_rollouts=0, sampling='zero-order hold of saved states',
                      quaternion_conversion='Isaac XYZW -> MuJoCo WXYZ',
                      terminal_handling='freeze final active frame; ignore all post-reset inactive frames',
                      input_dir=str(input_dir), reference=str(reference),
                      training_scene=str(TRAINING_SCENE), profile=str(PROFILE),
                      panel_size=[PANEL_WIDTH, PANEL_HEIGHT], fps=FPS,
                      camera=dict(distance_m=4., azimuth_deg=125., elevation_deg=-24.,
                                  follow_xy=True, lookat_height_m=.72),
                      terrain=terrain_provenance,
                      archived_geometry_sha256={name: sha256(reference.parent/'params'/f'{name}.py')
                                                for name in modules},
                      trace_sha256={label: sha256(input_dir/f'traces_{label}.npz') for label in LABELS},
                      contactsheet=str(contactsheet), videos=reports)
    manifest = output_dir/'replay_manifest.json'
    manifest.write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False))
    print(f'Contact sheet: {contactsheet}\nManifest: {manifest}', flush=True)


if __name__ == '__main__':
    main()
