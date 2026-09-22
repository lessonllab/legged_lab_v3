"""Export the checkpoint's inverted stair tile as exact MuJoCo boxes.

Only the two pure geometry functions are loaded from Isaac Lab's local source;
the resulting XML runs without Isaac Sim. Coordinates are relative to the
training terrain origin (the central platform, not the outer border).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from run_mujoco import ROOT, read_config


def training_meshes(cfg, height, isaaclab):
    namespace = {'np': np, 'trimesh': trimesh}
    directory = isaaclab / 'source/isaaclab/isaaclab/terrains/trimesh'
    for filename, name in [('utils.py', 'make_border'), ('mesh_terrains.py', 'inverted_pyramid_stairs_terrain')]:
        source = directory / filename
        function = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    terrain = cfg['scene']['terrain']['terrain_generator']['sub_terrains']['pyramid_stairs_inv']
    original = dict(terrain['original'])
    original.update(size=terrain['size'], step_height_range=(height, height))
    meshes, origin = namespace['inverted_pyramid_stairs_terrain'](0., SimpleNamespace(**original))
    for mesh in meshes:
        mesh.apply_translation(-origin)
    return meshes, origin, original


def export(config, destination, height, isaaclab):
    cfg = read_config(config)
    meshes, origin, terrain = training_meshes(cfg, height, isaaclab)
    # Training tiles touch along X. Omitting the next tile makes the camera
    # see a void at the exit, although the robot is still on the last steps.
    heights = [0., .08, .10, .12, .14, .16, .18, .20, .215, .23]
    level = next((i for i, value in enumerate(heights) if np.isclose(value, height)), None)
    neighbors = []
    if level is not None:
        for direction in (-1, 1):
            neighbor_level = level + direction
            if not 1 <= neighbor_level < len(heights):
                continue
            neighbor_height = heights[neighbor_level]
            extra, extra_origin, _ = training_meshes(cfg, neighbor_height, isaaclab)
            shift = np.array([direction * terrain['size'][0], 0., extra_origin[2] - origin[2]])
            for mesh in extra:
                mesh.apply_translation(shift)
            meshes.extend(extra)
            neighbors.append({'offset_x': float(shift[0]), 'height': neighbor_height})
    source = ROOT / 'hiking-in-the-wild-sim2sim/unitree_mujoco/unitree_robots/g1/scene_29dof_terrain_with_camera.xml'
    root = ET.parse(source).getroot()
    root.set('model', 'G1 training inverted stairs')
    root.find('include').set('file', os.path.relpath(source.parent / 'g1_29dof_with_camera.xml', destination.parent))
    world = root.find('worldbody')
    for geom in list(world.findall('geom')):
        world.remove(geom)
    for index, mesh in enumerate(meshes):
        low, high = mesh.bounds
        # These source functions generate axis-aligned boxes exclusively.
        if len(mesh.vertices) != 8 or not np.isclose(mesh.volume, np.prod(high - low)):
            raise ValueError('Training terrain contains a non-box mesh')
        ET.SubElement(world, 'geom', name=f'training_terrain_{index}', type='box',
                      pos=' '.join(f'{v:.12g}' for v in (high + low) / 2),
                      size=' '.join(f'{v:.12g}' for v in (high - low) / 2),
                      rgba='.45 .53 .6 1')
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding='unicode')
    metadata = {'config': str(config.relative_to(ROOT)), 'height': height,
                'tread_width': terrain['step_width'], 'tile_size': terrain['size'],
                'training_origin': origin.tolist(), 'spawn': [0., 0., .8],
                'target': [6.7, 0., float(-origin[2])], 'rises': round(-origin[2] / height),
                'neighbor_tiles': neighbors}
    destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'scripts/sim2sim/scenes/training_stairs_12cm.xml')
    parser.add_argument('--height', type=float, default=.12)
    parser.add_argument('--isaaclab', type=Path, default=Path('/home/ljc/isaaclab6/IsaacLab'))
    args = parser.parse_args()
    if args.height <= 0:
        parser.error('height must be positive')
    export(args.config.resolve(), args.output.resolve(), args.height, args.isaaclab)
