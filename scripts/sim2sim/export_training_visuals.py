"""Export exact training surface triangles for camera occlusion and display."""
import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom
import trimesh


def export(profile_path):
    profile = json.loads(profile_path.read_text())
    stage = Usd.Stage.Open(str(Path(profile['source']).resolve()))
    cache = UsdGeom.XformCache()
    folder = profile_path.parent / 'assets'
    visuals = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() != 'Mesh' or '/visuals/' not in str(prim.GetPath()):
            continue
        parts = str(prim.GetPath()).split('/')
        body = stage.GetPrimAtPath('/' + '/'.join(parts[1:3]))
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        transform = np.asarray(cache.ComputeRelativeTransform(prim, body)[0])
        points = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        faces = []
        start = 0
        for count in counts:
            face = indices[start:start + count]
            faces.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))
            start += count
        name = f'training_visual_{len(visuals)}'
        trimesh.Trimesh(vertices=points, faces=faces, process=False).export(folder / f'{name}.stl')
        display_only = parts[-2] in {'head_link', 'waist_support_link'}
        visuals.append({'body': body.GetName(), 'mesh': name, 'group': 3 if display_only else 1,
                        'source': str(prim.GetPath())})
    profile['visuals'] = visuals
    profile_path.write_text(json.dumps(profile, indent=2) + '\n')
    print(f'Exported {len(visuals)} training visual meshes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    export(parser.parse_args().profile)
