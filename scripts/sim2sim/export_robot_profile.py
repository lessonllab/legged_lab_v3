"""Extract inertial and joint-frame calibration from the project's training USD."""
import argparse
import hashlib
import json
from pathlib import Path
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import numpy as np
from scipy.spatial import ConvexHull


def quaternion(q):
    return [q.GetReal(), *q.GetImaginary()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('usd', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    stage = Usd.Stage.Open(str(args.usd.resolve()))
    profile = {'source': str(args.usd), 'sha256': hashlib.sha256(args.usd.read_bytes()).hexdigest(), 'bodies': {}, 'joints': {}}
    profile['collisions'] = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(prim)
            profile['bodies'][prim.GetName()] = {
                'mass': mass.GetMassAttr().Get(), 'ipos': list(mass.GetCenterOfMassAttr().Get()),
                'iquat': quaternion(mass.GetPrincipalAxesAttr().Get()), 'inertia': list(mass.GetDiagonalInertiaAttr().Get())}
        if prim.GetTypeName() == 'PhysicsRevoluteJoint':
            joint = UsdPhysics.RevoluteJoint(prim)
            if list(joint.GetLocalPos1Attr().Get()) != [0., 0., 0.] or quaternion(joint.GetLocalRot1Attr().Get()) != [1., 0., 0., 0.]:
                raise ValueError(f'Nonidentity child joint frame: {prim.GetPath()}')
            profile['joints'][prim.GetName()] = {'pos': list(joint.GetLocalPos0Attr().Get()), 'quat': quaternion(joint.GetLocalRot0Attr().Get())}
    cache = UsdGeom.XformCache()
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        parts = str(prim.GetPath()).split('/')
        body = stage.GetPrimAtPath('/' + '/'.join(parts[1:3]))
        transform = Gf.Transform(cache.ComputeRelativeTransform(prim, body)[0])
        shape = {'body': body.GetName(), 'pos': list(transform.GetTranslation()), 'quat': quaternion(transform.GetRotation().GetQuat())}
        scale = list(transform.GetScale())
        kind = prim.GetTypeName()
        if kind == 'Mesh':
            mesh = UsdGeom.Mesh(prim)
            name = f'collision_{len(profile["collisions"])}'
            folder = args.output.parent / 'assets'; folder.mkdir(exist_ok=True)
            # MuJoCo mesh contacts use convex hulls. Save that same hull rather
            # than duplicating tens of MB of non-collision surface triangles.
            points = np.asarray(mesh.GetPointsAttr().Get()) * scale
            hull = ConvexHull(points)
            used = np.unique(hull.simplices)
            remap = {old: new + 1 for new, old in enumerate(used)}
            lines = ['# Convex collision hull extracted from the project training USD.']
            lines += ['v ' + ' '.join(map(str, p)) for p in points[used]]
            for face, equation in zip(hull.simplices, hull.equations):
                a, b, c = points[face]
                if np.dot(np.cross(b-a, c-a), equation[:3]) < 0:
                    face = face[::-1]
                lines.append('f ' + ' '.join(str(remap[i]) for i in face))
            (folder / f'{name}.obj').write_text('\n'.join(lines) + '\n')
            shape.update(type='mesh', mesh=name)
        elif kind == 'Cube':
            shape.update(type='box', size=[s * UsdGeom.Cube(prim).GetSizeAttr().Get() / 2 for s in scale])
        elif kind == 'Sphere':
            shape.update(type='sphere', size=[UsdGeom.Sphere(prim).GetRadiusAttr().Get() * scale[0]])
        elif kind == 'Cylinder':
            cylinder = UsdGeom.Cylinder(prim)
            if cylinder.GetAxisAttr().Get() != 'Z':
                raise ValueError(f'Unsupported cylinder axis: {prim.GetPath()}')
            shape.update(type='cylinder', size=[cylinder.GetRadiusAttr().Get() * scale[0], cylinder.GetHeightAttr().Get() * scale[2] / 2])
        else:
            raise ValueError(f'Unsupported collider {kind}')
        profile['collisions'].append(shape)
    args.output.write_text(json.dumps(profile, indent=2) + '\n')


if __name__ == '__main__':
    main()
