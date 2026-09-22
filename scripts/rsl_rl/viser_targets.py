"""CPU terrain picking and queued playback commands, independent of the renderer."""
from queue import Empty, SimpleQueue

import numpy as np


class TerrainRayPicker:
    def __init__(self):
        self.meshes = []
        self.plane_z = None

    def add_mesh(self, vertices, faces):
        self.meshes.append((vertices, faces))

    def pick(self, origin, direction):
        """Nearest positive, double-sided ray/triangle hit in world coordinates."""
        origin, direction = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
        if origin.shape != (3,) or direction.shape != (3,):
            return None
        norm = np.linalg.norm(direction)
        if not np.isfinite(origin).all() or not np.isfinite(norm) or norm < 1e-12:
            return None
        direction = direction / norm
        nearest = np.inf
        # Bound temporary memory even for large generated terrain meshes.
        for vertices, faces in self.meshes:
            for start in range(0, len(faces), 65536):
                triangles = vertices[faces[start:start + 65536]].astype(np.float64)
                edge1 = triangles[:, 1] - triangles[:, 0]
                edge2 = triangles[:, 2] - triangles[:, 0]
                p = np.cross(direction, edge2)
                det = np.einsum('ij,ij->i', edge1, p)
                valid = np.abs(det) > 1e-12
                inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
                offset = origin - triangles[:, 0]
                u = np.einsum('ij,ij->i', offset, p) * inv
                q = np.cross(offset, edge1)
                v = q @ direction * inv
                t = np.einsum('ij,ij->i', edge2, q) * inv
                valid &= (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1. + 1e-8) & (t > 1e-8)
                if valid.any():
                    nearest = min(nearest, float(t[valid].min()))
        if self.plane_z is not None and abs(direction[2]) > 1e-12:
            t = (self.plane_z - origin[2]) / direction[2]
            if t > 1e-8:
                nearest = min(nearest, t)
        return origin + nearest * direction if np.isfinite(nearest) else None


class ClickTargetControl:
    """Callbacks enqueue CPU data; apply() alone may touch simulation tensors."""
    def __init__(self, command, picker):
        self.command = command
        self.picker = picker
        self.events = SimpleQueue()
        self.owner = None
        self.message = '单击地形，给当前机器人设置目标；其他机器人自动随机目标。'

    def click(self, env_id, origin, direction):
        self.events.put(('click', env_id, tuple(origin), tuple(direction)))

    def release(self, env_id):
        self.events.put(('release', env_id, None, None))

    def apply(self, selected, enabled=True):
        changed = False
        if self.owner is not None and (self.owner != selected or not enabled):
            self.command.release_manual_target(self.owner)
            self.owner = None
            changed = True
            self.message = '已恢复随机目标。单击地形可控制当前机器人。'
        while True:
            try:
                event, env_id, origin, direction = self.events.get_nowait()
            except Empty:
                break
            # A click captured before switching selection must never steer the new robot.
            if env_id != selected:
                continue
            if event == 'release':
                self.command.release_manual_target(env_id)
                self.owner = None
                self.message = f'机器人 {env_id} 已恢复随机目标。'
                changed = True
            elif enabled:
                point = self.picker.pick(origin, direction)
                if point is None:
                    self.message = '未命中地形，请点击台阶、坡面或地面。'
                    continue
                self.command.set_manual_target(env_id, point)
                self.owner = env_id
                self.message = f'机器人 {env_id} 目标：({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}) 米。'
                changed = True
        return changed
