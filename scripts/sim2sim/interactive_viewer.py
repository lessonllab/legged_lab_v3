"""MuJoCo GLFW viewer with terrain picking and the policy's live depth images."""
from collections import deque

import glfw
import mujoco
import numpy as np


def depth_rgb(depth, width, height):
    """Display normalized optical depth: yellow near, blue far; top row first."""
    values = np.nan_to_num(np.asarray(depth), nan=1., posinf=1., neginf=0.).clip(0., 1.)
    palette = np.array([[255, 218, 80], [29, 189, 159], [24, 90, 171], [13, 22, 58]])
    rgb = np.stack([np.interp(values, np.linspace(0, 1, len(palette)), palette[:, c]) for c in range(3)], axis=-1)
    y = np.minimum(np.arange(height) * values.shape[0] // height, values.shape[0] - 1)
    x = np.minimum(np.arange(width) * values.shape[1] // width, values.shape[1] - 1)
    return rgb[y[:, None], x].astype(np.uint8)


def walkable_surface(model, data, geom_id, point):
    """Reject the vertical sides of the bundled box stairs and corridor walls."""
    matrix = data.geom_xmat[geom_id].reshape(3, 3)
    kind = model.geom_type[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_PLANE:
        return matrix[2, 2] > .5
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        local = matrix.T @ (point - data.geom_xpos[geom_id])
        axis = np.argmin(np.abs(model.geom_size[geom_id] - np.abs(local)))
        return matrix[2, axis] * np.sign(local[axis]) > .5
    return True


class InteractiveViewer:
    def __init__(self, model, data, key_callback, *, visible=True):
        if not glfw.init():
            raise RuntimeError('Cannot initialize GLFW; use --headless if no desktop display is available.')
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
        self.window = glfw.create_window(1280, 800, 'G1 sim2sim | Click terrain to set a target', None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError('Cannot create MuJoCo display window')
        glfw.set_window_size_limits(self.window, 1000, 720, glfw.DONT_CARE, glfw.DONT_CARE)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.model, self.data = model, data
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = 4.5
        self.cam.azimuth = 135.
        self.cam.elevation = -25.
        self.cam.lookat[:] = data.qpos[:3]
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = [1, 1, 0, 1, 0, 0]
        self.key_callback = key_callback
        self.targets = deque()
        self.follow = True
        self.target_point = None
        self.message = 'Click a floor or stair surface to walk there'
        self.course_help = ''
        self.mouse_down = None
        self.last_cursor = None
        self.dragged = False
        glfw.set_key_callback(self.window, self._key)
        glfw.set_mouse_button_callback(self.window, self._mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._cursor)
        glfw.set_scroll_callback(self.window, self._scroll)

    def viewport(self):
        width, height = glfw.get_framebuffer_size(self.window)
        panel = min(360, max(240, width // 3))
        return mujoco.MjrRect(0, 0, max(1, width - panel), max(1, height))

    def window_point(self, x, y):
        """GLFW logical pixels -> framebuffer pixels (including HiDPI)."""
        ww, wh = glfw.get_window_size(self.window)
        fw, fh = glfw.get_framebuffer_size(self.window)
        return x * fw / max(1, ww), fh - y * fh / max(1, wh)

    def pick(self, x, y):
        """Return world terrain surface position; ignore robot, background, panel."""
        px, py = self.window_point(x, y)
        viewport = self.viewport()
        if not (0 <= px < viewport.width and 0 <= py < viewport.height):
            return None
        point = np.zeros(3)
        geom, flex, skin = (np.full(1, -1, dtype=np.int32) for _ in range(3))
        body = mujoco.mjv_select(self.model, self.data, self.opt, viewport.width / viewport.height,
                                px / viewport.width, py / viewport.height, self.scene, point, geom, flex, skin)
        if body != 0 or geom[0] < 0 or not np.isfinite(point).all():
            return None
        if not walkable_surface(self.model, self.data, geom[0], point):
            return None
        return point

    def _key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_F:
            self.follow = not self.follow
        else:
            self.key_callback(key)

    def _mouse_button(self, window, button, action, mods):
        x, y = glfw.get_cursor_pos(window)
        if action == glfw.PRESS:
            px, _ = self.window_point(x, y)
            if px >= self.viewport().width:
                return
            self.mouse_down = (button, x, y)
            self.last_cursor = (x, y)
            self.dragged = False
        elif action == glfw.RELEASE and self.mouse_down is not None:
            pressed, start_x, start_y = self.mouse_down
            if button != pressed:
                return
            if button == glfw.MOUSE_BUTTON_LEFT and not self.dragged and np.hypot(x-start_x, y-start_y) <= 5:
                point = self.pick(x, y)
                if point is not None:
                    self.target_point = point.copy()
                    self.targets.append(point)
                    self.message = f'Target: ({point[0]:.2f}, {point[1]:.2f})'
                else:
                    self.message = 'Click floor or stair tops; walls and robot are ignored'
            self.mouse_down = None
            self.last_cursor = None

    def _cursor(self, window, x, y):
        if self.mouse_down is None:
            return
        button, sx, sy = self.mouse_down
        dx, dy = x - self.last_cursor[0], y - self.last_cursor[1]
        self.last_cursor = (x, y)
        self.dragged |= np.hypot(x-sx, y-sy) > 5
        if not self.dragged:
            return
        if button == glfw.MOUSE_BUTTON_LEFT:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
            self.follow = False
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        height = max(1, glfw.get_window_size(window)[1])
        mujoco.mjv_moveCamera(self.model, action, dx/height, dy/height, self.scene, self.cam)

    def _scroll(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0., -.05*yoffset, self.scene, self.cam)

    def poll(self):
        glfw.poll_events()

    def is_running(self):
        return self.window is not None and not glfw.window_should_close(self.window)

    def _marker(self, target):
        if self.target_point is None or not np.allclose(self.target_point[:2], target):
            # Keyboard and CLI targets also get an actual terrain height.
            origin = np.array([target[0], target[1], 100.])
            geom = np.full(1, -1, dtype=np.int32)
            distance = mujoco.mj_ray(self.model, self.data, origin, np.array([0., 0., -1.]),
                                    np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), 1, -1, geom)
            self.target_point = np.array([target[0], target[1], 100.-distance if distance >= 0 else 0.])
        point = self.target_point + np.array([0., 0., .015])
        if self.scene.ngeom + 2 > self.scene.maxgeom:
            return
        rgba = np.array([.15, 1., .55, .65], dtype=np.float32)
        geom = self.scene.geoms[self.scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([.18, .012, 0.]), point, np.eye(3).ravel(), rgba)
        self.scene.ngeom += 1
        geom = self.scene.geoms[self.scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, .04), point + [0, 0, .15], np.eye(3).ravel(), rgba)
        self.scene.ngeom += 1

    def _text(self, text, left, bottom, width, height=30):
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           mujoco.MjrRect(int(left), int(bottom), int(width), int(height)), text, '', self.context)

    def _depth_panel(self, runner, left, width, height, target, command, mode):
        mujoco.mjr_rectangle(mujoco.MjrRect(left, 0, width, height), .045, .06, .085, 1.)
        margin = 16
        image_width = max(16, width - 2*margin)
        # Keep both feeds and their labels visible on smaller windows.
        image_height = min(int(image_width*9/16), max(18, (height-330)//2))
        y = height - 42
        self._text('LIVE DEPTH', left+margin, y, image_width)
        y -= 36
        self._text('Optical depth | 64 x 36', left+margin, y, image_width)
        y -= image_height
        if runner.raw_depth is not None:
            image = depth_rgb(runner.raw_depth/2.5, image_width, image_height)
            # White outline identifies the crop used by the actor.
            x0, x1, y0 = image_width//4, 3*image_width//4, image_height//2
            image[y0:y0+2, x0:x1] = 255
            image[y0:, x0:x0+2] = 255
            image[y0:, x1-2:x1] = 255
            image[-2:, x0:x1] = 255
            mujoco.mjr_drawPixels(np.ascontiguousarray(image[::-1]).ravel(), None, mujoco.MjrRect(left+margin, y, image_width, image_height), self.context)
        y -= 46
        self._text('Actor input | latest 32 x 18', left+margin, y, image_width)
        y -= image_height
        if runner.policy_depth is not None:
            image = depth_rgb(runner.policy_depth, image_width, image_height)
            mujoco.mjr_drawPixels(np.ascontiguousarray(image[::-1]).ravel(), None, mujoco.MjrRect(left+margin, y, image_width, image_height), self.context)
        y -= 40
        self._text('Yellow: near     Blue: far / no return', left+margin, y, image_width)
        y -= 25
        self._text('Depth range: 0 - 2.5 m', left+margin, y, image_width)
        y -= 40
        self._text(f'Target: {target[0]:.2f}, {target[1]:.2f}\nDistance: {np.linalg.norm(target-runner.data.qpos[:2]):.2f} m\nCommand: {command[0]:.2f}, {command[1]:.2f}, {command[2]:.2f}\nMode: {mode}    Time: {runner.data.time:.1f} s', left+margin, y-70, image_width, 100)

    def draw(self, runner, target, command, mode, *, capture=False):
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return None
        viewport = self.viewport()
        if self.follow:
            self.cam.lookat[:] = self.data.qpos[:3]
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        if mode == 'target':
            self._marker(target)
        mujoco.mjr_render(viewport, self.scene, self.context)
        self._text(self.message, 12, height-45, viewport.width-24)
        if self.course_help:
            self._text(self.course_help, 12, height-100, viewport.width-24, 55)
        self._text('Click: target | Drag: orbit | Right drag: pan | Wheel: zoom\nSpace: stop | F: follow camera | Esc: exit', 12, 10, viewport.width-24, 55)
        self._depth_panel(runner, viewport.width, width-viewport.width, height, target, command, mode)
        pixels = None
        if capture:
            pixels = np.empty((height, width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(pixels, None, mujoco.MjrRect(0, 0, width, height), self.context)
            pixels = pixels[::-1].copy()
        glfw.swap_buffers(self.window)
        return pixels

    def close(self):
        if self.window is not None:
            glfw.make_context_current(self.window)
            self.context.free()
            glfw.destroy_window(self.window)
            self.window = None
            glfw.terminate()
