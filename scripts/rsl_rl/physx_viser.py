"""Browser viewer for the project's PhysX articulation, without Newton imports.

USD supplies static visual geometry; live body poses come from PhysX tensors.
Rendering reads live poses; optional terrain clicks queue playback target commands.
"""

import numpy as np
import viser
from pxr import Usd, UsdGeom
from viser_targets import ClickTargetControl, TerrainRayPicker
from sole_diagnostics import sole_diagnostic
from vision_overlay import view_boundary, enlarged_depth


def _numpy(value):
    if hasattr(value, "torch"):
        value = value.torch
    return value.detach().cpu().numpy()


def _mesh(prim, relative_to=None):
    mesh = UsdGeom.Mesh(prim)
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if len(vertices) == 0 or len(indices) == 0:
        return None
    transform = np.asarray(mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    if relative_to is not None:
        transform = transform @ np.linalg.inv(relative_to)
    vertices = vertices @ transform[:3, :3] + transform[3, :3]
    # Imported robot/terrain meshes are triangles; support polygons as a fan too.
    if np.all(counts == 3):
        faces = indices.reshape(-1, 3)
    else:
        faces, start = [], 0
        for count in counts:
            face = indices[start:start + count]
            faces.extend((face[0], face[j], face[j + 1]) for j in range(1, count - 1))
            start += count
        faces = np.asarray(faces).reshape(-1, 3)
    if mesh.GetOrientationAttr().Get() == "leftHanded":
        faces = faces[:, ::-1]
    return vertices.astype(np.float32), faces.astype(np.uint32)


def _mesh_prims(root):
    return (prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if prim.IsA(UsdGeom.Mesh))


class PhysxViser:
    def __init__(self, env, host="127.0.0.1", port=8080, follow=False, asset_name="robot", selected_env=0):
        self.env = env
        self.reset_observations = False
        self.control_requests = None
        self.robot = env.scene[asset_name]
        self.server = viser.ViserServer(host=host, port=port, label="G1 depth locomotion")
        self.server.scene.set_up_direction("+z")
        self.pause = self.server.gui.add_checkbox("暂停播放", False)
        self.follow = self.server.gui.add_checkbox("跟随机器人", follow)
        terrain_names = getattr(env, "terrain_showcase_names", None)
        terrain_labels = {"pyramid_stairs": "上阶梯", "pyramid_stairs_inv": "下阶梯",
                          "boxes": "块状地形", "random_rough": "起伏路面",
                          "hf_pyramid_slope": "坡面", "hf_pyramid_slope_inv": "反向坡面",
                          "hf_steppingstones": "梅花桩"}
        options = [f"{i} · {terrain_labels.get(name, name)}" for i, name in enumerate(terrain_names)] if terrain_names else [str(i) for i in range(env.num_envs)]
        self._selection_ids = {name: i for i, name in enumerate(options)}
        if not 0 <= selected_env < env.num_envs:
            raise ValueError('Selected viewer environment is out of range')
        self.selected = self.server.gui.add_dropdown("观察机器人", options, initial_value=options[selected_env])
        self.overview = self.server.gui.add_button("全部机器人总览") if terrain_names else None
        self.server.gui.add_markdown("拖动旋转视角，滚轮缩放。" + (
            "深度预览为策略最近一帧输入。" if asset_name == "robot" else
            "参考动作直接设置机器人姿态，不经过策略、平衡控制或接触动力学。"))
        if getattr(env, 'terrain_control_play', False):
            from queue import SimpleQueue
            from legged_lab.tasks.locomotion.amp.mdp.stair_play_matrix import CONTROL_TERRAINS
            self.control_requests = SimpleQueue()
            self.control_kind = self.server.gui.add_dropdown('地形类型', list(CONTROL_TERRAINS), initial_value='上楼')
            self.control_level = self.server.gui.add_dropdown('难度 / 阶高',
                ['1 · 8 cm', '2 · 10 cm', '3 · 12 cm', '4 · 14 cm', '5 · 16 cm', '6 · 18 cm', '7 · 20 cm', '8 · 25 cm', '9 · 30 cm'],
                initial_value='2 · 10 cm')
            self.control_speed = self.server.gui.add_slider('速度指令上限 (m/s)', min=.1, max=1.5, step=.05,
                                                           initial_value=env.cfg.stair_matrix_speed)
            self.control_apply = self.server.gui.add_button('应用地形并重置机器人')
            self.control_info = self.server.gui.add_markdown('当前：' + env.command_manager.get_term('base_velocity').control_label())
            self.server.gui.add_markdown('仅生成 1 个机器人。楼梯固定 17 级；其他地形使用难度 1–9，平地忽略难度。修改后点击应用，统计重新计数。')

            @self.control_apply.on_click
            def on_control_apply(_):
                self.control_requests.put((self.control_kind.value, int(self.control_level.value.split(' · ')[0]),
                                           float(self.control_speed.value)))
        self.depth_preview = None
        self.depth_camera = None
        try:
            self.depth_camera = env.scene['depth_camera']
        except KeyError:
            pass
        if self.depth_camera is not None:
            self.show_policy_view = self.server.gui.add_checkbox('显示策略视觉范围（绿色）', True, order=-30)
            self.show_camera_view = self.server.gui.add_checkbox('显示相机完整视野（蓝色）', False, order=-29)
            self.vision_info = self.server.gui.add_markdown(
                '绿色为策略使用区域，蓝色为完整相机视野。边界随相机运动；遮挡后的区域不可见。', order=-28)
            self.policy_view = self.server.scene.add_line_segments('/vision/policy',
                points=np.zeros((1,2,3),dtype=np.float32),colors=(40,240,110),line_width=2.)
            self.camera_view = self.server.scene.add_line_segments('/vision/camera',
                points=np.zeros((1,2,3),dtype=np.float32),colors=(70,160,255),line_width=1.5,visible=False)
        self.play_episodes = np.zeros(env.num_envs, dtype=int)
        self.play_timeouts = np.zeros(env.num_envs, dtype=int)
        self.play_crossings = np.zeros(env.num_envs, dtype=int)
        self.play_failures = np.zeros(env.num_envs, dtype=int)
        self.matrix_status = self.server.gui.add_markdown('多高度测试：等待回合结果。') if getattr(env,'stair_matrix_play',False) or getattr(env,'terrain_control_play',False) else None
        if getattr(env, 'stair_matrix_play', False):
            self.server.gui.add_markdown(f'每块地形一个机器人；楼梯均为17级，阶高8–30cm。速度上限 {env.cfg.stair_matrix_speed:g} m/s。\n\n选择机器人近看，或点击“全部机器人总览”。穿越计数不代表速度或踩边指标合格。')
        if getattr(env, 'play_checkpoint_label', None):
            self.server.gui.add_markdown('当前模型：' + env.play_checkpoint_label)
        self.play_status = self.server.gui.add_markdown("重放统计：等待回合结束。") if getattr(env, "instinct_play", False) else None
        self.velocity_arrows = None
        self.target_marker = None
        self.target_control = None
        if asset_name == "robot" and "base_velocity" in env.command_manager.active_terms:
            self.show_velocity = self.server.gui.add_checkbox("显示速度箭头", True)
            self.velocity_info = self.server.gui.add_markdown("绿色：目标平移速度；橙色：实际平移速度。箭头长度 0.6 米代表 1 m/s。")
            self.velocity_arrows = [self.server.scene.add_line_segments(
                f"/velocity/{name}", points=np.zeros((3, 2, 3), dtype=np.float32),
                colors=color, line_width=5., visible=False,
            ) for name, color in (("command", (45, 205, 90)), ("actual", (255, 150, 35)))]
            command_term = env.command_manager.get_term("base_velocity")
            if hasattr(command_term, "pos_command_w"):
                self.show_target = self.server.gui.add_checkbox("显示行进目标", True)
                angle = np.linspace(0., 2*np.pi, 33)
                radius = command_term.cfg.target_dis_threshold
                circle = np.stack((radius*np.cos(angle), radius*np.sin(angle), np.zeros_like(angle)), -1)
                self.target_marker = self.server.scene.add_line_segments(
                    "/command_target", points=np.stack((circle[:-1], circle[1:]), 1).astype(np.float32),
                    colors=(245, 80, 80), line_width=5.)
                self.target_label = self.server.scene.add_label("/command_target_label", "行进目标")
                if hasattr(command_term, "set_manual_target"):
                    self.target_control = ClickTargetControl(command_term, TerrainRayPicker())
                    self.click_targets = self.server.gui.add_checkbox("点击地形设置目标", True)
                    if getattr(env, 'stair_matrix_play', False):
                        self.click_targets.value = False
                    self.random_target = self.server.gui.add_button("当前机器人恢复随机目标")
                    self.target_help = self.server.gui.add_markdown(
                        "单击地形设置目标，拖动旋转视角。到达后停下，等待下一次点击。\n\n"
                        "切换机器人时，原机器人恢复随机目标；摔倒或超时重置也恢复随机目标。")
                    self.target_feedback = self.server.gui.add_markdown(self.target_control.message)

                    @self.server.scene.on_click()
                    def on_terrain_click(event):
                        if self.click_targets.value:
                            self.target_control.click(self._selection_ids[self.selected.value],
                                                      event.ray_origin, event.ray_direction)

                    @self.random_target.on_click
                    def on_random_target(_):
                        self.target_control.release(self._selection_ids[self.selected.value])
        self._camera_targets = {}
        self._positions = _numpy(self.robot.data.body_pos_w)
        self._frames = []
        self._batched_bodies = []
        self._robot_labels = [self.server.scene.add_label(f"/robot_labels/{i}", label,
                              font_screen_scale=.7) for i, label in enumerate(options)] if terrain_names else []
        stage = env.sim.stage
        ground = stage.GetPrimAtPath("/World/ground")
        ground_meshes = 0
        for index, prim in enumerate(_mesh_prims(ground)):
            geometry = _mesh(prim)
            if geometry is not None:
                if self.target_control is not None:
                    self.target_control.picker.add_mesh(*geometry)
                self.server.scene.add_mesh_simple(f"/terrain/{index}", *geometry,
                                                  color=(155, 165, 160), side="double",
                                                  flat_shading=True)
                ground_meshes += 1
        if ground_meshes == 0 and env.scene.terrain.cfg.terrain_type == "plane":
            self.server.scene.add_grid("/ground", width=200, height=200, plane="xy", plane_opacity=1.0)
            if self.target_control is not None:
                self.target_control.picker.plane_z = 0.
        elif ground_meshes == 0:
            raise RuntimeError("Viser could not find terrain geometry")

        root_path = self.robot.cfg.prim_path.replace("{ENV_REGEX_NS}", env.scene.env_prim_paths[0])
        # InteractiveScene may already have expanded the regex in asset cfg.
        root_path = root_path.replace("env_[^/]+", "env_0").replace("env_.*", "env_0")
        body_count = 0
        for body_id, body_name in enumerate(self.robot.body_names):
            body = stage.GetPrimAtPath(f"{root_path}/{body_name}")
            visuals = stage.GetPrimAtPath(f"{root_path}/{body_name}/visuals")
            if not body.IsValid() or not visuals.IsValid():
                continue
            body_world = np.asarray(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            geometries = [geometry for prim in _mesh_prims(visuals)
                          if (geometry := _mesh(prim, body_world)) is not None]
            if not geometries:
                continue
            vertices, faces, offset = [], [], 0
            for v, f in geometries:
                vertices.append(v)
                faces.append(f + offset)
                offset += len(v)
            geometry = np.concatenate(vertices), np.concatenate(faces)
            if getattr(env, 'stair_matrix_play', False):
                mesh = self.server.scene.add_batched_meshes_simple(
                    f'/robots/{body_name}', *geometry,
                    batched_wxyzs=np.tile([1., 0., 0., 0.], (env.num_envs, 1)),
                    batched_positions=np.zeros((env.num_envs, 3)),
                    batched_colors=(205, 210, 220), lod='off')
                self._batched_bodies.append((body_id, mesh))
                body_count += 1
                continue
            for env_id in range(env.num_envs):
                name = f"/robots/{env_id}/{body_name}"
                frame = self.server.scene.add_frame(name, show_axes=False)
                self.server.scene.add_mesh_simple(name + "/mesh", *geometry, color=(205, 210, 220))
                self._frames.append((env_id, body_id, frame))
            body_count += 1
        if body_count == 0:
            raise RuntimeError(f"Viser could not find robot visual meshes under {root_path}")

        @self.server.on_client_connect
        def on_connect(client):
            if self.overview is not None and not self.follow.value:
                self._overview(client)
            else:
                self._focus(client)

        @self.server.on_client_disconnect
        def on_disconnect(client):
            self._camera_targets.pop(client.client_id, None)

        @self.selected.on_update
        def on_select(_):
            if self.overview is not None:
                self.follow.value = True
            for client in self.server.get_clients().values():
                self._focus(client)

        if self.overview is not None:
            @self.overview.on_click
            def on_overview(_):
                self.follow.value = False
                for client in self.server.get_clients().values():
                    self._overview(client)

        self.virtual_info = None
        from legged_lab.tasks.locomotion.amp.mdp.stair_virtual_safety import virtual_stair_reward
        edge_cfg = getattr(env.cfg.rewards, 'feet_edge', None) if hasattr(env.cfg, 'rewards') else None
        if edge_cfg is not None and edge_cfg.func is virtual_stair_reward:
            self.virtual_info = self.server.gui.add_markdown('虚拟边缘检测：等待数据')
            self.show_virtual = self.server.gui.add_checkbox('显示边缘轴线与脚部采样点', True)
            self.virtual_lines = self.server.scene.add_line_segments('/safety/edge_axes',
                points=np.zeros((1,2,3),dtype=np.float32),colors=(255,180,40),line_width=3.)
            self.virtual_points = self.server.scene.add_point_cloud('/safety/foot_points',
                points=np.zeros((1,3),dtype=np.float32),colors=(50,210,70),point_size=.012,precision='float32')
        self.sole_info = None
        # Backend-created ray casters may live outside scene.sensors.
        try:
            for name in ('left_sole_scanner', 'right_sole_scanner'):
                env.scene[name]
        except KeyError:
            pass
        else:
            if self.virtual_info is None:
                self.sole_info = self.server.gui.add_markdown('脚底支撑：等待数据')
        self.update()
        print(f"[INFO] PhysX Viser ready: http://localhost:{self.server.get_port()} "
              f"({env.num_envs} robots, {body_count} visual bodies each; no Newton)", flush=True)

    def _focus(self, client):
        target = self._positions[self._selection_ids[self.selected.value], 0].copy()
        client.camera.up_direction = (0.0, 0.0, 1.0)
        names = getattr(self.env, 'terrain_showcase_names', [])
        selected = self._selection_ids[self.selected.value]
        offset = ([5., -5., 5.] if names and names[selected] == 'hf_steppingstones' else
                  [4.0, .75, 1.0] if getattr(self.env, "instinct_play", False) else [3., -3., 2.])
        client.camera.position = target + np.array(offset)
        client.camera.look_at = target
        self._camera_targets[client.client_id] = target

    def _overview(self, client):
        points = self._positions[:, 0]
        center = points.mean(axis=0)
        span = max(8., float(np.ptp(points, axis=0).max()))
        client.camera.up_direction = (0., 0., 1.)
        client.camera.position = center + np.array([span*.7, -span*.55, span*.8])
        client.camera.look_at = center
        self._camera_targets[client.client_id] = points[self._selection_ids[self.selected.value]].copy()

    def update(self, observations=None):
        self._positions = _numpy(self.robot.data.body_pos_w)
        quaternions = _numpy(self.robot.data.body_quat_w)[..., [3, 0, 1, 2]]  # XYZW -> WXYZ
        with self.server.atomic():
            if self.matrix_status is not None:
                selected = self._selection_ids[self.selected.value]
                n, crossed, failed = self.play_episodes[selected], self.play_crossings[selected], self.play_failures[selected]
                self.matrix_status.content = f'{self.selected.value}\n\n已结束 {n} 回合；失败 {failed} 次。'
                if int(self.env.command_manager.get_term('base_velocity').group[selected]) in (1,2):
                    self.matrix_status.content += f'\n\n穿越 {crossed} 次；未穿越超时 {n-crossed-failed} 次。'
            if self.virtual_info is not None:
                from legged_lab.tasks.locomotion.amp.mdp.stair_virtual_safety import get_stair_virtual_safety
                safety = get_stair_virtual_safety(self.env).update(self.env, force=True)
                selected = self._selection_ids[self.selected.value]
                command = safety.command
                active = bool(command.stairs[selected])
                depth = _numpy(safety.depth)[selected]
                near = _numpy(safety.toe_near)[selected]
                lines = []
                for foot, label in enumerate(('左脚','右脚')):
                    lines.append(f'{label}：虚拟边缘区内 {(depth[foot]>0).sum()}/100 点；'
                                 f'最大穿入 {depth[foot].max()*100:.2f} cm；脚尖临近立面：{"是" if near[foot] else "否"}')
                self.virtual_info.content = ('\n\n'.join(lines) if active else '当前非楼梯，未启用楼梯虚拟边缘检测。') + (
                    '\n\nInstinct 式：半径 5 cm 的虚拟边缘圆柱，代价 = 穿入深度 × 点速度之和。'
                    '进入安全区不等于实际撞击；轴线为橙色，区内采样点为红色。')
                visible = bool(self.show_virtual.value and active)
                self.virtual_lines.visible = self.virtual_points.visible = visible
                if visible:
                    row = int(_numpy(command.terrain.terrain_levels)[selected])
                    direction = int(command.group[selected])-1
                    origin = _numpy(command.terrain.env_origins)[selected]
                    segments = []
                    for radius,z in _numpy(safety.rings)[direction,row]:
                        if radius > 1e5: continue
                        corners = np.array([[-radius,-radius,z],[radius,-radius,z],
                                            [radius,radius,z],[-radius,radius,z]])+origin
                        segments.extend([[corners[i],corners[(i+1)%4]] for i in range(4)])
                    self.virtual_lines.points = np.asarray(segments,dtype=np.float32).reshape(-1,2,3)
                    self.virtual_points.points = _numpy(safety.points_w)[selected].reshape(-1,3)
                    self.virtual_points.colors = np.where((depth.reshape(-1)>0)[:,None],
                                                          [245,50,50],[50,210,70]).astype(np.uint8)
            if self.sole_info is not None and self.virtual_info is None:
                selected = self._selection_ids[self.selected.value]
                contact = self.env.scene['contact_forces']
                lines = []
                for side, label in (('left', '左脚'), ('right', '右脚')):
                    name = f'{side}_ankle_roll_link'
                    body = self.robot.body_names.index(name)
                    hits = _numpy(self.env.scene[f'{side}_sole_scanner'].data.ray_hits_w)[selected]
                    q = _numpy(self.robot.data.body_quat_w)[selected, body]
                    v = hits - self._positions[selected, body]
                    # Inverse XYZW quaternion rotation into the sole frame.
                    with np.errstate(invalid='ignore'):
                        local = v + 2 * np.cross(-q[:3], np.cross(-q[:3], v) + q[3] * v)
                    h = local[:, 2].reshape(5, 11)
                    force = _numpy(contact.data.net_normal_forces_w)[selected, contact.body_names.index(name), 2]
                    status, fraction, missing = sole_diagnostic(h, force)
                    lines.append(f'{label}：{status}；支撑采样 {fraction:.0%}；Fz {force:.0f} N；缺失 {missing}/55')
                self.sole_info.content = '\n\n'.join(lines) + (
                    '\n\n27 点估计脚底支撑，外圈仅提示附近边缘；非真实接触面积。'
                    '抬脚、脚掌倾斜或正常蹬离可影响采样，红色也仅为疑似。'
                    '训练仍使用原来的外圈临边惩罚口径。')
            for env_id, body_id, frame in self._frames:
                frame.position = self._positions[env_id, body_id]
                frame.wxyz = quaternions[env_id, body_id]
            for body_id, mesh in self._batched_bodies:
                mesh.batched_positions = self._positions[:, body_id]
                mesh.batched_wxyzs = quaternions[:, body_id]
            for i, label in enumerate(self._robot_labels):
                label.position = self._positions[i, 0] + np.array([0., 0., 1.05])
            target = self._positions[self._selection_ids[self.selected.value], 0].copy()
            if self.play_status is not None:
                selected = self._selection_ids[self.selected.value]
                level = int(_numpy(self.env.scene.terrain.terrain_levels)[selected])
                seconds = float(_numpy(self.env.episode_length_buf)[selected]) * self.env.step_dt
                total, timeouts = self.play_episodes[selected], self.play_timeouts[selected]
                self.play_status.content = (
                    f"站姿初始化重放 · 地形行 {level}/3 · 本回合 {seconds:.1f}/10 秒\n\n"
                    f"已结束 {total} 回合；时限结束 {timeouts}；提前终止 {total-timeouts}。\n\n"
                    "时限结束不等于通过障碍；请同时观察红色目标与距离。")
            if self.velocity_arrows is not None:
                selected = self._selection_ids[self.selected.value]
                command = _numpy(self.env.command_manager.get_command("base_velocity"))[selected]
                actual = _numpy(self.robot.data.root_lin_vel_w)[selected]
                x, y, z, w = _numpy(self.robot.data.root_quat_w)[selected]
                # Commands are body-heading-relative; display both vectors in
                # world XY so a turning robot does not get misleading arrows.
                yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
                c, s = np.cos(yaw), np.sin(yaw)
                desired = np.array([c*command[0] - s*command[1], s*command[0] + c*command[1]])
                for i, (handle, velocity) in enumerate(zip(self.velocity_arrows, (desired, actual[:2]))):
                    vector = np.array([velocity[0], velocity[1], 0.]) * .6
                    length = np.linalg.norm(vector)
                    handle.visible = bool(self.show_velocity.value and np.isfinite(length) and length > .006)
                    if handle.visible:
                        direction = vector / length
                        side = np.array([-direction[1], direction[0], 0.])
                        head = min(.12, length * .35)
                        handle.points = np.array([[np.zeros(3), vector],
                            [vector, vector - head*direction + .5*head*side],
                            [vector, vector - head*direction - .5*head*side]], dtype=np.float32)
                        handle.position = target + np.array([0., 0., .85 + i*.12])
                self.velocity_info.content = (
                    "绿色：目标；橙色：实际。箭头长度 0.6 米代表 1 m/s。\n\n"
                    f"目标：前向 {command[0]:+.2f}，侧向 {command[1]:+.2f} m/s；"
                    f"转向 {command[2]:+.2f} rad/s。\n\n"
                    f"实际平移速度：{np.linalg.norm(actual[:2]):.2f} m/s。")
                if self.target_marker is not None:
                    command_term = self.env.command_manager.get_term("base_velocity")
                    goal = _numpy(command_term.pos_command_w)[selected]
                    self.target_marker.position = goal + np.array([0., 0., .06])
                    self.target_label.position = goal + np.array([0., 0., .3])
                    if self.target_control is not None:
                        manual = bool(_numpy(command_term.manual_target)[selected])
                        self.target_label.text = f"机器人 {selected} · {'手动' if manual else '随机'}目标"
                        self.target_feedback.content = (
                            self.target_control.message + '\n\n当前模式：'
                            + ('手动目标' if manual else '自动随机目标'))
                    self.target_marker.visible = self.target_label.visible = self.show_target.value
                    distance = np.linalg.norm(goal[:2] - target[:2])
                    standing = bool(_numpy(command_term.is_standing_env)[selected])
                    status = "站立指令" if standing else (
                        "已到达目标" if distance <= command_term.cfg.target_dis_threshold else "前往目标")
                    self.velocity_info.content += f"\n\n{status}；距行进目标：{distance:.2f} 米；红圈为到达范围。"
            for client in self.server.get_clients().values():
                previous = self._camera_targets.get(client.client_id, target)
                if self.follow.value:
                    delta = target - previous
                    client.camera.position = np.asarray(client.camera.position) + delta
                    client.camera.look_at = np.asarray(client.camera.look_at) + delta
                self._camera_targets[client.client_id] = target
            if observations is not None and "depth" in observations:
                depth = _numpy(observations["depth"])[self._selection_ids[self.selected.value], -1]
                pixels = enlarged_depth(depth)
                if self.depth_preview is None:
                    self.depth_preview = self.server.gui.add_image(pixels, label="策略深度输入 · 放大显示", format="png", order=-27)
                else:
                    self.depth_preview.image = pixels
                if self.depth_camera is not None:
                    selected = self._selection_ids[self.selected.value]
                    camera = self.depth_camera.data
                    height, width = camera.image_shape
                    intrinsic = _numpy(camera.intrinsic_matrices)[selected]
                    position = _numpy(camera.pos_w)[selected]
                    rotation = _numpy(camera.quat_w_ros)[selected][[3,0,1,2]]
                    full = (0, 0, width, height)
                    # Match preprocess_instinct_depth: lower half, central 32 columns.
                    from legged_lab.tasks.locomotion.amp.mdp.instinct_depth import InstinctDepthHistory
                    is_cropped = self.env.cfg.observations.depth.image.func is InstinctDepthHistory
                    crop = (16, 18, width-16, height) if is_cropped else full
                    distance = float(self.depth_camera.cfg.max_distance)
                    for handle, bounds, shown in ((self.policy_view,crop,self.show_policy_view.value),
                                                   (self.camera_view,full,self.show_camera_view.value)):
                        handle.visible = shown
                        if shown:
                            handle.points = view_boundary(intrinsic, bounds, distance)
                            handle.position = position
                            handle.wxyz = rotation
                    self.vision_info.content = (
                        f'绿色：策略区域 {depth.shape[1]}×{depth.shape[0]}；蓝色：相机 {width}×{height}。'
                        f'沿光轴深度上限 {distance:g} m；遮挡后的区域不可见。\n\n'
                        '下图是策略最近一帧输入，保留原始像素放大；黑近白远。移到图片右下角可展开。'
                        '策略帧可能延迟 0–1 个控制步，绿色边界显示当前相机位姿。')

    def record_step(self, dones):
        done = _numpy(dones).astype(bool).reshape(-1)
        timeout = _numpy(self.env.termination_manager.time_outs).astype(bool).reshape(-1)
        self.play_episodes += done
        self.play_timeouts += done & timeout
        if self.matrix_status is not None:
            failed = _numpy(self.env.termination_manager.terminated).astype(bool).reshape(-1)
            crossed = _numpy(self.env.termination_manager.get_term('stair_crossing')).astype(bool).reshape(-1)
            self.play_failures += done & failed
            self.play_crossings += done & crossed & ~failed

    def process_inputs(self):
        """Apply browser input on the simulation thread, including while paused."""
        if self.control_requests is not None and not self.control_requests.empty():
            from queue import Empty
            request = None
            while True:
                try:
                    request = self.control_requests.get_nowait()
                except Empty:
                    break
            command = self.env.command_manager.get_term('base_velocity')
            command.select_terrain(*request)
            if self.target_control is not None:
                self.target_control.owner = None
                while not self.target_control.events.empty():
                    self.target_control.events.get_nowait()
            self.env.reset()
            self.reset_observations = True
            for counts in (self.play_episodes, self.play_timeouts, self.play_crossings, self.play_failures):
                counts[:] = 0
            self.control_info.content = '当前：' + command.control_label()
            self._positions = _numpy(self.robot.data.body_pos_w)
            for client in self.server.get_clients().values():
                self._focus(client)
            print('[TERRAIN CONTROL] ' + command.control_label(), flush=True)
            return True
        if self.target_control is not None:
            return self.target_control.apply(self._selection_ids[self.selected.value], self.click_targets.value)
        return False

    def close(self):
        self.server.stop()
