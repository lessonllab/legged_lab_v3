"""Shared playback visuals; importing this module never launches simulation."""
import torch


def show_perception_samples(viewer, env):
    """Visualize the actor's held elevation sample, without querying new terrain.

    Capture sampling pose only on publication (age=0). Reinterpreting held
    heights at the moving robot's newer pose would misplace the perceived gaps.
    Points are projected above the physical floor so virtual voids stay visible.
    This is elevation-map input, not a raw LiDAR scan or an assumed mount pose.
    """
    import numpy as np
    from legged_lab.tasks.locomotion.beamdojo.perception import grid_points_world
    from legged_lab.tasks.locomotion.beamdojo.upstream_perception import scanner_spec, sampling_points_world
    upstream = (getattr(env.cfg, 'beamdojo_upstream_aligned', False)
                or getattr(env.cfg, 'beamdojo_unitree_motion_aligned', False))
    spec = scanner_spec(env.cfg) if upstream else {'shape_xy': [15,15], 'resolution_m': .1, 'count': 225, 'ordering': 'ij', 'size_xy_m': [1.4,1.4]}
    resolution = spec['resolution_m']
    nx, ny = spec['shape_xy']
    count = spec['count']
    sx, sy = spec['size_xy_m']
    corner_ids = ([0, nx-1, count-1, (ny-1)*nx, 0] if spec['ordering'] == 'xy'
                  else [0, ny-1, count-1, (nx-1)*ny, 0])
    rate = round(1/env.step_dt) if upstream else 10
    cloud = viewer.server.scene.add_point_cloud(
        '/perception/samples', points=np.zeros((0,3),dtype=np.float32),
        colors=np.zeros((0,3),dtype=np.uint8), point_size=.035, point_shape='circle')
    outline = viewer.server.scene.add_line_segments(
        '/perception/coverage', points=np.zeros((0,2,3),dtype=np.float32),
        colors=(245,190,35), line_width=3.)
    control = viewer.server.gui.add_checkbox('显示地形感知采样点', initial_value=True)
    @control.on_update
    def toggle(_):
        cloud.visible = outline.visible = control.value
    viewer.server.gui.add_markdown(
        f'**感知输入：{nx}×{ny} 高程图，范围 {sx:g}×{sy:g} m，间距 {resolution*100:g} cm，更新 {rate} Hz。**\n\n'
        '黄点：感知到的桩面；红点：间隙；橙色框：采样范围。\n\n'
        '显示的是策略实际接收的模拟高程图采样，点的位置投影到地面。')
    info = viewer.server.gui.add_markdown('等待感知采样。')
    state = {'publication_tick':None}
    def update(observations, force=False):
        age = 0. if upstream else float(env._beamdojo_map.age[0])
        tick = env.common_step_counter
        if force or (age < 1e-6 and state['publication_tick'] != tick):
            data = env.scene['robot'].data
            pos = data.root_pos_w.torch if hasattr(data.root_pos_w,'torch') else data.root_pos_w
            quat = data.root_quat_w.torch if hasattr(data.root_quat_w,'torch') else data.root_quat_w
            yaw = torch.atan2(2*(quat[:,3]*quat[:,2]+quat[:,0]*quat[:,1]),
                              1-2*(quat[:,1].square()+quat[:,2].square()))
            if env.cfg.beamdojo_map_noise and not upstream:
                yaw = yaw + env._beamdojo_map_noise.yaw
            points = (sampling_points_world(env)[0].clone() if upstream else
                      grid_points_world(pos,yaw,size=15,resolution=resolution)[0].clone())
            observed = observations['terrain'][0]
            heights = pos[0,2]-.5-observed[:count]/5 if upstream else pos[0,2]-.78-observed[:225]
            valid = torch.ones(count,device=env.device,dtype=torch.bool) if upstream else observed[225:450]>.5
            origin = env.scene.env_origins[0]
            origin = origin.torch if hasattr(origin,'torch') else origin
            unsafe = heights < origin[2]-.1
            colors = np.tile(np.array([250,210,40],dtype=np.uint8),(count,1))
            colors[unsafe.detach().cpu().numpy()] = (240,65,65)
            colors[(~valid).detach().cpu().numpy()] = (125,125,125)
            points[:,2] = origin[2]+.022
            projected = points.detach().cpu().numpy()
            cloud.points, cloud.colors = projected, colors
            corners = projected[corner_ids].copy()
            corners[:,2] += .003
            outline.points = np.stack((corners[:-1],corners[1:]),axis=1)
            state['publication_tick'] = tick
        info.content = f'采样点 {count} 个 · 当前帧龄 {age*1000:.0f} ms · 模拟高程图'
    return update

def active_terrain_bounds(env):
    """World bounds of the currently selected course, including its platforms."""
    geometry = env.beamdojo_geometry[0,env.beamdojo_rectangle_mask[0]]
    origin = env.scene.env_origins[0]
    origin = origin.torch if hasattr(origin,'torch') else origin
    xmin,ymin = (geometry[:,:2]-geometry[:,2:4]/2).amin(0)+origin[:2]
    xmax,ymax = (geometry[:,:2]+geometry[:,2:4]/2).amax(0)+origin[:2]
    return (float(xmin)-1e-4,float(xmax)+1e-4,float(ymin)-1e-4,float(ymax)+1e-4)
