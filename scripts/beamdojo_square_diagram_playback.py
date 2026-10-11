"""Playback-only four-route maps matching the approved square layout sketch.

One policy, spawn yaw faces the selected route; commands are body-forward.
No training rewards or training terrain are changed. Geometry checks here only
certify construction/raster consistency, not dynamic reachability or success.
"""
import math
import numpy as np
from beamdojo_square_course_geometry import layout, rasterize

FAMILIES = {'正向：前后左右': (0,90,180,-90),
            '斜向：四个对角': (45,135,-135,-45)}


def rotation(degrees):
    angle=math.radians(degrees);c,s=math.cos(angle),math.sin(angle)
    return np.array([[c,-s],[s,c]])


def diagram_layout(seed=0,level=1.,family='cardinal'):
    base=layout(seed,level)
    polygons=[np.asarray(p) for p in base['polygons_m'][:5]]
    angles=(0,90,180,-90) if family=='cardinal' else (45,135,-135,-45)
    if family not in ('cardinal','diagonal'):raise ValueError(family)
    # Opposite routes start in opposite lateral columns: exact reflection in
    # the cardinal map; opposite diagonal pairs are reflected about Y=-X/X=Y.
    first_columns=(0,0,1,1)
    if family=='diagonal':
        for a in angles:
            triangle=np.array([[2.5,3.],[3.,2.5],[3.,3.]])
            polygons.append(triangle @ rotation(a-45).T)
        start_s=.7*math.sqrt(2)
        goal_s=5.5/math.sqrt(2)
    else:
        start_s=.7;goal_s=3.
    first=start_s+.06
    last=goal_s-.06-.28
    intervals=int(math.floor(2*(last-first)/(.28+.07+.13*level)+.5))
    phase=(last-first)/intervals
    routes=[];stone_polygons=[]
    for angle,first_column in zip(angles,first_columns):
        r=rotation(angle);ids=[]
        for i in range(intervals+1):
            col=(i+first_column)%2
            t=.2 if col==0 else -.2
            s=first+i*phase
            p=np.array([[s,t-.14],[s+.28,t-.14],[s+.28,t+.14],[s,t+.14]])@r.T
            ids.append(len(polygons));polygons.append(p);stone_polygons.append(p)
        routes.append(dict(angle_degrees=angle,first_column='left' if first_column==0 else 'right',
                           stone_polygon_indices=ids,goal_start_s_m=goal_s))
    boxes=[[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())] for p in polygons]
    base.update(schema='square_diagram_playback_v1',family=family,
        rectangles_m=boxes,support_polygons_m=[p.tolist() for p in polygons],
        polygons_m=[p.tolist() for p in polygons],routes=routes,
        stone_rectangles_m=[[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())] for p in stone_polygons],
        witness=None,checks={'construction_only':True,'opposite_first_columns_mirrored':True})
    base['metadata'].update(branch_count=4,body_yaw=0.,
        actual_same_column_gap_m=2*phase-.28,longitudinal_phase_m=phase,
        requested_same_column_gap_m=.07+.13*level,
        corner_landing_m=.5 if family=='diagonal' else 0.,
        scope='Playback geometry only; no policy/dynamics guarantee',
        terminal='Both feet on selected route outer ring/corner landing, stable 1s')
    # Base +X witness belongs only to the old single-route map, never advertise
    # its feet/support claims as validation of these new directions.
    for key in ('actual_same_column_gaps_m','exit_edge_gaps_m','other_column_entry_gap_m','command_body_xy','ankle_lateral_separation_m'):
        base['metadata'].pop(key,None)
    for key in ('stone_feet','initial_feet_xy_m','first_entry_direction'):
        base.pop(key,None)
    return base


def diagram_mesh(difficulty,cfg):
    """Native terrain callback: the 22 atlas rows contain A/B × 11 levels."""
    import trimesh
    index=min(21,int(difficulty*22))
    family='cardinal' if index<11 else 'diagonal'
    grid=rasterize(diagram_layout(cfg.layout_seed,(index%11)/10,family))['collision_height_m'].T
    meshes=[];floor=trimesh.creation.box(extents=(8,8,.1))
    floor.apply_translation((4,4,-1.05));meshes.append(floor)
    active={};spans=[]
    for x in range(grid.shape[0]+1):
        runs=[]
        if x<grid.shape[0]:
            edges=np.flatnonzero(np.diff(np.r_[False,grid[x]==0,False].astype(np.int8)))
            runs=list(zip(edges[::2],edges[1::2]))
        for run in list(active):
            if run not in runs:spans.append((active.pop(run),x,*run))
        for run in runs:active.setdefault(run,x)
    for x0,x1,y0,y1 in spans:
        box=trimesh.creation.box(extents=((x1-x0)*.02,(y1-y0)*.02,1.))
        box.apply_translation(((x0+x1)*.01,(y0+y1)*.01,-.5));meshes.append(box)
    return meshes,np.array([4.,4.,0.])


def set_route(cfg,angle):
    diagonal=abs(angle)%90==45
    goal_s=5.5/math.sqrt(2) if diagonal else 3.
    start_s=.7*math.sqrt(2) if diagonal else .7
    cfg.beamdojo_play_route_degrees=float(angle)
    yaw=math.radians(angle)
    cfg.events.reset_base.params['pose_range']['yaw']=(yaw,yaw)
    cfg.beamdojo_square_course_params.update(start_platform_m=[-start_s,-.7,start_s,.7],
        end_platform_m=[goal_s,-.6,5.6 if diagonal else 4.,.6],
        target_root_x=goal_s+.25,braking_start_x=goal_s+.05,
        boundary_limit=5.6 if diagonal else 3.9)



def set_play_route(env,angle):
    """Recreate Sim6's reset functor: it caches ranges at construction."""
    set_route(env.cfg,angle)
    manager=env.event_manager
    term=manager.get_term_cfg('reset_base')
    term.params['pose_range']['yaw']=env.cfg.events.reset_base.params['pose_range']['yaw']
    term.func=type(term.func)(term,env)
    manager.set_term_cfg('reset_base',term)


def configure_diagram(cfg,initial_level=10,initial_angle=0):
    import torch
    from beamdojo_square_course_profile import install_geometry
    import beamdojo_square_course_runtime as runtime
    courses=[diagram_layout(cfg.seed,i/10,family) for family in ('cardinal','diagonal') for i in range(11)]
    cfg.beamdojo_initial_level=initial_level
    cfg.scene.terrain.max_init_terrain_level=initial_level
    generator=cfg.scene.terrain.terrain_generator
    generator.num_rows=22;generator.num_cols=2
    generator.sub_terrains['stepping_stones'].function=diagram_mesh
    install_geometry(cfg,courses=courses)
    original_read=runtime.read_measurements
    def route_measurements(env):
        m=original_read(env)
        r=torch.as_tensor(rotation(env.cfg.beamdojo_play_route_degrees),device=env.device,dtype=m['root'].dtype)
        for key in ('root','corners','lin'):
            value=m[key].clone();value[...,:2]=value[...,:2]@r;m[key]=value
        return m
    runtime.read_measurements=route_measurements
    base=cfg.commands.base_velocity.class_type
    class DiagramCommand(base):
        def _resample_command(self,env_ids):
            super()._resample_command(env_ids)
            self._update_command()
        def _update_command(self):
            super()._update_command()
            # Native root reset points the body at the selected route.
            # Diagonal routes also receive (forward, 0, 0).
    cfg.commands.base_velocity.class_type=DiagramCommand
    set_route(cfg,initial_angle)
