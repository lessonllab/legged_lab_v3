"""Paper VI-B terrain geometry, plus explicit G1 route-length adaptation.

Levels 0..8 use the published sizes/gap schedules and +/-5 cm stone heights.
Stage 2 covers 2/4/6 total stones across successive groups of three levels;
this count schedule and 2 m center/start/end platforms are G1 adaptations.
Every mesh and ideal/safety query consumes these same deterministic layouts.
"""
import math
import random

STEPPING_SIZES = (.8,.65,.5,.4,.35,.3,.25,.2,.2)


def layout(stage, level, seed=42):
    if stage not in (1,2) or not 0 <= level <= 8:
        raise ValueError('Expected stage 1/2 and level 0..8')
    rng = random.Random(f'beamdojo-paper-v2:{stage}:{level}:{seed}')
    if stage == 1:
        side = max(.25,1.5*(1-.1*level))
        gap = .05*math.ceil(level/2)
        pitch = side+gap
        n = math.ceil(8/pitch)
        rectangles = [(0.,0.,2.,2.,0.)]
        for ix in range(n):
            for iy in range(n):
                cx = -4+(ix+.5)*pitch+rng.uniform(-gap/2,gap/2)
                cy = -4+(iy+.5)*pitch+rng.uniform(-gap/2,gap/2)
                lo_x,hi_x=max(-4,cx-side/2),min(4,cx+side/2)
                lo_y,hi_y=max(-4,cy-side/2),min(4,cy+side/2)
                rectangles.append(((lo_x+hi_x)/2,(lo_y+hi_y)/2,max(0.,hi_x-lo_x),
                                   max(0.,hi_y-lo_y),rng.uniform(-.05,.05)))
        return {'rectangles':rectangles,'grid_pitch':pitch,'grid_n':n,'num_stones':n*n,
                'stone_indices':list(range(1,len(rectangles))),'start_index':0,'end_index':0,
                'bounds':(-4.,4.,-4.,4.),'terrain_size':(8.,8.),'mesh_origin':(4.,4.,0.)}
    side=STEPPING_SIZES[level]
    count=(2,4,6)[level//3]
    rectangles=[(0.,0.,2.,2.,0.)]
    edge=1.
    for pair in range(count//2):
        gap=rng.uniform(.03,.1+.05*level)
        cx=edge+gap+side/2
        for sign in (1.,-1.):
            rectangles.append((cx+rng.uniform(-.01,.01),sign*(side/2+.05),side,side,rng.uniform(-.05,.05)))
        edge=max(r[0]+r[2]/2 for r in rectangles[-2:])
    rectangles.append((edge+.03+1.,0.,2.,2.,0.))
    return {'rectangles':rectangles,'num_stones':count,'stone_indices':list(range(1,count+1)),
            'start_index':0,'end_index':len(rectangles)-1,'bounds':(-1.,7.,-1.,1.),
            'terrain_size':(8.,2.),'mesh_origin':(1.,1.,0.)}


def mesh_terrain(difficulty,cfg):
    import numpy as np
    import trimesh
    level=min(8,max(0,int(difficulty*9)))
    spec=layout(cfg.beamdojo_stage,level,cfg.layout_seed)
    origin=np.asarray(spec['mesh_origin'],dtype=float)
    if cfg.beamdojo_stage==1:
        box=trimesh.creation.box(extents=(*spec['terrain_size'],.1))
        box.apply_translation((4.,4.,-.05))
        return [box],origin
    meshes=[]
    for x,y,sx,sy,top in spec['rectangles']:
        box=trimesh.creation.box(extents=(sx,sy,1.+top))
        box.apply_translation((x+origin[0],y+origin[1],(top-1.)/2))
        meshes.append(box)
    return meshes,origin


def geometry_tables(stage,seed,device):
    import torch
    layouts=[layout(stage,l,seed) for l in range(9)]
    width=max(len(spec['rectangles']) for spec in layouts)
    rectangles=torch.zeros(9,width,5,device=device)
    valid=torch.zeros(9,width,dtype=torch.bool,device=device)
    stones=torch.zeros_like(valid)
    for level,spec in enumerate(layouts):
        n=len(spec['rectangles'])
        rectangles[level,:n]=torch.tensor(spec['rectangles'],device=device)
        valid[level,:n]=True
        stones[level,spec['stone_indices']]=True
    return rectangles,valid,stones,layouts


def sample_surface(points,origins,levels,tables,stage):
    """O(points) grid lookup for Stage 1; small rectangle batches for Stage 2."""
    import torch
    rectangles,mask,_,layouts=tables
    n=points.shape[0]
    shape=points.shape[:-1]
    local=(points-origins.reshape((n,)+(1,)*(points.ndim-2)+(3,))).reshape(n,-1,3)
    geo=rectangles[levels]
    if stage==1:
        pitch=torch.tensor([v['grid_pitch'] for v in layouts],device=points.device)[levels]
        grid_n=torch.tensor([v['grid_n'] for v in layouts],device=points.device)[levels]
        index=torch.floor((local[...,:2]+4.)/pitch[:,None,None]).long()
        inside=(index>=0).all(-1)&(index<grid_n[:,None,None]).all(-1)
        flat=1+index[...,0]*grid_n[:,None]+index[...,1]
        flat=flat.clamp(0,geo.shape[1]-1)
        candidate=geo.gather(1,flat[...,None].expand(-1,-1,5))
        support=inside & ((local[...,:2]-candidate[...,:2]).abs()<=candidate[...,2:4]/2).all(-1)
        support &= (local[...,:2].abs()<=4.).all(-1)
        top=torch.where(support,candidate[...,4],-1.)
        platform=(local[...,:2].abs()<=1.).all(-1)
        top=torch.where(platform,0.,top)
        support |= platform
    else:
        support=torch.zeros(local.shape[:2],dtype=torch.bool,device=points.device)
        top=torch.full(local.shape[:2],-1.,device=points.device)
        for i in range(geo.shape[1]):
            rectangle=geo[:,i]
            hit=((local[...,:2]-rectangle[:,None,:2]).abs()<=rectangle[:,None,2:4]/2).all(-1)
            hit &= mask[levels,i,None]
            top=torch.where(hit,rectangle[:,None,4],top)
            support |= hit
    return (top+origins[:,2:3]).reshape(shape),support.reshape(shape)
