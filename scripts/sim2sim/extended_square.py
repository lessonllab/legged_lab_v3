"""Playback transfer maps; archived training maps remain unchanged."""
import math
import numpy as np
import trimesh
from beamdojo_square_course_geometry import support_mask
from beamdojo_square_diagram_playback import rotation

def extended_layout(seed,level,family,variant,size=12.,ring_width=2.):
 half=size/2;inner=half-ring_width
 if inner<=1.5 or ring_width<1.4:raise ValueError('extended map requires a longer crossing and a safe parking platform')
 rects=[[-.7,-.7,.7,.7],[inner,-inner,half,inner],[-half,-inner,-inner,inner],[-half,inner,half,half],[-half,-half,half,-inner]]
 polygons=[np.array([[a,b],[c,b],[c,d],[a,d]]) for a,b,c,d in rects]
 angles=(0,90,180,-90) if family=='cardinal' else (45,135,-135,-45)
 if family=='diagonal':
  for a in angles:polygons.append(np.array([[inner-.5,inner],[inner,inner-.5],[inner,inner]])@rotation(a-45).T)
 start=.7 if family=='cardinal' else .7*math.sqrt(2);goal=inner if family=='cardinal' else (2*inner-.5)/math.sqrt(2)
 first=start+.06;last=goal-.34;intervals=int(math.floor(2*(last-first)/(.35+.13*level)+.5));phase=(last-first)/intervals
 rng=np.random.default_rng(seed);profiles={0:(.28,.28,0.,0.),1:(.26,.30,.01,0.),2:(.25,.31,.02,0.),3:(.24,.32,.03,.015)};lo,hi,jx,jy=profiles[variant]
 routes=[]
 for j,angle in enumerate(angles):
  ids=[];rot=rotation(angle)
  for i in range(intervals+1):
   column=(i+(0,0,1,1)[j])%2;t=.2 if column==0 else -.2
   center=np.array([first+i*phase+.14,t]);center+=rng.uniform([-jx,-jy],[jx,jy]);dims=rng.uniform(lo,hi,2);a,b=center-dims/2;c,d=center+dims/2
   p=np.array([[a,b],[c,b],[c,d],[a,d]])@rot.T;ids.append(len(polygons));polygons.append(p)
  routes.append(dict(angle_degrees=angle,stone_polygon_indices=ids,goal_start_s_m=goal))
 return dict(schema='mujoco_extended_playback_v1',family=family,variant=variant,routes=routes,polygons_m=[p.tolist() for p in polygons],support_polygons_m=[p.tolist() for p in polygons],map_size_m=size,ring_width_m=ring_width,metadata={'scope':'new playback transfer map; not the archived training map','phase_m':phase,'same_column_nominal_gap_m':2*phase-.28,'seed':seed})

def raster_mesh(course,size):
 n=int(round(size/.02));axis=-size/2+(np.arange(n)+.5)*.02;xx,yy=np.meshgrid(axis,axis,indexing='xy');mask=support_mask(course,np.stack([xx,yy],-1))
 floor=trimesh.creation.box(extents=(size,size,.1));floor.apply_translation((size/2,size/2,-1.05));meshes=[floor];active={};spans=[]
 for x in range(n+1):
  runs=[]
  if x<n:
   edge=np.flatnonzero(np.diff(np.r_[False,mask[:,x],False].astype(np.int8)));runs=list(zip(edge[::2],edge[1::2]))
  for run in list(active):
   if run not in runs:spans.append((active.pop(run),x,*run))
  for run in runs:active.setdefault(run,x)
 for x0,x1,y0,y1 in spans:
  box=trimesh.creation.box(extents=((x1-x0)*.02,(y1-y0)*.02,1.));box.apply_translation(((x0+x1)*.01,(y0+y1)*.01,-.5));meshes.append(box)
 return mask,meshes,np.array([size/2,size/2,0.])
