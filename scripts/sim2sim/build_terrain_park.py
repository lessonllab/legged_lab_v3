"""Build separated, walkable terrain lanes with clear flat approaches."""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from run_mujoco import ROOT, TRAINING_SCENE

out=ROOT/'scripts/sim2sim/scenes/terrain_park.xml'
root=ET.parse(TRAINING_SCENE).getroot();root.set('model','G1 separated terrain park')
world=root.find('worldbody')
for g in list(world.findall('geom')):world.remove(g)
ET.SubElement(world,'geom',name='park_floor',type='plane',size='60 50 .1',material='groundplane')
lanes=[]
def box(name,x,y,z,sx,sy,sz,color='.45 .55 .65 1',**extra):
 ET.SubElement(world,'geom',name=name,type='box',pos=f'{x} {y} {z}',size=f'{sx} {sy} {sz}',rgba=color,**extra)
for i,h in enumerate([.08,.12,.16,.20,.23]):
 y=(i-2)*10.;x=4.;w=.3;n=6;top=n*h;plateau=2.
 color=['.3 .7 .5 1','.3 .6 .8 1','.7 .6 .3 1','.85 .4 .25 1','.7 .3 .55 1'][i]
 for j in range(n):
  z=(j+1)*h
  box(f'stairs_{h*100:g}_up_{j}',x+(j+.5)*w,y,z/2,w/2,1.5,z/2,color)
 box(f'stairs_{h*100:g}_platform',x+n*w+plateau/2,y,top/2,plateau/2,1.5,top/2,color)
 for j in range(n):
  z=(n-j)*h
  box(f'stairs_{h*100:g}_down_{j}',x+n*w+plateau+(j+.5)*w,y,z/2,w/2,1.5,z/2,color)
 lanes.append(dict(name=f'{h*100:g} cm stairs',spawn=[0,y,.8],target=[x+n*w+plateau/2,y],height=h,tread=w))
# Smooth triangular hills: each rotated box top begins at ground level.
for angle,y in [(8,-20),(15,-10)]:
 run=4.;theta=math.radians(angle);height=run*math.tan(theta);thick=.15
 for side in [-1,1]:
  # Left half rises +X; right half descends +X. Offset centre below the top surface.
  pitch=side*theta;mid=22. if side==-1 else 26.
  box(f'slope_{angle}_{side}',mid-math.sin(pitch)*thick,y,height/2-math.cos(pitch)*thick,run/(2*math.cos(theta)),1.5,thick,'.55 .65 .35 1',quat=f'{math.cos(pitch/2)} 0 {math.sin(pitch/2)} 0')
 lanes.append(dict(name=f'{angle} degree hill',spawn=[16,y,.8],target=[30,y]))
rng=np.random.default_rng(42)
for ix in range(12):
 for iy in range(6):
  h=float(rng.uniform(.015,.065))
  box(f'rough_{ix}_{iy}',20+(ix+.5)*.4,(iy-2.5)*.5,h/2,.2,.25,h/2,'.6 .52 .42 1')
lanes.append(dict(name='uneven ground 1.5-6.5 cm',spawn=[16,0,.8],target=[27,0]))
for i in range(7):
 box(f'low_hurdle_{i}',20+i*.65,10,.02,.075,1.5,.02,'.7 .6 .25 1')
lanes.append(dict(name='4 cm low hurdles',spawn=[16,10,.8],target=[26,10]))
for i in range(8):
 box(f'stone_{i}',20+i*.65,20+(.18 if i%2 else -.18),.04,.25,.55,.04,'.55 .58 .65 1')
lanes.append(dict(name='8 cm stepping blocks',spawn=[16,20,.8],target=[26,20]))
ET.indent(root);ET.ElementTree(root).write(out,encoding='unicode')
out.with_suffix('.json').write_text(json.dumps({'lane_center_spacing':10,'lane_clear_gap':7,'lanes':lanes},indent=2)+'\n')
print(out)
