"""Record native MuJoCo physics, with the same front45 following viewpoint."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,math,argparse
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np,torch,mujoco,imageio.v2 as imageio
from beamdojo_mujoco import Runner,ROOT

def main():
 p=argparse.ArgumentParser();p.add_argument('--map-size',type=float,default=8.);p.add_argument('--ring-width',type=float,default=1.);p.add_argument('--stop-depth',type=float,default=.2);p.add_argument('--show-scan',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'artifacts/sim2sim_mujoco_8999/L10_cardinal_v2.mp4');a=p.parse_args()
 torch.set_num_threads(1);r=Runner(map_size=a.map_size,ring_width=a.ring_width,stop_depth=a.stop_depth);out=a.output.parent;out.mkdir(parents=True,exist_ok=True)
 r.model.vis.global_.offwidth=1280;r.model.vis.global_.offheight=720
 renderer=mujoco.Renderer(r.model,height=720,width=1280);cam=mujoco.MjvCamera();cam.distance=4.;cam.azimuth=25;cam.elevation=-45
 opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,0,1,0,0]
 with imageio.get_writer(a.output,fps=25) as writer:
  for i in range(650):
   if r.status=='running':r.step()
   if i%2==0:
    cam.lookat[:]=r.data.qpos[:3];cam.lookat[2]=.5
    renderer.update_scene(r.data,camera=cam,scene_option=opt)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
    if a.show_scan:
     scan=r.last_scan if r.last_scan is not None else r.scan_world()
     for point in scan:
      if renderer.scene.ngeom>=renderer.scene.maxgeom:break
      geom=renderer.scene.geoms[renderer.scene.ngeom];color=[.15,.85,.4,1.] if point[2]>=0 else [1.,.2,.15,1.]
      mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.018]*3),point+np.array([0,0,.025]),np.eye(3).ravel(),np.array(color))
      renderer.scene.ngeom+=1
    frame=renderer.render()
    image=Image.fromarray(frame);draw=ImageDraw.Draw(image);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    draw.rectangle((0,0,1280,82),fill=(20,25,32))
    draw.text((20,10),f'Native MuJoCo | model_8999 | L10 | {a.map_size:g}m map | green: support / red: gap',font=font,fill='white')
    draw.text((20,43),f'Cruise command: 1.00 m/s | Simulation: {r.data.time:.2f}s | Progress: {r.max_progress:.2f}m | {r.status}',font=font,fill=(175,220,255))
    frame=np.asarray(image);writer.append_data(frame)
    if i in (0,150,300,450):imageio.imwrite(out/f'frame_{i:03d}.png',frame)
   if r.status!='running' and i>520:break
 renderer.close();report={'status':r.status,'seconds':r.data.time,'progress':r.max_progress,'map_size_m':a.map_size,'ring_width_m':a.ring_width,'stop_depth_m':a.stop_depth,'parking_trigger_depth_m':getattr(r,'parking_depth',None),'root_position':r.data.qpos[:3].tolist(),'level':10,'family':'cardinal','variant':2,'seed':42,'model_iteration':8999};a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
