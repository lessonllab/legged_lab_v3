"""Record native MuJoCo physics, with the same front45 following viewpoint."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,math
from PIL import Image,ImageDraw,ImageFont
import numpy as np,torch,mujoco,imageio.v2 as imageio
from beamdojo_mujoco import Runner,ROOT

def main():
 torch.set_num_threads(1);r=Runner();out=ROOT/'artifacts/sim2sim_mujoco_8999';out.mkdir(parents=True,exist_ok=True)
 r.model.vis.global_.offwidth=1280;r.model.vis.global_.offheight=720
 renderer=mujoco.Renderer(r.model,height=720,width=1280);cam=mujoco.MjvCamera();cam.distance=4.;cam.azimuth=25;cam.elevation=-45
 opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,0,1,0,0]
 with imageio.get_writer(out/'L10_cardinal_v2.mp4',fps=25) as writer:
  for i in range(650):
   if r.status=='running':r.step()
   if i%2==0:
    cam.lookat[:]=r.data.qpos[:3];cam.lookat[2]=.5
    renderer.update_scene(r.data,camera=cam,scene_option=opt);frame=renderer.render()
    image=Image.fromarray(frame);draw=ImageDraw.Draw(image);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    draw.rectangle((0,0,1280,82),fill=(20,25,32))
    draw.text((20,10),'Native MuJoCo sim2sim | Frozen model_8999 | L10 cardinal | variant 2',font=font,fill='white')
    draw.text((20,43),f'Cruise command: 1.00 m/s | Simulation: {r.data.time:.2f}s | Progress: {r.max_progress:.2f}m | {r.status}',font=font,fill=(175,220,255))
    frame=np.asarray(image);writer.append_data(frame)
    if i in (0,150,300,450):imageio.imwrite(out/f'frame_{i:03d}.png',frame)
   if r.status!='running' and i>520:break
 renderer.close();print(json.dumps({'status':r.status,'seconds':r.data.time,'progress':r.max_progress}))
if __name__=='__main__':main()
