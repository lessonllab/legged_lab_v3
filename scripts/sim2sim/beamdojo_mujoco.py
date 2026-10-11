"""Frozen BeamDojo policy in native MuJoCo; identical archived square raster."""
from pathlib import Path
import sys,json,math,time,argparse,os,xml.etree.ElementTree as ET
import numpy as np
import torch,mujoco
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from run_mujoco import calibrated_model,read_config,resolve,rotation
DEFAULT=ROOT/'logs/rsl_rl/g1_beamdojo_fullbody_sim6/2026-10-11_02-54-30-055773_square_mixed_generalize_5000_4096/model_8999.pt'

class Runner:
 def __init__(self,checkpoint=DEFAULT,level=10,family='cardinal',variant=2,angle=0,speed=1.,seed=42):
  self.checkpoint=Path(checkpoint); self.cfg=read_config(self.checkpoint.parent/'params/env.yaml')
  self.contract=json.loads((self.checkpoint.parent/'params/source_contract.json').read_text())
  self.names=self.contract['joint_names']; self.profile=json.loads((ROOT/'scripts/sim2sim/g1_training_profile.json').read_text())
  assert self.profile['sha256']==self.contract['asset_sha256']
  # Resolve terrain generator from the checkpoint archive, rather than live changes.
  sys.path.insert(0,str(self.checkpoint.parent/'params'))
  from beamdojo_square_mixed_geometry import mixed_layout,grid_mesh
  from beamdojo_square_course_geometry import rasterize
  self.course=mixed_layout(seed,level/10,family,variant)
  self.mask=rasterize(self.course)['reward_support_mask']; self.meshes,origin=grid_mesh(np.where(self.mask,0.,-1.))
  root=ET.Element('mujoco',model='G1 BeamDojo MuJoCo'); ET.SubElement(root,'include',file=str(ROOT/'hiking-in-the-wild-sim2sim/unitree_mujoco/unitree_robots/g1/g1_29dof.xml'))
  visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='1280',offheight='720')
  wb=ET.SubElement(root,'worldbody');ET.SubElement(wb,'light',pos='0 0 5',dir='0 0 -1',directional='true',diffuse='.8 .8 .8',ambient='.4 .4 .4')
  for i,mesh in enumerate(self.meshes):
   lo,hi=mesh.bounds-origin
   ET.SubElement(wb,'geom',name=f'terrain_{i}',type='box',pos=' '.join(map(str,(lo+hi)/2)),size=' '.join(map(str,(hi-lo)/2)),group='0',friction='1 0.005 0.0001',rgba='.12 .13 .15 1' if i==0 else '.42 .60 .72 1')
  scene=ROOT/f'.runtime/mujoco_square_scene_{os.getpid()}.xml';scene.write_text(ET.tostring(root,encoding='unicode'))
  self.model=calibrated_model(scene,self.profile);scene.unlink();self.data=mujoco.MjData(self.model)
  m=self.model;d=self.data
  for name,b in self.profile['bodies'].items():
   bid=m.body(name).id
   for field,key in [('body_mass','mass'),('body_ipos','ipos'),('body_iquat','iquat'),('body_inertia','inertia')]:getattr(m,field)[bid]=b[key]
  for name,j in self.profile['joints'].items():
   bid=m.jnt_bodyid[m.joint(name).id];m.body_pos[bid]=j['pos'];m.body_quat[bid]=j['quat']
  self.ids=np.array([m.joint(n).id for n in self.names]);self.qadr=m.jnt_qposadr[self.ids];self.vadr=m.jnt_dofadr[self.ids]
  self.actids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in self.ids])
  robot=self.cfg['scene']['robot'];self.default=np.array([resolve(robot['init_state']['joint_pos'],n,0.) for n in self.names])
  for n,j,v,a in zip(self.names,self.ids,self.vadr,self.actids):
   import re
   groups=[g for g in robot['actuators'].values() if any(re.fullmatch(p,n) for p in g['joint_names_expr'])];assert len(groups)==1
   g=groups[0];kp=resolve(g['stiffness'],n);kd=resolve(g['damping'],n);lim=resolve(g['effort_limit_sim'],n)
   m.actuator_gaintype[a]=mujoco.mjtGain.mjGAIN_FIXED;m.actuator_biastype[a]=mujoco.mjtBias.mjBIAS_AFFINE
   m.actuator_gainprm[a,:]=0.;m.actuator_gainprm[a,0]=kp;m.actuator_biasprm[a,:]=0.;m.actuator_biasprm[a,:3]=[0,-kp,-kd]
   m.actuator_ctrllimited[a]=False;m.actuator_forcelimited[a]=True;m.actuator_forcerange[a]=[-lim,lim]
   m.jnt_actfrclimited[j]=True;m.jnt_actfrcrange[j]=[-lim,lim]
   m.dof_armature[v]=resolve(g['armature'],n);m.dof_damping[v]=0;m.dof_frictionloss[v]=0
  m.opt.timestep=self.cfg['sim']['dt'];m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  self.decimation=self.cfg['decimation'];self.dt=m.opt.timestep*self.decimation
  mujoco.mj_setConst(m,d)
  saved=torch.load(self.checkpoint,map_location='cpu',weights_only=False);state=saved['actor_state_dict']
  self.actor=torch.nn.Sequential(torch.nn.Linear(708,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,29)).eval()
  self.actor.load_state_dict({k.removeprefix('mlp.'):v for k,v in state.items() if k.startswith('mlp.')},strict=True)
  b=saved['beamdojo']['action_mean_tail_bounds'];self.lower=torch.tensor(b['lower']);self.upper=torch.tensor(b['upper']);half=(self.upper-self.lower)/2
  self.il=self.lower-b['identity_margin_halfwidth']*half;self.iu=self.upper+b['identity_margin_halfwidth']*half;self.tw=b['tail_width_halfwidth']*half
  self.footids=[m.body(n).id for n in ['left_ankle_roll_link','right_ankle_roll_link']]
  axis=np.linspace(-.7,.7,15);xx,yy=np.meshgrid(axis,axis,indexing='ij');self.points=np.c_[xx.ravel(),yy.ravel()]
  self.angle=angle;self.speed=speed;self.reset()
 def heights(self,xy):
  ix=np.floor((np.asarray(xy)+4)/.02).astype(int);valid=((ix>=0)&(ix<400)).all(-1);safe=ix.clip(0,399)
  return np.where(valid & self.mask[safe[...,1],safe[...,0]],0.,-1.)
 def reset(self):
  mujoco.mj_resetData(self.model,self.data);d=self.data;a=math.radians(self.angle);self.direction=np.array([math.cos(a),math.sin(a)])
  d.qpos[:3]=[0,0,.8];d.qpos[3:7]=[math.cos(a/2),0,0,math.sin(a/2)];d.qpos[self.qadr]=self.default;d.ctrl[self.actids]=self.default
  mujoco.mj_forward(self.model,d);self.action=np.zeros(29);self.hist=None;self.imu=0.;self.active=True;self.parked=False;self.stable=0.;self.status='running';self.max_progress=0.
 def feet(self):
  # Trained ankle-frame envelope including 1cm margin.
  corners=np.array([[-.065,-.045,-.035],[-.065,.045,-.035],[.155,-.045,-.035],[.155,.045,-.035]])
  return np.array([self.data.xpos[b]+corners@self.data.xmat[b].reshape(3,3).T for b in self.footids])
 def command(self):
  s=self.data.qpos[:2]@self.direction;self.max_progress=max(self.max_progress,float(s))
  goal=next(r['goal_start_s_m'] for r in self.course['routes'] if abs(r['angle_degrees']-self.angle)<1e-5)
  feet=self.feet();local=feet[...,:2]@self.direction
  lateral=feet[...,:2]@np.array([-self.direction[1],self.direction[0]])
  full=bool((local>=goal).all() and (np.abs(lateral)<=.6).all() and (np.abs(feet[...,2])<=.035).all() and (np.abs(feet[...,:2])<4.).all() and (self.heights(feet[...,:2])==0).all())
  self.parked|=full and s>=goal+.20
  speed=0. if self.parked else max(.25,np.clip(self.speed,.5,1.)*np.clip((goal+.25-s)/.20,0,1))
  return np.array([speed,0.,0.]),full
 def observations(self,cmd):
  d=self.data;R=rotation(d.qpos[3:7]);g=R.T@np.array([0.,0.,-1.]);omega=d.qvel[3:6]
  active=abs(cmd[0])>=.1
  if active and self.active:self.imu+=-(omega[1]*g[1]+omega[2]*g[2])/max(.25,g[1]**2+g[2]**2)*self.dt
  else:self.imu=0.
  self.active=active
  terms=[omega*.2,g,cmd,d.qpos[self.qadr]-self.default,d.qvel[self.vadr]*.05,self.action]
  if self.hist is None:self.hist=[np.repeat(t[None],5,axis=0) for t in terms]
  else:
   for h,t in zip(self.hist,terms):h[:-1]=h[1:];h[-1]=t
  yaw=math.atan2(R[1,0],R[0,0]);rot=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
  points=self.points@rot.T+d.qpos[:2];height=np.clip(d.qpos[2]-.5-self.heights(points),-1,1)*5
  course=np.array([math.sin(self.imu),math.cos(self.imu)-1,float(active)])
  return torch.tensor(np.r_[np.concatenate([h.ravel() for h in self.hist]),height,course],dtype=torch.float32)[None]
 def contact_forces(self):
  forces=np.zeros(2)
  for contact_id,contact in enumerate(self.data.contact):
   b1,b2=self.model.geom_bodyid[[contact.geom1,contact.geom2]]
   for i,b in enumerate(self.footids):
    if b in (b1,b2):
     wrench=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,contact_id,wrench)
     world=contact.frame.reshape(3,3).T@wrench[:3]
     forces[i]+=world[2] if b2==b else -world[2]
  return forces
 def step(self):
  cmd,full=self.command()
  with torch.inference_mode():
   raw=self.actor(self.observations(cmd))[0];above=torch.relu(raw-self.iu);below=torch.relu(self.il-raw)
   action=torch.where(raw>self.iu,self.iu+self.tw*above/(self.tw+above),torch.where(raw<self.il,self.il-self.tw*below/(self.tw+below),raw)).clamp(self.lower,self.upper)
  self.action=action.numpy();self.data.ctrl[self.actids]=self.default+.25*self.action
  for _ in range(self.decimation):mujoco.mj_step(self.model,self.data)
  mujoco.mj_forward(self.model,self.data)
  R=rotation(self.data.qpos[3:7]);tilt=math.acos(np.clip(R[2,2],-1,1));_,full=self.command();self.stable=self.stable+self.dt if self.parked and np.linalg.norm(cmd)<1e-4 and full and (self.contact_forces()>5).all() and np.linalg.norm(self.data.qvel[:3])<.15 and np.linalg.norm(self.data.qvel[3:6])<.3 and tilt<.35 else 0.
  if self.stable>=1:self.status='success'
  if self.data.qpos[2]<.2 or tilt>.8 or not np.isfinite(self.data.qpos).all():self.status='fallen'
  s=self.data.qpos[:2]@self.direction;t=self.data.qpos[:2]@np.array([-self.direction[1],self.direction[0]])
  if (np.abs(self.data.qpos[:2])>=3.9).any() or (s>(.7 if self.course['family']=='cardinal' else .7*math.sqrt(2)) and abs(t)>.42):self.status='boundary'
  if self.data.time>=30 and self.status=='running':self.status='timeout'
  return cmd

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=DEFAULT);p.add_argument('--level',type=int,default=10);p.add_argument('--family',default='cardinal');p.add_argument('--variant',type=int,default=2);p.add_argument('--angle',type=float,default=0);p.add_argument('--speed',type=float,default=1);p.add_argument('--port',type=int,default=8091);p.add_argument('--headless',action='store_true');a=p.parse_args()
 torch.set_num_threads(1);r=Runner(a.checkpoint,a.level,a.family,a.variant,a.angle,a.speed)
 if a.headless:
  while r.status=='running':r.step()
  print(json.dumps({'status':r.status,'time':r.data.time,'max_progress':r.max_progress,'root':r.data.qpos[:3].tolist(),'level':a.level,'family':a.family,'variant':a.variant}));return
 import viser,trimesh
 server=viser.ViserServer(port=a.port);server.scene.set_up_direction('+z')
 nodes=[];terrain_nodes=[]
 def build_visuals():
  for node in terrain_nodes:node.remove()
  for node,bid in nodes:node.remove()
  terrain_nodes.clear();nodes.clear()
  for i,mesh in enumerate(r.meshes):
   mesh=mesh.copy();mesh.apply_translation([-4,-4,0]);mesh.visual.face_colors=[50,52,58,255] if i==0 else [122,166,190,255];terrain_nodes.append(server.scene.add_mesh_trimesh(f'/terrain/{i}',mesh))
  for v in r.profile['visuals']:
   mesh=trimesh.load(ROOT/'scripts/sim2sim/assets'/f"{v['mesh']}.stl");node=server.scene.add_mesh_trimesh('/robot/'+v['mesh'],mesh);nodes.append((node,r.model.body(v['body']).id))
 build_visuals()
 info=server.gui.add_markdown('MuJoCo 原生动力学 · model_8999 · 相同 2 cm 栅格地图')
 pause=server.gui.add_checkbox('暂停',False);follow=server.gui.add_checkbox('跟随视角',True)
 family_ui=server.gui.add_dropdown('地图类型',['cardinal','diagonal'],initial_value=a.family)
 level_ui=server.gui.add_slider('难度',min=0,max=10,step=1,initial_value=a.level)
 variant_ui=server.gui.add_slider('大小/间距扰动',min=0,max=3,step=1,initial_value=a.variant)
 angle_ui=server.gui.add_dropdown('出生方向',['0','90','180','-90','45','135','-135','-45'],initial_value=str(int(a.angle)))
 speed_ui=server.gui.add_slider('过桩指令速度',min=.5,max=1.,step=.05,initial_value=a.speed)
 button=server.gui.add_button('应用设置并重新开始');pending=[]
 @button.on_click
 def reset(_):
  angle=float(angle_ui.value);family=family_ui.value
  if (family=='cardinal' and angle not in (0,90,180,-90)) or (family=='diagonal' and angle not in (45,135,-135,-45)):
   info.content='方向与地图不匹配：正向选 0/90/180/-90，斜向选 45/135/-135/-45。';return
  pending.append((int(level_ui.value),family,int(variant_ui.value),angle,float(speed_ui.value)))
 @server.on_client_connect
 def connected(_):pending.append((a.level,a.family,a.variant,a.angle,a.speed))
 while True:
  start=time.monotonic()
  if pending:
   level,family,variant,angle,speed=pending.pop();pending.clear()
   r=Runner(a.checkpoint,level,family,variant,angle,speed);build_visuals();pause.value=False
  if not pause.value:r.step()
  for node,bid in nodes:node.position=r.data.xpos[bid];node.wxyz=r.data.xquat[bid]
  if int(r.data.time/r.dt)%10==0:
   info.content=f'MuJoCo · **8999 轮** · {r.course["family"]} · 方向 {r.angle:g}°\n\n状态：{r.status} · 时间 {r.data.time:.1f}s · 进度 {r.max_progress:.2f}m'
   if follow.value:
    for client in server.get_clients().values():
     target=r.data.qpos[:3].copy();target[2]=.6;client.camera.look_at=target;client.camera.position=target+np.r_[r.direction*2.8,2.8]
  if r.status!='running':pause.value=True
  time.sleep(max(0,r.dt-(time.monotonic()-start)))
if __name__=='__main__':main()
