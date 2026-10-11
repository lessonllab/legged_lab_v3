"""Interface parity plus a small native MuJoCo directional transfer matrix."""
import json,sys,hashlib
from pathlib import Path
import numpy as np,torch,mujoco
from beamdojo_mujoco import Runner,ROOT,DEFAULT
from tensordict import TensorDict
from rsl_rl.models import MLPModel

def main():
 torch.set_num_threads(1);r=Runner();saved=torch.load(DEFAULT,map_location='cpu',weights_only=False)
 obs=r.observations(np.array([1.,0,0]));td=TensorDict({'policy':obs[:,:480],'terrain':obs[:,480:705],'course_tracking':obs[:,705:]},batch_size=[1])
 reference=MLPModel(td,{'actor':['policy','terrain','course_tracking']},'actor',29,hidden_dims=[512,256,128],activation='elu',obs_normalization=False,distribution_cfg={'class_name':'GaussianDistribution','init_std':1.,'std_type':'scalar'})
 reference.load_state_dict(saved['actor_state_dict'],strict=True)
 with torch.inference_mode():assert torch.equal(reference(td),r.actor(obs))
 # Actual native rays against static terrain: 160000 centers, exact binary raster.
 group=np.array([1,0,0,0,0,0],dtype=np.uint8);gid=np.zeros(1,dtype=np.int32);errors=0
 for y in range(400):
  for x in range(400):
   p=np.array([-4+(x+.5)*.02,-4+(y+.5)*.02,2.]);distance=mujoco.mj_ray(r.model,r.data,p,np.array([0.,0.,-1.]),group,True,-1,gid)
   expected=0. if r.mask[y,x] else -1.
   if abs(2-distance-expected)>1e-7:errors+=1
 assert errors==0,errors
 rows=[]
 for family,angles in [('cardinal',[0,90,180,-90]),('diagonal',[45,135,-135,-45])]:
  for level in [0,10]:
   for angle in angles:
    r=Runner(level=level,family=family,variant=2,angle=angle,speed=1.)
    while r.status=='running':r.step()
    row={'family':family,'level':level,'angle':angle,'variant':2,'speed':1.,'status':r.status,'seconds':r.data.time,'progress':r.max_progress,'root':r.data.qpos[:3].tolist()};rows.append(row);print(json.dumps(row),flush=True)
 report={'checkpoint':str(DEFAULT),'sha256':hashlib.sha256(DEFAULT.read_bytes()).hexdigest(),'actor_raw_bitwise_equal':True,'terrain_ray_centers_checked':160000,'terrain_ray_mismatches':errors,'episodes':rows,'successes':sum(x['status']=='success' for x in rows),'count':len(rows),'note':'Fixed seed42 variant2; small deterministic transfer test, not broad success estimate. Ideal terrain queries and proprioception; native MuJoCo physics.'}
 (ROOT/'artifacts/sim2sim_mujoco_8999/results.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
