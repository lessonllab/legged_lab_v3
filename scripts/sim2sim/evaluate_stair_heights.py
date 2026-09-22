"""Evaluate a frozen checkpoint over exact training stair heights and initial yaws."""
import argparse
import json
import subprocess
from pathlib import Path
from export_training_terrain import export
from run_mujoco import ROOT

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--heights',type=float,nargs='+',default=[.08,.10,.12,.14,.16,.18,.20,.215,.23])
p.add_argument('--yaws',type=float,nargs='+',default=[0.])
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.checkpoint=a.checkpoint.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
results=[]
for height in a.heights:
 scene=a.output/f'stairs_{height*100:g}cm.xml'
 export(a.checkpoint.parent/'params/env.yaml',scene,height,Path('/home/ljc/isaaclab6/IsaacLab'))
 for yaw in a.yaws:
  report=a.output/f'result_{height*100:g}cm_yaw{yaw:g}.json'
  cmd=['/home/ljc/isaaclab/bin/python',str(ROOT/'scripts/sim2sim/run_mujoco.py'),'--checkpoint',str(a.checkpoint),'--scene',str(scene),'--headless','--target','6.7','0','--speed','.65','--yaw',str(yaw),'--duration','30','--report',str(report)]
  if not report.exists():
   completed=subprocess.run(cmd)
   if completed.returncode not in (0,2) or not report.exists():
    raise RuntimeError(f'Evaluation failed: {completed.returncode}')
  elif json.loads(report.read_text())['checkpoint'] != str(a.checkpoint):
   raise ValueError('Existing output belongs to another checkpoint')
  row=json.loads(report.read_text());row.update(step_height=height,yaw=yaw);results.append(row)
  (a.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
