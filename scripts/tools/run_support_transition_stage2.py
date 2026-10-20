"""One1000-update corrective block, then serial paired evaluation and replays."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import traceback


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    root=Path(cfg['project']);out=Path(cfg['output']);out.mkdir(exist_ok=True)
    state=dict(pid=os.getpid(),state='initializing',jobs=[])
    def save(**values):
        state.update(values,updated_at=datetime.now().astimezone().isoformat())
        p=out/'pipeline.json';tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=2));tmp.replace(p)
    def command(cmd,log):
        job=dict(command=cmd,log=str(log),started_at=datetime.now().astimezone().isoformat())
        with log.open('wb') as stream:
            child=subprocess.Popen(cmd,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            job['pid']=child.pid;state['jobs'].append(job);save(child_pid=child.pid)
            code=child.wait()
        job.update(exit_code=code,finished_at=datetime.now().astimezone().isoformat());save(child_pid=None)
        if code:raise RuntimeError('Child failed: '+str(log))
    def evaluate(scope,label,reference,candidate):
        name=f'{label}_vs_corrected_{scope}';folder=out/name;folder.mkdir(exist_ok=False)
        cmd=['bash','scripts/run_with_unitree_official.sh','scripts/tools/compare_square_policy_effects.py',
            '--reference',reference,'--candidate',candidate,'--environment',cfg['baseline'],
            '--labels',label,'corrected16700','--variants',*(['0','7'] if scope=='saved' else ['3','4','6','7']),
            '--arm-audit','--output',str(folder),'--device','cpu','--visualizer','none']
        if scope=='fresh':cmd+=['--geometry-seed',str(cfg['fresh_geometry_seed'])]
        save(current_evaluation=name);command(cmd,folder/'process.log')
        for script in ('audit_square_arm_coordination.py','audit_support_transition_stage1.py'):
            command(['bash','scripts/run_with_unitree_official.sh','scripts/tools/'+script,
                     '--input',str(folder)],folder/(script.removesuffix('.py')+'.log'))
        # Two complementary replay sets include fixed flat/cardinal/diagonal
        # samples plus the worst relative stone case selected by the renderer.
        if (scope,label) in [('saved','14700'),('fresh','candidate15700')]:
            command(['bash','scripts/run_with_unitree_official.sh','scripts/tools/render_support_transition_replay.py',
                     '--input',str(folder)],folder/'render_support_transition_replay.log')
    save()
    try:
        diagnosis=json.loads((out/'diagnosis_pipeline.json').read_text())
        if diagnosis['state']!='complete':raise ValueError('Diagnostic jobs must exit before starting a native train scene')
        save(state='training')
        command(cfg['training_command'],out/'training.log')
        train=json.loads(Path(cfg['training_status']).read_text())
        if train['state']!='budget_complete' or train['completed_updates']!=1000:
            raise ValueError('Corrective training did not finish the declared1000budget')
        candidate=str(Path(train['run_dir'])/'model_16700.pt')
        if not Path(candidate).is_file():raise FileNotFoundError(candidate)
        save(state='paired_evaluations',candidate_checkpoint=candidate)
        for scope in ('saved','fresh'):
            for label,reference in [('14700',cfg['baseline']),('candidate15700',cfg['previous'])]:
                evaluate(scope,label,reference,candidate)
        save(state='complete',current_evaluation=None)
    except BaseException as error:
        save(state='failed',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
