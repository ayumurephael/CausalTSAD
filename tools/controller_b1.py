"""Finite per-node GPU queue. Stops its own queue on an error; touches no other jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json


def main(a):
    root=a.root;root.mkdir(parents=True,exist_ok=True)
    guard=root/'controller.lock'
    # Exclusive creation prevents accidental double launch.
    handle=guard.open('x');handle.write(str(os.getpid()));handle.close()
    plan=json.loads(a.plan.read_text());active={};pending=list(plan['jobs']);complete=[]
    status=dict(status='running',pid=os.getpid(),jobs=len(pending),completed=0,active=[])
    def save():
        status.update(completed=len(complete),active=[dict(gpu=g,run=v['job']['name'],stage=v['stage'],pid=v['process'].pid) for g,v in active.items()],pending=len(pending))
        atomic_json(root/'controller_progress.json',status)
    def launch(gpu,job,stage):
        dest=root/'runs'/job['name'];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        env['PYTHONUNBUFFERED']='1';code=Path(__file__).resolve().parent
        if stage=='train':
            cmd=[sys.executable,str(code/'train_b1.py'),'--source',str(a.source),'--output',str(dest),
                '--config',str(code.parent/'experiments/b1_shared_v1.json'),'--mode',job['mode'],'--seed',str(job['seed']),
                '--local-commit',plan['local_commit'],'--github-commit',plan['github_commit']]
        else:
            cmd=[sys.executable,str(code/'evaluate_b1.py'),'--run',str(dest),'--benchmark',str(a.benchmark),'--metric-code',str(a.metric_code)]
        log=root/'logs'/(job['name']+'_'+stage+'.log');log.parent.mkdir(exist_ok=True)
        with log.open('x') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        active[gpu]=dict(job=job,stage=stage,process=p,started=time.monotonic())
    try:
        while pending or active:
            for gpu in plan['gpus']:
                if gpu not in active and pending:launch(gpu,pending.pop(0),'train')
            save()
            time.sleep(5)
            for gpu,entry in list(active.items()):
                code=entry['process'].poll()
                if code is None:
                    if time.monotonic()-entry['started']>3*3600:
                        entry['process'].terminate();raise TimeoutError('This project worker exceeded its 3h stage deadline')
                    continue
                if code!=0:raise RuntimeError(f"{entry['job']['name']} {entry['stage']} failed: exit {code}; no new jobs will start")
                if entry['stage']=='train':launch(gpu,entry['job'],'evaluate')
                else:complete.append(entry['job']);del active[gpu]
        status.update(status='complete',completed_runs=complete);save()
    except Exception as e:
        status.update(status='needs_attention',error=str(e),new_jobs_stopped=True)
        # Already-running workers may finish; no task owned by another project is stopped.
        save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--metric-code',type=Path,required=True);main(p.parse_args())
