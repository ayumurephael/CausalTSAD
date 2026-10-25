"""Bounded independent C03 source-only queue, never reads a benchmark path."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json


def main(a):
    r=a.root;code=Path(__file__).resolve().parents[1];config=code/'experiments/c03_source_v1.json'
    cfg=json.loads(config.read_text());plan=json.loads(a.plan.read_text());started=time.monotonic()
    with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
    status=dict(status='running',stage='preflight',pid=os.getpid(),active=[],pending=9,completed=[])
    atomic_json(r/'controller_progress.json',status)
    def launch(cmd,name,gpu):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTHONUNBUFFERED']='1'
        with (r/'logs'/(name+'.log')).open('x') as f:
            return subprocess.Popen([sys.executable]+list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
    def bounded_wait(p,hours):
        began=time.monotonic()
        while p.poll() is None:
            if time.monotonic()-began>hours*3600:raise TimeoutError('Pre-training stage deadline')
            time.sleep(5)
        assert p.returncode==0,'Failed project worker; no next stage started'
    p=launch([code/'tests/preflight_c03.py','--output',r/'preflight.json'],'preflight',plan['gpus'][0]);bounded_wait(p,1)
    assert json.loads((r/'preflight.json').read_text())['status']=='passed'
    status['stage']='source_context_cache';atomic_json(r/'controller_progress.json',status)
    p=launch([code/'tools/prepare_c03.py','--source',a.source,'--output',r/'source_context_v1'],'prepare',plan['gpus'][0]);bounded_wait(p,2)
    pending=[dict(arm=arm,seed=seed,name=f'{arm}_s{seed}') for seed in cfg['seeds'] for arm in cfg['arms']]
    active={};complete=[]
    while pending or active:
        if time.monotonic()-started>cfg['max_campaign_hours']*3600:raise TimeoutError('Campaign technical deadline')
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode()
        free={int(l.split(',')[0]):float(l.split(',')[1]) for l in raw.strip().splitlines()}
        for gpu in plan['gpus']:
            if gpu in active or not pending or free.get(gpu,0)<8192:continue
            job=pending.pop(0);out=r/'runs'/job['name']
            command=[code/'tools/train_c03.py','--cache',r/'source_context_v1','--config',config,'--output',out,
                '--arm',job['arm'],'--seed',job['seed'],'--local-commit',plan['local_commit'],'--github-commit',plan['github_commit']]
            p=launch(command,job['name'],gpu);active[gpu]=dict(job=job,process=p,started=time.monotonic())
        status.update(stage='source_training',pending=len(pending),active=[dict(gpu=g,pid=e['process'].pid,run=e['job']['name']) for g,e in active.items()],completed=complete)
        atomic_json(r/'controller_progress.json',status);time.sleep(5)
        for gpu,e in list(active.items()):
            ret=e['process'].poll()
            if ret is None:
                if time.monotonic()-e['started']>3*3600:raise TimeoutError('Training worker technical deadline')
                continue
            if ret!=0:raise RuntimeError(e['job']['name']+' failed; no new jobs launched')
            complete.append(e['job']['name']);del active[gpu]
    status.update(status='complete',stage='complete',active=[],pending=0,completed=complete);atomic_json(r/'controller_progress.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    try:main(a)
    except Exception:
        path=a.root/'controller_progress.json';s=json.loads(path.read_text()) if path.exists() else {}
        s.update(status='needs_attention',error=traceback.format_exc(),new_jobs_stopped=True);atomic_json(path,s);raise
