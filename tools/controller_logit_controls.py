"""Bounded supplementary score audit with no training or encoder inference."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json


def main(a):
    plan=json.loads(a.plan.read_text());r=a.output;assert not r.exists();r.mkdir();(r/'logs').mkdir()
    with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
    pending=list(plan['jobs']);active={};completed=[];start=time.monotonic()
    code=Path(__file__).resolve().parent
    while pending or active:
        if time.monotonic()-start>8*3600:raise TimeoutError('Readout audit technical deadline')
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode()
        free={int(l.split(',')[0]):float(l.split(',')[1]) for l in raw.strip().splitlines()}
        for gpu in plan['gpus']:
            if gpu in active or not pending or free.get(gpu,0)<8192:continue
            name=pending.pop(0);cmd=[sys.executable,str(code/'evaluate_logit_controls.py'),'--parent',str(a.parent),
                '--run',str(a.parent/'recovery_v1/runs'/name),'--output',str(r/'runs'/name),
                '--benchmark',str(a.benchmark),'--metric-code',str(a.metric_code)]
            if name=='pretrained_none_s348':cmd.append('--compute-base')
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTHONUNBUFFERED']='1'
            with (r/'logs'/(name+'.log')).open('x') as f:
                p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
            active[gpu]=dict(name=name,process=p,started=time.monotonic())
        atomic_json(r/'controller_progress.json',dict(status='running',pid=os.getpid(),pending=len(pending),completed=completed,
            active=[dict(gpu=g,pid=e['process'].pid,run=e['name']) for g,e in active.items()]))
        time.sleep(5)
        for gpu,e in list(active.items()):
            result=e['process'].poll()
            if result is None:
                if time.monotonic()-e['started']>3.1*3600:raise TimeoutError('Readout worker deadline')
                continue
            if result!=0:raise RuntimeError(e['name']+' readout audit failed, new jobs stopped')
            completed.append(e['name']);del active[gpu]
    atomic_json(r/'controller_progress.json',dict(status='complete',pid=os.getpid(),pending=0,active=[],completed=completed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--metric-code',type=Path,required=True);a=p.parse_args()
    try:main(a)
    except Exception:
        path=a.output/'controller_progress.json';s=json.loads(path.read_text()) if path.exists() else {}
        s.update(status='needs_attention',error=traceback.format_exc(),new_jobs_stopped=True);atomic_json(path,s);raise
