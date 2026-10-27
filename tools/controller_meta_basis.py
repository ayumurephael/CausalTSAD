"""Finite parallel queue. Stops only this stage's new dispatch on any error."""
import os,sys,argparse,json,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json
def run(a):
    plan=json.loads(a.plan.read_text());a.root.mkdir(parents=True,exist_ok=True);fd=os.open(str(a.root/'controller.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    pending=plan['jobs'].copy();active={};completed=[];started=time.time();error=None
    while pending or active:
        if (a.root/'STOP_NEW_DISPATCH').exists() or list((a.root/'errors').glob('*.json')):error='stage error or stop marker; no new dispatch'
        if time.time()>plan['deadline_unix']:error='fixed six-hour technical deadline'
        for gpu,(p,job) in list(active.items()):
            if p.poll() is None:continue
            if p.returncode: error='worker failed; preserve prior results'
            else:completed.append(job)
            del active[gpu]
        if error:
            atomic_json(a.root/'progress.json',dict(status='needs_attention',reason=error,completed=completed,active=[v[1] for v in active.values()],pending=pending));return
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode();free={int(v.split(',')[0]):float(v.split(',')[1]) for v in raw.strip().splitlines()}
        for gpu in plan['gpus']:
            if pending and gpu not in active and free.get(gpu,0)>=5120:
                job=pending.pop(0);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
                log=a.root/'logs'/(job['arm']+'_s'+str(job['seed'])+'.log');log.parent.mkdir(exist_ok=True)
                with log.open('x') as f:p=subprocess.Popen([sys.executable,plan['code']+'/tools/train_meta_basis.py','--root',str(a.root),'--plan',str(a.plan),'--arm',job['arm'],'--seed',str(job['seed'])],env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                active[gpu]=(p,job)
        atomic_json(a.root/'progress.json',dict(status='running',completed=completed,active=[dict(gpu=g,pid=p.pid,**j) for g,(p,j) in active.items()],pending=pending,seconds=time.time()-started));time.sleep(5)
    atomic_json(a.root/'progress.json',dict(status='complete',completed=completed,pending=[],active=[],seconds=time.time()-started))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);run(p.parse_args())
