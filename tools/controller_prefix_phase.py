"""Two GPUs, four fixed source-probe jobs; no training or benchmark arrays."""
import os,sys,json,argparse,time,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json
def run(a):
    plan=json.loads(a.plan.read_text());fd=os.open(str(a.root/'controller.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);jobs=[dict(arm=arm,mode=mode) for arm in ['frozen_basis','normal_only'] for mode in ['true','empty']];active={};done=[]
    while jobs or active:
        if list((a.root/'errors').glob('*.json')) or time.time()>plan['deadline_unix']:atomic_json(a.root/'progress.json',dict(status='needs_attention',completed=done,pending=jobs));return
        for gpu,(p,j) in list(active.items()):
            if p.poll() is None:continue
            assert p.returncode==0;done.append(j);del active[gpu]
        for gpu in plan['gpus']:
            if jobs and gpu not in active:
                j=jobs.pop(0);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
                with (a.root/'logs'/(j['arm']+'_'+j['mode']+'.log')).open('x') as f:p=subprocess.Popen([sys.executable,plan['code']+'/tools/probe_prefix_phase.py','--root',str(a.root),'--plan',str(a.plan),'--arm',j['arm'],'--mode',j['mode']],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=(p,j)
        atomic_json(a.root/'progress.json',dict(status='running',completed=done,active=[dict(gpu=g,pid=p.pid,**j) for g,(p,j) in active.items()],pending=jobs));time.sleep(5)
    atomic_json(a.root/'progress.json',dict(status='complete',completed=done));atomic_json(a.root/'manifest.json',dict(status='complete',jobs=[json.loads((a.root/(j['arm']+'_'+j['mode']+'_manifest.json')).read_text()) for j in done],source_only=True,benchmark_arrays_opened=False,optimizer_updates=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors/controller.json',dict(error=traceback.format_exc()));atomic_json(a.root/'progress.json',dict(status='needs_attention',reason='controller exception'));raise
