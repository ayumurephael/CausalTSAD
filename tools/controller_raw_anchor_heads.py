"""Bounded head-only queue. Reuse completed caches; never create new features."""
import os,sys,time,json,argparse,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256
from proposals.raw_anchor_task_heads import ARMS
def run(a):
    fd=os.open(str(a.root/'controller.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    plan=json.loads(a.plan.read_text());pending=[(s,m) for s in plan['seeds'] for m in ARMS];active=[];done=[]
    while pending or active:
        assert time.time()<plan['deadline_unix'],'technical deadline; do not extend'
        assert not list((a.root/'errors').glob('*.json')),'project failure; stop new dispatch'
        for item in list(active):
            if item['process'].poll() is not None:
                assert item['process'].returncode==0;done.append(item['name']);active.remove(item)
        used={v['gpu'] for v in active}
        for gpu in plan['gpus']:
            if gpu in used or not pending:continue
            seed,mode=pending.pop(0);name=mode+'_s'+str(seed);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
            with (a.root/'logs'/(name+'.log')).open('x') as f:p=subprocess.Popen([sys.executable,plan['code']+'/tools/train_raw_anchor_heads.py','--root',str(a.root),'--plan',str(a.plan),'--seed',str(seed),'--mode',mode],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            active.append(dict(process=p,name=name,gpu=gpu))
        atomic_json(a.root/'progress.json',dict(status='running',completed=done,pending=len(pending),active=[dict(name=v['name'],pid=v['process'].pid,gpu=v['gpu']) for v in active]));time.sleep(3)
    runs={n:json.loads((a.root/'runs'/n/'manifest.json').read_text()) for n in done}
    assert all(m['status']=='complete' for m in runs.values())
    atomic_json(a.root/'manifest.json',dict(status='complete',runs=runs,new_backbone_forwards=0,target_arrays_opened=False,plan_SHA256=sha256(a.plan)))
    atomic_json(a.root/'progress.json',dict(status='complete',runs=len(runs)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors/controller.json',dict(error=traceback.format_exc()));atomic_json(a.root/'progress.json',dict(status='needs_attention'));raise
