"""One bounded source-only worker per allocated GPU."""
import os,sys,json,time,argparse,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256
def run(a):
    fd=os.open(str(a.root/'controller.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);plan=json.loads(a.plan.read_text());jobs=[]
    for i,g in enumerate(plan['gpus']):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(g)
        with (a.root/'logs'/('worker'+str(i)+'.log')).open('x') as f:p=subprocess.Popen([sys.executable,plan['code']+'/tools/infer_sparse_edge.py','--root',str(a.root),'--plan',str(a.plan),'--worker',str(i),'--workers',str(len(plan['gpus']))],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append(p)
    while any(p.poll() is None for p in jobs):
        assert time.time()<plan['deadline_unix'] and not list((a.root/'errors').glob('*.json'))
        atomic_json(a.root/'progress.json',dict(status='running',active=sum(p.poll() is None for p in jobs),workers=len(jobs)));time.sleep(3)
    assert all(p.returncode==0 for p in jobs);workers=[json.loads((a.root/('worker'+str(i)+'_manifest.json')).read_text()) for i in range(len(jobs))]
    atomic_json(a.root/'manifest.json',dict(status='complete',workers=workers,new_backbone_forwards=0,optimizer_updates=0,target_arrays_opened=False,plan_SHA256=sha256(a.plan)));atomic_json(a.root/'progress.json',dict(status='complete',workers=len(jobs)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors/controller.json',dict(error=traceback.format_exc()));atomic_json(a.root/'progress.json',dict(status='needs_attention'));raise
