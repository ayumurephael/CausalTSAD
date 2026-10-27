"""Static bounded task partition; no restart, training or result-based scheduling."""
import os,sys,json,time,argparse,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json
def run(a):
    fd=os.open(str(a.root/'controller.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    p=json.loads(a.plan.read_text());children=[]
    for w,g in enumerate(p['gpus']):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(g)
        with (a.root/'logs'/('worker'+str(w)+'.log')).open('x') as f:c=subprocess.Popen([sys.executable,p['code']+'/tools/infer_innovation_probe.py','--root',str(a.root),'--plan',str(a.plan),'--worker',str(w),'--workers',str(len(p['gpus']))],env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        children.append(dict(process=c,worker=w,gpu=g))
    while any(c['process'].poll() is None for c in children):
        if time.time()>p['deadline_unix'] or list((a.root/'errors').glob('*.json')):
            atomic_json(a.root/'progress.json',dict(status='needs_attention',reason='deadline or project error; preserve existing evidence'));return
        atomic_json(a.root/'progress.json',dict(status='running',active=[dict(worker=c['worker'],gpu=c['gpu'],pid=c['process'].pid) for c in children if c['process'].poll() is None]));time.sleep(5)
    assert all(c['process'].returncode==0 for c in children)
    manifests=[json.loads((a.root/('worker'+str(c['worker'])+'_manifest.json')).read_text()) for c in children]
    assert sum(len(v['records']) for v in manifests)==len(p['tasks'])
    atomic_json(a.root/'manifest.json',dict(status='complete',workers=manifests,records=len(p['tasks']),full_input_slots=sum(v['full_input_slots'] for v in manifests),query_slots=sum(v['query_slots'] for v in manifests),optimizer_updates=0,target_updates=0))
    atomic_json(a.root/'progress.json',dict(status='complete',records=len(p['tasks'])))
if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--root',type=Path,required=True);q.add_argument('--plan',type=Path,required=True);a=q.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors/controller.json',dict(error=traceback.format_exc()));atomic_json(a.root/'progress.json',dict(status='needs_attention'));raise
