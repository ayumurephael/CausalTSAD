"""GPU inference followed by a bounded CPU metric stage; never retrains."""
import os,sys,time,json,argparse,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json
def run(a):
    plan=json.loads(a.plan.read_text());fd=os.open(str(a.root/'controller.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);active=[]
    for worker,gpu in enumerate(plan['gpus']):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with (a.root/'logs'/('worker'+str(worker)+'.log')).open('x') as f:p=subprocess.Popen([sys.executable,plan['code']+'/tools/infer_meta_transfer.py','--root',str(a.root),'--plan',str(a.plan),'--worker',str(worker),'--workers',str(len(plan['gpus']))],env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        active.append(dict(process=p,worker=worker,gpu=gpu))
    while any(v['process'].poll() is None for v in active):
        if time.time()>plan['deadline_unix'] or list((a.root/'errors').glob('*.json')):atomic_json(a.root/'progress.json',dict(status='needs_attention',phase='inference',reason='fixed deadline or project error; leave existing worker evidence'));return
        atomic_json(a.root/'progress.json',dict(status='running',phase='inference',active=[dict(worker=v['worker'],gpu=v['gpu'],pid=v['process'].pid) for v in active if v['process'].poll() is None]));time.sleep(5)
    assert all(v['process'].returncode==0 for v in active);manifests=[json.loads((a.root/('worker'+str(v['worker'])+'_manifest.json')).read_text()) for v in active];assert sum(len(v['records']) for v in manifests)==len(plan['rows'])
    atomic_json(a.root/'progress.json',dict(status='running',phase='original_four_metrics',GPU_inference_complete=True))
    with (a.root/'logs/metrics.log').open('x') as f:p=subprocess.Popen([sys.executable,plan['code']+'/tools/metrics_meta_transfer.py','--root',str(a.root),'--plan',str(a.plan),'--workers','8'],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    while p.poll() is None:
        if time.time()>plan['deadline_unix']:atomic_json(a.root/'progress.json',dict(status='needs_attention',phase='original_four_metrics',reason='fixed deadline'));return
        time.sleep(5)
    assert p.returncode==0;atomic_json(a.root/'manifest.json',dict(status='complete',workers=manifests,systems=len(plan['rows']),target_updates=0,TimeRCD_new_forwards=0));atomic_json(a.root/'progress.json',dict(status='complete',phase='complete',systems=len(plan['rows'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors/controller.json',dict(error=traceback.format_exc()));atomic_json(a.root/'progress.json',dict(status='needs_attention',reason='controller exception; evidence retained'));raise
