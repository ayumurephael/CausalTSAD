"""All local GPUs build a reusable seed-specific feature bank, then finite heads."""
import os,sys,json,time,argparse,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json
def spawn(a,script,args,gpu,log):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with (a.root/'logs'/log).open('x') as f:return subprocess.Popen([sys.executable,json.loads(a.plan.read_text())['code']+'/tools/'+script,'--root',str(a.root),'--plan',str(a.plan),*args],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
def run(a):
    fd=os.open(str(a.root/'controller.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);p=json.loads(a.plan.read_text());gpus=p['gpus'];jobs=[]
    for w,g in enumerate(gpus):jobs.append(dict(process=spawn(a,'cache_temporal_innovations.py',['--worker',str(w),'--workers',str(len(gpus))],g,'cache'+str(w)+'.log'),gpu=g))
    def wait(jobs,phase):
        while any(v['process'].poll() is None for v in jobs):
            if time.time()>p['deadline_unix'] or list((a.root/'errors').glob('*.json')):atomic_json(a.root/'progress.json',dict(status='needs_attention',phase=phase,reason='deadline or project error; no new jobs'));return False
            atomic_json(a.root/'progress.json',dict(status='running',phase=phase,active=[dict(gpu=v['gpu'],pid=v['process'].pid) for v in jobs if v['process'].poll() is None]));time.sleep(5)
        assert all(v['process'].returncode==0 for v in jobs);return True
    if not wait(jobs,'frozen_feature_bank'):return
    manifests=[json.loads((a.root/('cache'+str(w)+'_manifest.json')).read_text()) for w in range(len(gpus))];index={}
    for m in manifests:
        assert m['normal_model_parameters_unchanged']
        for v in m['records']:key=str(v['seed'])+'/'+v['key']+'/'+v['background'];assert key not in index;index[key]=v
    assert len(index)==len(p['seeds'])*112*2;atomic_json(a.root/'feature_index.json',index)
    jobs=[]
    for i,(seed,mode) in enumerate((s,m) for s in p['seeds'] for m in ['point','fixed_mix','multiscale']):
        gpu=gpus[i%len(gpus)];jobs.append(dict(process=spawn(a,'train_temporal_innovation_head.py',['--seed',str(seed),'--mode',mode],gpu,mode+'_s'+str(seed)+'.log'),gpu=gpu))
    if not wait(jobs,'finite_readout_learning'):return
    runs={}
    for s in p['seeds']:
        for mode in ['point','fixed_mix','multiscale']:
            name=mode+'_s'+str(s);m=json.loads((a.root/'runs'/name/'manifest.json').read_text());assert m['status']=='complete';runs[name]=m
    atomic_json(a.root/'manifest.json',dict(status='complete',cache_workers=manifests,runs=runs,target_arrays_opened=False,backbone_optimizer_updates=0));atomic_json(a.root/'progress.json',dict(status='complete',runs=len(runs)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors/controller.json',dict(error=traceback.format_exc()));atomic_json(a.root/'progress.json',dict(status='needs_attention'));raise
