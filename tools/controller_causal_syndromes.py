"""Finite GPU evidence then CPU tiny-prior optimization; no duplicated node work."""
import os,sys,json,time,argparse,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controller_protected_fusion import queue
from causal_tsad.io import atomic_json,sha256
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();r=a.root;plan=json.loads(a.plan.read_text());code=Path(__file__).resolve().parents[1];deadline=time.time()+4*3600
 with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 tasks=[dict(name='worker'+str(i),cmd=[sys.executable,str(code/'tools/cache_causal_syndromes.py'),'--source',str(r/'source'),'--output',str(r/'cache'),'--modes',*plan['modes'],'--worker',str(i),'--workers',str(len(plan['gpus'])),'--deadline',str(deadline)]) for i in range(len(plan['gpus']))]
 queue(r,tasks,plan['gpus'],deadline,'evidence',minimum_free_MiB=1792);records=[];query=slots=0;factors=conditioning=0
 for p in (r/'cache').glob('worker*_manifest.json'):
  v=json.loads(p.read_text());assert v['status']=='complete';records+=v['records'];query+=v['query_scalar_forwards'];slots+=v['full_input_scalar_slots'];factors+=v['Gaussian_covariance_factorizations'];conditioning+=v['context_ridge_factorizations']
 assert len(records)==80*len(plan['modes']) and len({(v['key'],v['mode']) for v in records})==len(records);atomic_json(r/'cache/manifest.json',dict(status='complete',records=records,query_scalar_forwards=query,full_input_scalar_slots=slots,Gaussian_covariance_factorizations=factors,context_ridge_factorizations=conditioning,source_manifest_SHA256=sha256(r/'source/manifest.json'),benchmark_arrays_opened=False))
 for mode in plan['modes']:
  for seed in [348,349,350]:
   assert time.time()<deadline;name=mode+'_s'+str(seed);atomic_json(r/'controller_progress.json',dict(status='running',stage='tiny_prior_training',current=name,GPU_evidence_complete=True))
   with (r/'logs'/(name+'.log')).open('x') as f:p=subprocess.run([sys.executable,str(code/'tools/train_syndrome_prior.py'),'--source',str(r/'source'),'--cache',str(r/'cache'),'--output',str(r/'runs'/name),'--mode',mode,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=max(1,int(deadline-time.time())))
   assert p.returncode==0,(r/'logs'/(name+'.log')).read_text()[-3000:]
 atomic_json(r/'controller_progress.json',dict(status='complete',modes=plan['modes'],prior_runs=3*len(plan['modes']),optimizer_updates=32*3*len(plan['modes']),target_arrays_opened=False))
