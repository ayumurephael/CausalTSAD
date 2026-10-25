"""Reuse source likelihood fits, fixed masked-evidence queue only."""
import argparse,json,os,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controller_protected_fusion import queue
from causal_tsad.io import atomic_json,sha256
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();r=a.root;plan=json.loads(a.plan.read_text());code=Path(__file__).resolve().parents[1]
 with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 assert json.loads((r/'preflight.json').read_text())['status']=='passed';tasks=[];fits={}
 for seed in plan['seeds']:
  old=Path(plan['original_root'])/'parameters'/('s'+str(seed)+'.json');parameters=r/'parameters'/old.name;atomic_json(parameters,json.loads(old.read_text()));assert sha256(old)==sha256(parameters);fits[str(seed)]=sha256(old)
  for i in range(plan['workers_per_seed']):
   name='s'+str(seed)+'_w'+str(i);tasks.append(dict(name=name,cmd=[sys.executable,str(code/'tools/evaluate_missing_bayes.py'),'--output',str(r/'runs'/name),'--parameters',str(parameters),'--seed',str(seed),'--worker',str(i),'--workers',str(plan['workers_per_seed'])]))
 atomic_json(r/'source_fit_reuse.json',fits);queue(r,tasks,plan['gpus'],time.time()+4*3600,'source');atomic_json(r/'controller_progress.json',dict(status='complete',systems=128,fit_replicates=len(plan['seeds']),optimizer_updates=0,benchmark_arrays_opened=False))
