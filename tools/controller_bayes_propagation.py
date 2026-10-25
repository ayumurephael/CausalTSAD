"""Bounded GPU source feasibility queue, independent from C04 and retained benchmark."""
import argparse,json,os,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from proposals.bayes_propagation import fit_source
from controller_protected_fusion import queue
from causal_tsad.io import atomic_json
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();r=a.root;plan=json.loads(a.plan.read_text());code=Path(__file__).resolve().parents[1]
 with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 assert json.loads((r/'preflight.json').read_text())['status']=='passed';tasks=[]
 for seed in plan['seeds']:
  parameters=r/'parameters'/('s'+str(seed)+'.json');assert not parameters.exists();atomic_json(parameters,fit_source(seed))
  for i in range(plan['workers_per_seed']):
   name='s'+str(seed)+'_w'+str(i);tasks.append(dict(name=name,cmd=[sys.executable,str(code/'tools/evaluate_bayes_propagation.py'),'--output',str(r/'runs'/name),'--parameters',str(parameters),'--seed',str(seed),'--worker',str(i),'--workers',str(plan['workers_per_seed'])]))
 queue(r,tasks,plan['gpus'],time.time()+4*3600,'source');atomic_json(r/'controller_progress.json',dict(status='complete',seed_count=len(plan['seeds']),independent_evaluation_systems=128,raw_benchmark_arrays_opened=False,original_TimeRCD_or_C04_updated=False))
