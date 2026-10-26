"""Inference queues release GPU workers before original-metric CPU phase."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json
from controller_protected_fusion import queue
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();r=a.root;code=Path(__file__).resolve().parents[1];plan=json.loads(a.plan.read_text());deadline=time.time()+10*3600
 with (r/'target_controller.lock').open('x') as f:f.write(str(os.getpid()))
 tasks=[]
 for entry in plan['inference_jobs']:
  tasks.append(dict(name=entry['name'],cmd=[sys.executable,str(code/'tools/infer_fusion_development.py'),'--root',str(r),'--benchmark',plan['benchmark'],'--jobs',entry['path'],'--seed',str(entry['seed'])]))
 queue(r,tasks,plan['gpus'],time.time()+3*3600,'target_inference')
 atomic_json(r/'target_controller_progress.json',dict(status='running',stage='CPU_metrics_no_GPU_workers',pid=os.getpid()))
 # Two seeds on one node split CPU capacity; formulas spawn their own workers.
 active=[]
 for entry in plan['metric_jobs']:
  log=(r/'logs'/('target_metrics_s'+str(entry['seed'])+'.log')).open('x');cmd=[sys.executable,str(code/'tools/metric_fusion_development.py'),'--root',str(r),'--benchmark',plan['benchmark'],'--metric-code',plan['metric_code'],'--jobs',entry['path'],'--workers',str(max(1,plan['metric_workers']//len(plan['metric_jobs'])))];p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);active.append((p,log))
 while any(p.poll() is None for p,_ in active):
  assert time.time()<deadline,'Fixed target technical deadline';time.sleep(10)
 for p,log in active:log.close()
 failed=[p.returncode for p,_ in active if p.returncode]
 atomic_json(r/'target_controller_progress.json',dict(status='needs_attention' if failed else 'complete',stage='complete',failed=failed,parameter_updates=0,validation_or_locked_arrays_opened=False));assert not failed
