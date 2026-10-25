"""Independent frozen graph transfer; GPU phase then sequential seeds of CPU metrics."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controller_protected_fusion import queue
from causal_tsad.io import atomic_json
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();r=a.root;plan=json.loads(a.plan.read_text());code=Path(__file__).resolve().parents[1];deadline=time.time()+10*3600
 with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 tasks=[dict(name=v['name'],cmd=[sys.executable,str(code/'tools/infer_causal_standalone.py'),'--root',str(r),'--benchmark',plan['benchmark'],'--jobs',v['path']]) for v in plan['inference_jobs']]
 queue(r,tasks,plan['gpus'],time.time()+3*3600,'target_inference');atomic_json(r/'controller_progress.json',dict(status='running',stage='CPU_metrics_no_GPU_reservation',pid=os.getpid()))
 for row in plan['metric_jobs']:
  with (r/'logs'/('metric_s'+str(row['seed'])+'.log')).open('x') as log:
   p=subprocess.Popen([sys.executable,str(code/'tools/metric_causal_standalone.py'),'--root',str(r),'--benchmark',plan['benchmark'],'--metric-code',plan['metric_code'],'--jobs',row['path'],'--workers','8'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
   while p.poll() is None:assert time.time()<deadline;time.sleep(10)
   if p.returncode:atomic_json(r/'controller_progress.json',dict(status='needs_attention',stage='CPU_metrics',seed=row['seed']));raise RuntimeError('CPU metrics failed; preserve all GPU arrays')
 atomic_json(r/'controller_progress.json',dict(status='complete',systems_per_seed=540,target_updates=0,validation_or_locked_arrays_opened=False))
