"""One node queue interleaves two independent, committed proposal snapshots.

GPU tasks finish before CPU prior/metric jobs. Existing source/target evidence is
read-only; exclusive locks prevent duplicate launches. Failure stops new dispatch.
"""
import os,sys,json,time,argparse,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.controller_protected_fusion import queue
from causal_tsad.io import atomic_json,sha256
def main(a):
 r=a.root;plan=json.loads(a.plan.read_text());deadline=plan['GPU_deadline_unix'];pilot=Path(plan['pilot_root']);code=Path(__file__).resolve().parents[1]
 for root in [r,pilot]:
  with (root/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 tasks=[];local=len(plan['gpus'])
 for i in range(local):
  tasks.append(dict(name='covariance_worker'+str(i),cmd=[sys.executable,str(code/'tools/cache_causal_syndromes.py'),'--source',str(r/'source'),'--output',str(r/'cache'),'--modes',plan['control_mode'],'--worker',str(i),'--workers',str(local),'--deadline',str(deadline)]))
  tasks.append(dict(name='pilot_worker'+str(i),cmd=[sys.executable,plan['pilot_code']+'/tools/infer_syndrome_pilot.py','--root',str(pilot),'--plan',str(pilot/'job_plan.json'),'--worker',str(i),'--workers',str(local),'--deadline',str(deadline)]))
 atomic_json(r/'controller_progress.json',dict(status='running',stage='interleaved_GPU',planned_source_system_mode_records=80,planned_development_systems=plan['pilot_local_systems'],plan_SHA256=sha256(a.plan)))
 queue(r,tasks,plan['gpus'],deadline,'followup_GPU',minimum_free_MiB=1792);records=[];counts={k:0 for k in ['query_scalar_forwards','full_input_scalar_slots','Gaussian_covariance_factorizations','context_ridge_factorizations']}
 for i in range(local):
  v=json.loads((r/'cache'/('worker'+str(i)+'_manifest.json')).read_text());assert v['status']=='complete';records+=v['records']
  for k in counts:counts[k]+=v[k]
 assert len(records)==80 and len({(v['key'],v['mode']) for v in records})==80
 atomic_json(r/'cache/manifest.json',dict(status='complete',records=records,**counts,source_manifest_SHA256=sha256(r/'source/manifest.json'),benchmark_arrays_opened=False))
 rec=[];pc={k:0 for k in counts}
 for i in range(local):
  v=json.loads((pilot/('worker'+str(i)+'_manifest.json')).read_text());assert v['status']=='complete' and v['parameters_unchanged'] and v['target_updates']==0;rec+=v['records']
  for k in pc:pc[k]+=v[k]
 assert len(rec)==plan['pilot_local_systems'] and len({v['id'] for v in rec})==len(rec)
 atomic_json(pilot/'inference_manifest.json',dict(status='complete',records=rec,**pc,target_updates=0,TimeRCD_new_forwards=0,validation_or_locked_arrays_opened=False))
 # GPU allocations are now released; these are seven-parameter CPU fits only.
 for seed in [348,349,350]:
  assert time.time()<deadline;name=plan['control_mode']+'_s'+str(seed);atomic_json(r/'controller_progress.json',dict(status='running',stage='source_prior_CPU',current=name,GPU_complete=True))
  with (r/'logs'/(name+'.log')).open('x') as f:p=subprocess.run([sys.executable,str(code/'tools/train_syndrome_prior.py'),'--source',str(r/'source'),'--cache',str(r/'cache'),'--output',str(r/'runs'/name),'--mode',plan['control_mode'],'--seed',str(seed)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,int(deadline-time.time())))
  assert p.returncode==0,(r/'logs'/(name+'.log')).read_text()[-3000:]
 atomic_json(r/'controller_progress.json',dict(status='complete',source_prior_runs=3,optimizer_updates=96,GPU_complete=True,source_query_scalar_forwards=counts['query_scalar_forwards'],pilot_inference_complete=True))
 atomic_json(pilot/'controller_progress.json',dict(status='running',stage='original_metrics_CPU',inference_complete=True,local_systems=len(rec)))
 with (pilot/'logs/metrics.log').open('x') as f:p=subprocess.run([sys.executable,plan['pilot_code']+'/tools/metric_syndrome_pilot.py','--root',str(pilot),'--plan',str(pilot/'job_plan.json'),'--workers','8'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,timeout=4*3600)
 assert p.returncode==0,(pilot/'logs/metrics.log').read_text()[-3000:]
 atomic_json(pilot/'controller_progress.json',dict(status='complete',inference_complete=True,local_systems=len(rec),target_updates=0,validation_or_locked_arrays_opened=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
 try:main(a)
 except Exception:
  error=dict(error=traceback.format_exc(),new_dispatch_stopped=True);atomic_json(a.root/'controller_error.json',error);atomic_json(a.root/'controller_progress.json',dict(status='needs_attention',**error));raise
