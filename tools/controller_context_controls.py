"""One bounded queue per node; immutable siblings run as separate processes."""
import os,sys,time,json,argparse,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.controller_protected_fusion import queue
from causal_tsad.io import atomic_json,sha256
def main(a):
 r=a.root;plan=json.loads(a.plan.read_text());n=len(plan['gpus']);deadline=plan['deadline_unix'];tasks=[]
 with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 for i in range(n):
  for variant,kind in [('fixed','source'),('student','source'),('rolling','source'),('student','target'),('rolling','target')]:
   code=plan['student_code'] if variant=='student' else plan['context_code'];name=kind+'_'+variant+'_w'+str(i);tasks.append(dict(name=name,cmd=[sys.executable,code+'/tools/worker_context_controls.py','--root',str(r),'--plan',str(a.plan),'--kind',kind,'--variant',variant,'--worker',str(i),'--workers',str(n),'--deadline',str(deadline)]))
 atomic_json(r/'controller_progress.json',dict(status='running',source_mode=plan['source_mode'],planned_tasks=len(tasks),target_systems=len(plan['target_rows']),optimizer_updates=0))
 queue(r,tasks,plan['gpus'],deadline,'context_controls',minimum_free_MiB=1792);records=[];counts={k:0 for k in ['query_scalar_forwards','full_input_scalar_slots','Gaussian_covariance_factorizations','context_ridge_factorizations','cached_prior_query_applications']}
 for p in r.glob('*_w*_manifest.json'):
  m=json.loads(p.read_text());assert m['status']=='complete' and m['parameters_unchanged'] and m['optimizer_updates']==m['target_updates']==0;records+=m['records']
  for k in counts:counts[k]+=m[k]
 assert len(records)==288+4*len(plan['target_rows']) and len(records)==len(set(records))
 atomic_json(r/'manifest.json',dict(status='complete',files=[dict(path=p,SHA256=sha256(r/p),scores_SHA256=json.loads((r/p).read_text())['scores_SHA256']) for p in records],**counts,plan_SHA256=sha256(a.plan),optimizer_updates=0,target_updates=0,validation_or_locked_arrays_opened=False))
 atomic_json(r/'controller_progress.json',dict(status='complete',records=len(records),**counts,optimizer_updates=0,target_updates=0,next='fetch both nodes then pool only source-calibration normal scores; AP/AUC/RCA already computed; no target threshold fitting'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
 try:main(a)
 except Exception:atomic_json(a.root/'controller_error.json',dict(error=traceback.format_exc()));atomic_json(a.root/'controller_progress.json',dict(status='needs_attention',new_dispatch_stopped=True));raise
