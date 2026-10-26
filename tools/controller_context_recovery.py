"""Resume only missing jobs; reuse completed artifacts and let own live jobs finish."""
import os,sys,json,time,argparse,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.controller_protected_fusion import queue
from causal_tsad.io import atomic_json,sha256
def main(a):
 outer=a.outer;revision=a.revision;plan=json.loads((outer/'job_plan.json').read_text());recovery=json.loads(a.plan.read_text());deadline=plan['deadline_unix'];n=len(plan['gpus']);tasks=[];code=str(Path(__file__).resolve().parents[1]);retained=[]
 with (revision/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 for i in range(n):
  for variant,kind in [('fixed','source'),('student','source'),('rolling','source'),('student','target'),('rolling','target')]:
   name=kind+'_'+variant+'_w'+str(i);manifest=outer/(name+'_manifest.json')
   if manifest.exists():assert json.loads(manifest.read_text())['status']=='complete';retained.append(name);continue
   if name in recovery['existing_live_jobs']:retained.append(name);continue
   folder=plan['student_code'] if variant=='student' else code;cmd=[sys.executable,folder+'/tools/worker_context_controls.py','--root',str(outer),'--plan',str(outer/'job_plan.json'),'--kind',kind,'--variant',variant,'--worker',str(i),'--workers',str(n),'--deadline',str(deadline)]
   if variant!='student':
    cmd+=['--error-root',str(revision/'errors')]
    if (outer/(name+'_progress.json')).exists():assert kind=='source' and variant=='fixed';cmd+=['--resume']
   tasks.append(dict(name=name,cmd=cmd))
 atomic_json(revision/'controller_progress.json',dict(status='running',new_jobs=len(tasks),retained_jobs=retained,original_deadline_kept=True))
 queue(revision,tasks,recovery['gpus'],deadline,'context_controls',minimum_free_MiB=1792)
 for name,pid in recovery['existing_live_jobs'].items():
  while not (outer/(name+'_manifest.json')).exists():
   assert time.time()<deadline
   try:os.kill(pid,0)
   except ProcessLookupError:raise RuntimeError('Retained own worker exited without completion: '+name)
   time.sleep(5)
 records=[];counts={k:0 for k in ['query_scalar_forwards','full_input_scalar_slots','Gaussian_covariance_factorizations','context_ridge_factorizations','cached_prior_query_applications']}
 for i in range(n):
  for variant,kind in [('fixed','source'),('student','source'),('rolling','source'),('student','target'),('rolling','target')]:
   name=kind+'_'+variant+'_w'+str(i);m=json.loads((outer/(name+'_manifest.json')).read_text());assert m['status']=='complete' and m['parameters_unchanged'];records+=m['records']
   for k in counts:counts[k]+=m[k]
 assert len(records)==288+4*len(plan['target_rows']) and len(set(records))==len(records);assert not (outer/'manifest.json').exists()
 atomic_json(outer/'manifest.json',dict(status='complete',files=[dict(path=p,SHA256=sha256(outer/p),scores_SHA256=json.loads((outer/p).read_text())['scores_SHA256']) for p in records],**counts,plan_SHA256=sha256(outer/'job_plan.json'),optimizer_updates=0,target_updates=0,validation_or_locked_arrays_opened=False,recovery='recovery_r1',extra_failed_prior_query_applications=recovery['extra_failed_prior_query_applications']))
 atomic_json(revision/'controller_progress.json',dict(status='complete',records=len(records),**counts,optimizer_updates=0,target_updates=0,extra_failed_prior_query_applications=recovery['extra_failed_prior_query_applications'],original_deadline_kept=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--outer',type=Path,required=True);p.add_argument('--revision',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
 try:main(a)
 except Exception:atomic_json(a.revision/'controller_error.json',dict(error=traceback.format_exc()));atomic_json(a.revision/'controller_progress.json',dict(status='needs_attention',new_dispatch_stopped=True));raise
