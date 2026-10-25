import argparse,json,sys,os,subprocess,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json
def main(a):
 r=a.root;code=Path(__file__).resolve().parents[1];plan=json.loads(a.plan.read_text());pending=[dict(name=f'{mode}_s{seed}',mode=mode,seed=seed) for seed in [348,349,350] for mode in ['none','parents_current']];active={};complete=[];began=time.monotonic()
 with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 status=dict(status='running',stage='source_only_measurement_isolation',pid=os.getpid(),optimizer_updates=0)
 while pending or active:
  if time.monotonic()-began>2*3600:raise TimeoutError('Fixed2h diagnostic deadline')
  for gpu in plan['gpus']:
   if gpu in active or not pending:continue
   job=pending.pop(0);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTHONUNBUFFERED']='1'
   cmd=[sys.executable,str(code/'tools/probe_c05.py'),'--source',str(a.source),'--weights',str(r/'weights'),'--config',str(code/'experiments/c05_isolation_v1.json'),'--output',str(r/'runs'/job['name']),
    '--mode',job['mode'],'--seed',str(job['seed']),'--local-commit',plan['local_commit'],'--github-commit',plan['github_commit']]
   with (r/'logs'/(job['name']+'.log')).open('x') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
   active[gpu]=dict(job=job,process=p)
  status.update(active=[dict(gpu=g,pid=e['process'].pid,run=e['job']['name']) for g,e in active.items()],pending=len(pending),completed=complete);atomic_json(r/'controller_progress.json',status);time.sleep(3)
  for gpu,e in list(active.items()):
   ret=e['process'].poll()
   if ret is None:continue
   if ret!=0:raise RuntimeError(e['job']['name']+' failed, no new C05jobs')
   complete.append(e['job']['name']);del active[gpu]
 status.update(status='complete',stage='complete',active=[],pending=0,completed=complete);atomic_json(r/'controller_progress.json',status)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
 try:main(a)
 except Exception:
  p=a.root/'controller_progress.json';z=json.loads(p.read_text()) if p.exists() else {};z.update(status='needs_attention',error=traceback.format_exc(),new_jobs_stopped=True);atomic_json(p,z);raise
