"""Finite zero-update H100 queue, separate from all original C08 artifacts."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json
def main(a):
 r=a.root;code=Path(__file__).resolve().parents[1];cfgpath=code/'experiments/c08_coupling_audit_v1.json';cfg=json.loads(cfgpath.read_text());plan=json.loads(a.plan.read_text());start=time.monotonic()
 with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 pending=[dict(arm=arm,seed=seed,name=f'{arm}_s{seed}') for seed in cfg['seeds'] for arm in cfg['arms']];active={};complete=[]
 status=dict(status='running',stage='frozen_coupling_and_gradient_audit',pid=os.getpid(),optimizer_updates=0);atomic_json(r/'controller_progress.json',status)
 while pending or active:
  if time.monotonic()-start>cfg['max_campaign_hours']*3600:raise TimeoutError('Fixed4h diagnostic deadline')
  for gpu in plan['gpus']:
   if gpu in active or not pending:continue
   job=pending.pop(0);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTHONUNBUFFERED']='1'
   cmd=[sys.executable,str(code/'tools/coupling_audit_c08.py'),'--parent',str(a.parent),'--cache',str(a.parent/'cache'),'--config',str(cfgpath),'--output',str(r/'runs'/job['name']),
    '--arm',job['arm'],'--seed',str(job['seed']),'--local-commit',plan['local_commit'],'--github-commit',plan['github_commit']]
   with (r/'logs'/(job['name']+'.log')).open('x') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
   active[gpu]=dict(job=job,process=p)
  status.update(active=[dict(gpu=g,pid=e['process'].pid,run=e['job']['name']) for g,e in active.items()],completed=complete,pending=len(pending));atomic_json(r/'controller_progress.json',status);time.sleep(3)
  for gpu,e in list(active.items()):
   rc=e['process'].poll()
   if rc is None:continue
   if rc!=0:raise RuntimeError(e['job']['name']+' failed; no new diagnosis jobs')
   complete.append(e['job']['name']);del active[gpu]
 status.update(status='complete',stage='complete',active=[],pending=0,completed=complete);atomic_json(r/'controller_progress.json',status)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
 try:main(a)
 except Exception:
  p=a.root/'controller_progress.json';z=json.loads(p.read_text()) if p.exists() else {};z.update(status='needs_attention',error=traceback.format_exc(),new_jobs_stopped=True);atomic_json(p,z);raise
