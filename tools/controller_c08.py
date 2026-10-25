"""Bounded G0 -> immutable cache -> all24 matched pilot runs and evaluations."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256

def main(a):
 r=a.root;code=Path(__file__).resolve().parents[1];cfgpath=code/'experiments/c08_joint_v1.json';cfg=json.loads(cfgpath.read_text());plan=json.loads(a.plan.read_text());began=time.monotonic()
 with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 status=dict(status='running',stage='G0',pid=os.getpid(),active=[],pending=24,completed=[]);atomic_json(r/'controller_progress.json',status)
 def launch(cmd,name,gpu):
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTHONUNBUFFERED']='1'
  with (r/'logs'/(name+'.log')).open('x') as f:return subprocess.Popen([sys.executable]+list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
 def wait_stage(children,hours):
  start=time.monotonic()
  while any(p.poll() is None for p in children):
   if any(p.poll() not in [None,0] for p in children):raise RuntimeError('Technical stage failed; do not start new work')
   if time.monotonic()-start>hours*3600:raise TimeoutError('Technical stage deadline')
   time.sleep(3)
  assert all(p.returncode==0 for p in children)
 gpus=plan['gpus'];p=launch([code/'tests/preflight_c08.py','--output',r/'preflight.json'],'preflight',gpus[0]);wait_stage([p],1)
 assert json.loads((r/'preflight.json').read_text())['status']=='passed'
 cache=r/'cache';cache.mkdir(exist_ok=False);workers=[];status['stage']='frozen_teacher_cache';atomic_json(r/'controller_progress.json',status)
 for i,gpu in enumerate(gpus):
  workers.append(launch([code/'tools/cache_c08.py','--upstream',a.upstream,'--teacher',a.teacher,'--source',a.source,'--config',cfgpath,'--output',cache,
   '--worker',i,'--workers',len(gpus)],'cache_'+str(i),gpu))
 wait_stage(workers,2);upstream=json.loads((a.upstream/'manifest.json').read_text());rows=[]
 for task in upstream['files']:
  row=json.loads((cache/'records'/(task['name']+'.json')).read_text());assert sha256(cache/row['path'])==row['SHA256'];rows.append(row)
 audits=[json.loads((cache/f'worker_{i}.json').read_text()) for i in range(len(gpus))]
 assert all(z['status']=='complete' and z['teacher_parameter_sha_before']==z['teacher_parameter_sha_after'] for z in audits)
 assert len(rows)==1901 and sum(z['scalars'] for z in rows if z['kind']=='train' and z['channels']>1)==cfg['original_multivariate_scalars_per_pass']
 atomic_json(cache/'manifest.json',dict(status='complete',files=rows,workers=audits,feature_dim=cfg['feature_dim'],
  source_manifest_sha256=sha256(a.source/'manifest.json'),upstream_manifest_sha256=sha256(a.upstream/'manifest.json'),teacher_checkpoint_sha256=sha256(a.teacher),
  teacher_query_scalars=sum(z['teacher_query_scalars'] for z in audits),new_encoder_forwards=0,optimizer_updates=0,
  C1_teacher_skipped=True,code_local_commit=plan['local_commit'],code_github_commit=plan['github_commit'],target_validation_or_locked_arrays_opened=False,
  feature_compression='signed log1p standardized residual, log1p squared residual,logvariance,signedlog1p mean/query,coverage,causal moving averages8/32/64 of energy/signed residual; feature-only clipping+-50',
  teacher_exposure='C01none seed348 normal/source anomaly fusion trained previously, frozen univariate density only reused; same teacher across all graph/root/seed arms'))
 pending=[dict(name=f'{arm}_s{seed}',arm=arm,seed=seed) for seed in cfg['seeds'] for arm in cfg['arms']];active={};complete=[]
 def start(gpu,job,stage):
  out=r/'runs'/job['name']
  if stage=='train':cmd=[code/'tools/train_c08.py','--cache',cache,'--config',cfgpath,'--output',out,'--arm',job['arm'],'--seed',job['seed'],'--local-commit',plan['local_commit'],'--github-commit',plan['github_commit']]
  else:cmd=[code/'tools/evaluate_c08.py','--cache',cache,'--run',out,'--benchmark',a.benchmark,'--metric-code',a.metric_code]
  active[gpu]=dict(job=job,stage=stage,process=launch(cmd,job['name']+'_'+stage,gpu),start=time.monotonic())
 while pending or active:
  if time.monotonic()-began>cfg['max_campaign_hours']*3600:raise TimeoutError('Fixed8h campaign deadline')
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode();free={int(s.split(',')[0]):float(s.split(',')[1]) for s in raw.strip().splitlines()}
  for gpu in gpus:
   if gpu not in active and pending and free.get(gpu,0)>=10240:start(gpu,pending.pop(0),'train')
  status.update(stage='train_and_evaluate',pending=len(pending),completed=complete,active=[dict(gpu=g,pid=e['process'].pid,stage=e['stage'],run=e['job']['name']) for g,e in active.items()]);atomic_json(r/'controller_progress.json',status);time.sleep(3)
  for gpu,e in list(active.items()):
   rc=e['process'].poll()
   if rc is None:
    if time.monotonic()-e['start']>3*3600:raise TimeoutError('Worker3h technical deadline')
    continue
   if rc!=0:raise RuntimeError(e['job']['name']+' '+e['stage']+' failed; new jobs stopped, running children may finish')
   if e['stage']=='train':start(gpu,e['job'],'evaluate')
   else:complete.append(e['job']['name']);del active[gpu]
 status.update(status='complete',stage='complete',active=[],pending=0,completed=complete);atomic_json(r/'controller_progress.json',status)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
 p.add_argument('--teacher',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--metric-code',type=Path,required=True);a=p.parse_args()
 try:main(a)
 except Exception:
  p=a.root/'controller_progress.json';state=json.loads(p.read_text()) if p.exists() else {};state.update(status='needs_attention',error=traceback.format_exc(),new_jobs_stopped=True);atomic_json(p,state);raise
