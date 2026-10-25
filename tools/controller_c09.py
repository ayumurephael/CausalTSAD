"""A100-only independent C09 queue; does not depend on C08 training/results."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256
def main(a):
 r=a.root;code=Path(__file__).resolve().parents[1];config=code/'experiments/c09_tail_v1.json';cfg=json.loads(config.read_text());plan=json.loads(a.plan.read_text());started=time.monotonic()
 with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 status=dict(status='running',stage='G0',pid=os.getpid(),active=[],pending=12,completed=[]);atomic_json(r/'controller_progress.json',status)
 def launch(cmd,name,gpu):
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTHONUNBUFFERED']='1'
  with (r/'logs'/(name+'.log')).open('x') as f:return subprocess.Popen([sys.executable]+list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
 def wait(children,hours):
  start=time.monotonic()
  while any(p.poll() is None for p in children):
   if any(p.poll() not in [None,0] for p in children):raise RuntimeError('C09 technical stage failed, no new jobs')
   if time.monotonic()-start>hours*3600:raise TimeoutError('Fixed technical deadline')
   time.sleep(3)
  assert all(p.returncode==0 for p in children)
 gpus=plan['gpus'];pre=launch([code/'tests/preflight_c09.py','--output',r/'preflight.json'],'preflight',gpus[0]);wait([pre],1);assert json.loads((r/'preflight.json').read_text())['status']=='passed'
 cache=r/'cache';cache.mkdir(exist_ok=False);status['stage']='frozen_teacher_cache';atomic_json(r/'controller_progress.json',status);children=[]
 for i,gpu in enumerate(gpus):children.append(launch([code/'tools/cache_c09.py','--base',a.base,'--source',a.source,'--benchmark',a.benchmark,'--teacher',a.teacher,
  '--config',config,'--output',cache,'--worker',i,'--workers',len(gpus)],'cache_'+str(i),gpu))
 wait(children,3);refs=json.loads((a.base/'manifest.json').read_text());rows=[]
 for row in refs['files']:
  z=json.loads((cache/'records'/(row['name']+'.json')).read_text());assert sha256(cache/z['path'])==z['SHA256'];rows.append(z)
 audits=[json.loads((cache/f'worker_{i}.json').read_text()) for i in range(len(gpus))];assert all(z['status']=='complete' and z['teacher_parameter_sha_before']==z['teacher_parameter_sha_after'] for z in audits)
 assert len(rows)==535 and sum(r['ready_scalars'] for r in rows if r['kind']=='train')==cfg['ready_query_scalars_per_pass']
 atomic_json(cache/'manifest.json',dict(status='complete',files=rows,workers=audits,teacher_checkpoint_sha256=sha256(a.teacher),source_manifest_sha256=sha256(a.source/'manifest.json'),
  base_reference_manifest_sha256=sha256(a.base/'manifest.json'),teacher_query_scalars=sum(z['teacher_query_scalars'] for z in audits),teacher_Linear_forward_FLOPs=sum(z['teacher_Linear_forward_FLOPs'] for z in audits),
  code_local_commit=plan['local_commit'],code_github_commit=plan['github_commit'],encoder_forwards=0,optimizer_updates=0,target_validation_or_locked_arrays_opened=False))
 pending=[dict(name=f'{arm}_s{seed}',arm=arm,seed=seed) for seed in cfg['seeds'] for arm in cfg['arms']];active={};completed=[]
 def start(gpu,job,stage):
  out=r/'runs'/job['name']
  if stage=='train':cmd=[code/'tools/train_c09.py','--cache',cache,'--output',out,'--config',config,'--arm',job['arm'],'--seed',job['seed'],'--local-commit',plan['local_commit'],'--github-commit',plan['github_commit']]
  else:cmd=[code/'tools/evaluate_c09.py','--cache',cache,'--run',out,'--benchmark',a.benchmark,'--metric-code',a.metric_code]
  active[gpu]=dict(job=job,stage=stage,process=launch(cmd,job['name']+'_'+stage,gpu),started=time.monotonic())
 while pending or active:
  if time.monotonic()-started>cfg['max_campaign_hours']*3600:raise TimeoutError('C09 campaign10h deadline')
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode();free={int(s.split(',')[0]):float(s.split(',')[1]) for s in raw.strip().splitlines()}
  for gpu in gpus:
   if gpu not in active and pending and free.get(gpu,0)>=6144:start(gpu,pending.pop(0),'train')
  status.update(stage='train_and_evaluate',active=[dict(gpu=g,pid=e['process'].pid,stage=e['stage'],run=e['job']['name']) for g,e in active.items()],pending=len(pending),completed=completed);atomic_json(r/'controller_progress.json',status);time.sleep(3)
  for gpu,e in list(active.items()):
   rc=e['process'].poll()
   if rc is None:
    if time.monotonic()-e['started']>cfg['max_stage_hours']*3600:raise TimeoutError('C09 worker deadline')
    continue
   if rc!=0:raise RuntimeError(e['job']['name']+' '+e['stage']+' failed; new C09 jobs stopped')
   if e['stage']=='train':start(gpu,e['job'],'evaluate')
   else:completed.append(e['job']['name']);del active[gpu]
 status.update(status='complete',stage='complete',active=[],pending=0,completed=completed);atomic_json(r/'controller_progress.json',status)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
 p.add_argument('--source',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--teacher',type=Path,required=True);p.add_argument('--metric-code',type=Path,required=True);a=p.parse_args()
 try:main(a)
 except Exception:
  p=a.root/'controller_progress.json';s=json.loads(p.read_text()) if p.exists() else {};s.update(status='needs_attention',error=traceback.format_exc(),new_jobs_stopped=True);atomic_json(p,s);raise
