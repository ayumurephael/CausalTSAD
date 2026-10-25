"""Disjoint seed allocation on A100/H100; reuses SHA-verified C09 cache, no encoder work."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256

def main(a):
 r=a.root;code=Path(__file__).resolve().parents[1];config=code/'experiments/c03_specificity_v3.json';cfg=json.loads(config.read_text());plan=json.loads(a.plan.read_text());start_time=time.monotonic()
 with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 status=dict(status='running',stage='G0',pid=os.getpid(),active=[],pending=4*len(plan['seeds']),completed=[]);atomic_json(r/'controller_progress.json',status)
 def launch(cmd,name,gpu):
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTHONUNBUFFERED']='1'
  with (r/'logs'/(name+'.log')).open('x') as f:return subprocess.Popen([sys.executable]+list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
 p=launch([code/'tests/preflight_c03b.py','--output',r/'preflight.json'],'preflight',plan['gpus'][0]);assert p.wait(timeout=180)==0
 assert json.loads((r/'preflight.json').read_text())['status']=='passed'
 for mode in cfg['graph_modes']:
  cache=a.cache/mode;m=json.loads((cache/'manifest.json').read_text());assert sha256(cache/'manifest.json')==plan['cache_manifest_sha256'][mode] and m['status']=='complete'
  assert m['teacher_checkpoint_sha256']==cfg['teacher_checkpoint_sha256'] and m['source_manifest_sha256']==cfg['source_manifest_sha256']
  for row in m['files']:assert sha256(cache/row['path'])==row['SHA256']
 pending=[dict(name=f'{mode}__{arm}_s{seed}',mode=mode,arm=arm,seed=seed) for seed in plan['seeds'] for mode in cfg['graph_modes'] for arm in cfg['arms']];active={};completed=[]
 def begin(gpu,job,stage):
  out=r/'runs'/job['name'];cache=a.cache/job['mode']
  if stage=='train':cmd=[code/'tools/train_c03b.py','--cache',cache,'--config',config,'--output',out,'--arm',job['arm'],'--seed',job['seed'],'--local-commit',plan['local_commit'],'--github-commit',plan['github_commit']]
  else:cmd=[code/'tools/evaluate_c03b.py','--cache',cache,'--run',out,'--benchmark',a.benchmark,'--metric-code',a.metric_code]
  active[gpu]=dict(job=job,stage=stage,p=launch(cmd,job['name']+'_'+stage,gpu),started=time.monotonic())
 while pending or active:
  if time.monotonic()-start_time>cfg['max_campaign_hours']*3600:raise TimeoutError('Fixed10h campaign endpoint')
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode();free={int(s.split(',')[0]):float(s.split(',')[1]) for s in raw.strip().splitlines()}
  for gpu in plan['gpus']:
   if gpu not in active and pending and free.get(gpu,0)>=3072:begin(gpu,pending.pop(0),'train')
  status.update(stage='train_and_evaluate',active=[dict(gpu=g,pid=v['p'].pid,run=v['job']['name'],stage=v['stage']) for g,v in active.items()],pending=len(pending),completed=completed);atomic_json(r/'controller_progress.json',status)
  time.sleep(3)
  for gpu,v in list(active.items()):
   rc=v['p'].poll()
   if rc is None:
    if time.monotonic()-v['started']>cfg['max_stage_hours']*3600:raise TimeoutError('Fixed3h technical stage limit')
    continue
   if rc!=0:raise RuntimeError(v['job']['name']+' '+v['stage']+' failed; new local C03v3 jobs stopped')
   if v['stage']=='train':begin(gpu,v['job'],'evaluate')
   else:completed.append(v['job']['name']);del active[gpu]
 status.update(status='complete',stage='complete',active=[],pending=0,completed=completed);atomic_json(r/'controller_progress.json',status)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in ['root','plan','cache','benchmark','metric-code']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args()
 try:main(a)
 except Exception:
  path=a.root/'controller_progress.json';z=json.loads(path.read_text()) if path.exists() else {};z.update(status='needs_attention',error=traceback.format_exc(),new_local_jobs_stopped=True);atomic_json(path,z);raise
