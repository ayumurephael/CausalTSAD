"""Bounded, GPU-memory-aware independent run queue; never signals other work."""
import argparse,json,os,subprocess,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
 r=a.root;plan=json.loads(a.plan.read_text());code=Path(__file__).resolve().parents[1];cfg=json.loads((code/'experiments/raw_context_v4.json').read_text());r.mkdir(exist_ok=True)
 with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 assert json.loads((r/'preflight.json').read_text())['status']=='passed'
 todo=[dict(arm=arm,seed=seed,name=arm+'_s'+str(seed)) for seed in plan['seeds'] for arm in cfg['arms']];active={};done=[];failed=[];start=time.monotonic();deadline=time.time()+cfg['campaign_hours']*3600
 while todo or active:
  assert time.monotonic()-start<cfg['campaign_hours']*3600,'Fixed campaign deadline'
  for gpu,z in list(active.items()):
   rc=z['process'].poll()
   if rc is not None:
    z['log'].close();entry=dict(name=z['row']['name'],exit_code=rc)
    (done if rc==0 else failed).append(entry);del active[gpu]
  if failed:
   atomic_json(r/'controller_progress.json',dict(status='needs_attention',pid=os.getpid(),failed=failed,active=[dict(gpu=g,name=z['row']['name'],pid=z['process'].pid) for g,z in active.items()],pending=todo,completed=done));raise RuntimeError('Stop new dispatch after own run failure; retain already active jobs')
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode()
  free={int(x.split(',')[0]):float(x.split(',')[1]) for x in raw.strip().splitlines()}
  for gpu in plan['gpus']:
   if gpu in active or not todo or free[gpu]<6144:continue
   row=todo.pop(0);out=r/'runs'/row['name'];assert not out.exists();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
   cmd=[sys.executable,str(code/'tools/train_raw_context.py'),'--source',str(a.source),'--config',str(code/'experiments/raw_context_v4.json'),'--output',str(out),
    '--arm',row['arm'],'--seed',str(row['seed']),'--local-commit',plan['local_commit'],'--github-commit',plan['github_commit'],'--campaign-deadline',str(deadline)]
   log=(r/'logs'/(row['name']+'.log')).open('x');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
   active[gpu]=dict(process=proc,row=row,log=log)
  atomic_json(r/'controller_progress.json',dict(status='running',pid=os.getpid(),active=[dict(gpu=g,name=z['row']['name'],pid=z['process'].pid) for g,z in active.items()],pending=todo,completed=done,failed=failed))
  if todo or active:time.sleep(5)
 atomic_json(r/'controller_progress.json',dict(status='complete',pid=os.getpid(),completed=done,failed=failed,active=[],pending=[]))
