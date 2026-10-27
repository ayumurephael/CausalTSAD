"""Immutable finite source queue, no target input and no intervention on other processes."""
import argparse,json,subprocess,os,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['root','source','plan']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();r=a.root;plan=json.loads(a.plan.read_text());code=Path(__file__).resolve().parents[1]
 assert json.loads((r/'preflight.json').read_text())['status']=='passed' and json.loads((r/'predictor_preflight.json').read_text())['status']=='passed'
 assert sha256(a.source/'manifest.json')==plan['source_manifest_SHA256']
 with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 tasks=[dict(mode=mode,seed=s,name=mode+'_s'+str(s)) for s in plan['seeds'] for mode in ['true','degree','wrong']];active={};done=[];failed=[];deadline=time.time()+8*3600
 while tasks or active:
  if time.time()>deadline:raise TimeoutError('Fixed8h campaign, active workers share deadline')
  for gpu,z in list(active.items()):
   rc=z['p'].poll()
   if rc is not None:
    z['log'].close();(done if rc==0 else failed).append(dict(name=z['task']['name'],exit_code=rc));del active[gpu]
  if failed:
   atomic_json(r/'controller_progress.json',dict(status='needs_attention',failed=failed,completed=done,pending=tasks,active=[dict(gpu=g,name=z['task']['name'],pid=z['p'].pid) for g,z in active.items()]));raise RuntimeError('Stop own new dispatch; retain active and complete runs')
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode();free={int(s.split(',')[0]):float(s.split(',')[1]) for s in raw.strip().splitlines()}
  for gpu in plan['gpus']:
   if not tasks or gpu in active or free[gpu]<6144:continue
   task=tasks.pop(0);out=r/'runs'/task['name'];assert not out.exists();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
   cmd=[sys.executable,str(code/'tools/train_causal_contrast.py'),'--source',str(a.source),'--source-sha256',plan['source_manifest_SHA256'],'--config',str(code/'experiments/matched_pairs_v1.json'),'--output',str(out),
    '--mode',task['mode'],'--seed',str(task['seed']),'--local-commit',plan['local_commit'],'--github-commit',plan['github_commit'],'--deadline',str(deadline)]
   log=(r/'logs'/(task['name']+'.log')).open('x');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True);active[gpu]=dict(p=proc,task=task,log=log)
  atomic_json(r/'controller_progress.json',dict(status='running',pid=os.getpid(),active=[dict(gpu=g,name=z['task']['name'],pid=z['p'].pid) for g,z in active.items()],pending=tasks,completed=done,failed=failed))
  if tasks or active:time.sleep(5)
 atomic_json(r/'controller_progress.json',dict(status='complete',pid=os.getpid(),active=[],pending=[],completed=done,failed=[]))
