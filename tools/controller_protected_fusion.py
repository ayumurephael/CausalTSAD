"""Finite source phases: base caching, then frozen teacher features and fusion training."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256

def queue(root,tasks,gpus,deadline,stage,minimum_free_MiB=None):
 active={};done=[];failed=[]
 while tasks or active:
  assert time.time()<deadline,'Fixed source-stage deadline'
  for gpu,item in list(active.items()):
   rc=item['p'].poll()
   if rc is not None:
    item['log'].close();(done if rc==0 else failed).append(dict(name=item['task']['name'],exit_code=rc));del active[gpu]
  if failed:
   atomic_json(root/(stage+'_controller_progress.json'),dict(status='needs_attention',stage=stage,failed=failed,completed=done,pending=[t['name'] for t in tasks],active=[dict(gpu=g,name=z['task']['name'],pid=z['p'].pid) for g,z in active.items()]));raise RuntimeError('Stop new own dispatch after failure')
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode();free={int(v.split(',')[0]):float(v.split(',')[1]) for v in raw.strip().splitlines()}
  for gpu in gpus:
   if gpu in active or not tasks or free[gpu]<(minimum_free_MiB if minimum_free_MiB is not None else 1536 if stage=='fusion' else 2048):continue
   task=tasks.pop(0);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);log=(root/'logs'/(stage+'_'+task['name']+'.log')).open('x');p=subprocess.Popen(task['cmd'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True);active[gpu]=dict(p=p,task=task,log=log)
  atomic_json(root/(stage+'_controller_progress.json'),dict(status='running',stage=stage,pid=os.getpid(),completed=done,pending=[t['name'] for t in tasks],active=[dict(gpu=g,name=z['task']['name'],pid=z['p'].pid) for g,z in active.items()],failed=[]))
  if tasks or active:time.sleep(5)
 atomic_json(root/(stage+'_controller_progress.json'),dict(status='complete',stage=stage,completed=done,active=[],pending=[],failed=[]))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['base','source','fusion'],required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();r=a.root;code=Path(__file__).resolve().parents[1];plan=json.loads(a.plan.read_text());deadline=time.time()+8*3600
 cfg=json.loads((code/'experiments/protected_fusion_v2.json').read_text());assert all(k in cfg for k in ['data_seed','learning_rate','weight_decay','old_source_trust_weight'])
 with (r/(a.stage+'_controller.lock')).open('x') as f:f.write(str(os.getpid()))
 if a.stage=='base':
  tasks=[dict(name='worker'+str(i),cmd=[sys.executable,str(code/'tools/cache_fusion_base.py'),'--plan',str(r/'source_plan.json'),'--root',str(r),'--original-code',plan['original_runtime'],'--weights',plan['weights'],
   '--worker',str(i),'--workers',str(len(plan['gpus']))]) for i in range(len(plan['gpus']))]
  queue(r,tasks,plan['gpus'],time.time()+2*3600,'base');raise SystemExit(0)
 assert json.loads((r/'preflight.json').read_text())['status']=='passed'
 tasks=[]
 for seed in plan['seeds']:
  for mode in ['degree','true','wrong']:
   name=mode+'_s'+str(seed);tasks.append(dict(name=name,cmd=[sys.executable,str(code/'tools/cache_fusion_teacher.py'),'--root',str(r),'--plan',str(r/'source_plan.json'),'--checkpoint',str(r/'frozen_c04'/(name+'.pth')),'--output',str(r/'source_cache'/name)]))
 if a.stage=='source':queue(r,tasks,plan['gpus'],time.time()+2*3600,'features')
 tasks=[]
 for seed in plan['seeds']:
  for arm in ['basecal','degree','true','wrong']:
   alias='degree' if arm=='basecal' else arm;cache=r/'source_cache'/(alias+'_s'+str(seed));m=json.loads((cache/'manifest.json').read_text());assert m['status']=='complete' and m['parameters_unchanged'] and not m['target_arrays_opened']
   name=arm+'_s'+str(seed);tasks.append(dict(name=name,cmd=[sys.executable,str(code/'tools/train_protected_fusion.py'),'--cache',str(cache),'--config',str(code/'experiments/protected_fusion_v2.json'),'--output',str(r/'runs'/name),'--arm',arm,'--seed',str(seed),
    '--local-commit',plan['local_commit'],'--github-commit',plan['github_commit'],'--deadline',str(deadline)]))
 queue(r,tasks,plan['gpus'],deadline,'fusion')
 atomic_json(r/'controller_progress.json',dict(status='complete',stage='source_training_complete',source_only=True,target_arrays_opened=False))
