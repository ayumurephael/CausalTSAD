"""Finite warm-start clean-retention versus benign-nuisance continuation."""
import sys,json,argparse,os,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controller_protected_fusion import queue
from causal_tsad.io import atomic_json
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();r=a.root;plan=json.loads(a.plan.read_text());code=Path(__file__).resolve().parents[1];deadline=time.time()+8*3600
 with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 tasks=[]
 for seed in plan['seeds']:
  for mode in ['true','degree','wrong']:
   for arm in ['continued_clean','augmented']:
    name=mode+'_'+arm+'_s'+str(seed);tasks.append(dict(name=name,cmd=[sys.executable,str(code/'tools/train_robust_relations.py'),'--checkpoint',str(Path(plan['weights'])/(mode+'_s'+str(seed)+'.pth')),'--source',plan['source'],'--output',str(r/'runs'/name),'--mode',mode,'--arm',arm,'--seed',str(seed),'--deadline',str(deadline),'--local-commit',plan['local_commit'],'--github-commit',plan['github_commit']]))
 # Source queue's2GiB default is intentionally not used for4GiB backprop jobs.
 plan_gpus=plan['gpus'];queue(r,tasks,plan_gpus,deadline,'robust_training',minimum_free_MiB=6144);atomic_json(r/'controller_progress.json',dict(status='complete',runs=6*len(plan['seeds']),target_arrays_opened=False))
