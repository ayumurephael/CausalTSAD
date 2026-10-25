"""Cached-only factorial source probes, bounded, no target access."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controller_protected_fusion import queue
from causal_tsad.io import atomic_json
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();r=a.root;code=Path(__file__).resolve().parents[1];plan=json.loads(a.plan.read_text());cfg=json.loads((code/'experiments/score_evidence_v3.json').read_text());assert all(k in cfg for k in ['data_seed','learning_rate','pair_loss_weight']);deadline=time.time()+4*3600
 with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 tasks=[]
 for seed in plan['seeds']:
  combinations=[(arm,v) for arm in ['true','degree','wrong'] for v in cfg['variants']]+[('basecal','free_point')]
  for arm,variant in combinations:
   name=arm+'_'+variant+'_s'+str(seed);cache=Path(plan['cache_root'])/('degree' if arm=='basecal' else arm)/'unused'
   cache=Path(plan['cache_root'])/(('degree' if arm=='basecal' else arm)+'_s'+str(seed));assert json.loads((cache/'manifest.json').read_text())['status']=='complete'
   tasks.append(dict(name=name,cmd=[sys.executable,str(code/'tools/train_score_evidence.py'),'--cache',str(cache),'--config',str(code/'experiments/score_evidence_v3.json'),'--output',str(r/'runs'/name),'--arm',arm,'--variant',variant,'--seed',str(seed),'--deadline',str(deadline),'--local-commit',plan['local_commit'],'--github-commit',plan['github_commit']]))
 queue(r,tasks,plan['gpus'],deadline,'fusion');atomic_json(r/'controller_progress.json',dict(status='complete',runs=len(plan['seeds'])*13,source_only=True,target_arrays_opened=False))
