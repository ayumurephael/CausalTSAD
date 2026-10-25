"""Original4 metrics + AP/ROC/FPR, two explicit input and scoring protocols, TSAD only."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,json,pickle,sys,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from causal_tsad.io import atomic_json,sha256
from causal_tsad.frozen_detector import methods
METRICS=['Affiliation-F','F1_T','Standard-F1','VUS-PR']

def one(payload):
 root,benchmark,metric_code,row=payload;root=Path(root);benchmark=Path(benchmark);sys.path.insert(0,metric_code)
 from evaluate_table1 import compute_metrics
 from utils.slidingWindows import find_length_rank
 from sklearn.metrics import average_precision_score,roc_auc_score
 assert row['split']=='development';dest=root/'metrics'/(row['id']+'.json');assert not dest.exists()
 with (benchmark/'data'/row['shard']).open('rb') as f:sample=pickle.load(f)[row['offset']]
 assert sample['system_id']==row['system_id'];y=np.asarray(sample['labels'],int);two=len(np.unique(y))==2
 periods={policy:max(1,int(find_length_rank(np.asarray(sample['time_series'],dtype=np.float32 if policy=='causal_compat' else None),rank=1))) for policy in ['native','causal_compat']}
 thresholds=json.loads((root/'calibration.json').read_text())['thresholds'];out={};memo={}
 scores=root/'scores'/(row['id']+'.npz');record=json.loads((root/'inference'/(row['id']+'.json')).read_text());assert sha256(scores)==record['scores_SHA256']
 with np.load(scores) as f,(root/'logs'/('metric_'+row['id']+'.log')).open('w') as log,redirect_stdout(log),redirect_stderr(log):
  for method,weight in methods(row['channels']):
   for policy in ['native','causal_compat']:
    for representation in ['probability','logit']:
     score=np.asarray(f[f'{method}__{policy}__observed__{representation}']);normal=np.asarray(f[f'{method}__{policy}__normal__{representation}'])
     period=periods[policy];key=str(period)+'/'+sha256_array(score);threshold=thresholds[weight+'/'+representation]
     if key not in memo:
      m={k:None for k in METRICS}
      if two:
       raw=compute_metrics(score,y,period);m={k:float(raw[k]) for k in METRICS}
      memo[key]=dict(metrics=m,AP=float(average_precision_score(y,score)) if two else None,ROC_AUC=float(roc_auc_score(y,score)) if two else None)
     out[method+'/'+policy+'/'+representation]=dict(memo[key],normal_FPR_source99=float(np.mean(normal>threshold)),normal_FPR_at_probability_0_5=float(np.mean(normal>(.5 if representation=='probability' else 0))),
       source_threshold=threshold,checkpoint=weight,representation=representation,normalization=policy)
 result=dict(system_id=row['system_id'],track=row['track'],channels=row['channels'],family=row['family'],event_types=row['event_types'],length=row['length'],periods=periods,
  pilot100=row['pilot100'],methods=out,RCA_evaluated=False,split='development',scores_SHA256=record['scores_SHA256'],parameter_updates=0)
 atomic_json(dest,result);return result

def sha256_array(x):
 import hashlib
 return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['root','benchmark','metric-code','jobs']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--workers',type=int,default=8);a=p.parse_args();rows=json.loads(a.jobs.read_text());completed=0;failed=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(one,(str(a.root),str(a.benchmark),str(a.metric_code),r)):r for r in rows}
  for future in as_completed(futures):
   row=futures[future]
   try:future.result();completed+=1
   except Exception:failed.append(row['system_id']);atomic_json(a.root/'errors'/('metric_'+row['id']+'.json'),dict(error=traceback.format_exc()))
   atomic_json(a.root/'metric_progress.json',dict(status='running',completed=completed,total=len(rows),failed=failed))
 atomic_json(a.root/'metric_progress.json',dict(status='complete' if not failed else 'needs_attention',completed=completed,total=len(rows),failed=failed));assert not failed
