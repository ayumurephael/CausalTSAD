"""Original four formulas, same >=320 points, plus AP/AUC/source-normal FPR.

TimeRCD metrics are reused only after score-array SHA equality. No TimeRCD RCA.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,argparse,json,pickle,traceback,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from causal_tsad.io import atomic_json,sha256,array_sha
def one(args):
 root,planfile,row=args;root=Path(root);plan=json.loads(Path(planfile).read_text());sys.path.insert(0,plan['metric_code']);from evaluate_table1 import compute_metrics
 from sklearn.metrics import average_precision_score,roc_auc_score
 assert row['split']=='development' and row['pilot100']
 with (Path(plan['benchmark'])/'data'/row['shard']).open('rb') as f:sample=pickle.load(f)[row['offset']]
 assert sample['system_id']==row['system_id'] and array_sha(np.asarray(sample['labels']))==row['labels_sha256'];y=np.asarray(sample['labels'],int)[320:];two=len(np.unique(y))==2;out=root/'development';record=json.loads((out/'inference'/(row['id']+'.json')).read_text());p=out/'scores'/(row['id']+'.npz');assert sha256(p)==record['scores_SHA256'];methods={};memo={};old=Path(plan['reference'])/'development'/('s'+str(plan['reference_seed']));oldresult=json.loads((old/'metrics'/(row['id']+'.json')).read_text());oldmeta=json.loads((old/'inference'/(row['id']+'.json')).read_text());oldp=old/'scores'/(row['id']+'.npz');assert sha256(oldp)==oldmeta['scores_SHA256'];(out/'logs').mkdir(exist_ok=True)
 with np.load(p) as arrays,np.load(oldp) as oldarrays,(out/'logs'/(row['id']+'.log')).open('x') as log,redirect_stdout(log),redirect_stderr(log):
  assert np.array_equal(arrays['B0'],oldarrays['B0']) and oldresult['first_query']==320
  for name in ['B0']+[name for name in record['thresholds'] if name!='B0']:
   score=np.array(arrays[name]);normal=np.array(arrays[name+'_normal']);assert len(score)==len(y) and np.isfinite(score).all() and np.isfinite(normal).all();key=array_sha(score)
   if name=='B0':result={k:oldresult['methods']['B0'][k] for k in ['metrics','AP','ROC_AUC']}
   elif key in memo:result=memo[key]
   else:
    metrics={k:None for k in ['Affiliation-F','F1_T','Standard-F1','VUS-PR']}
    if two:
     raw=compute_metrics(score,y,row['period']);metrics={k:float(raw[k]) for k in metrics}
    result=dict(metrics=metrics,AP=float(average_precision_score(y,score)) if two else None,ROC_AUC=float(roc_auc_score(y,score)) if two else None)
   memo[key]=result;methods[name]=dict(result,normal_FPR_source99=float(np.mean(normal>record['thresholds'][name])),source_threshold=record['thresholds'][name],score_array_SHA256=key)
 atomic_json(out/'metrics'/(row['id']+'.json'),dict(system_id=row['system_id'],track=row['track'],channels=row['channels'],family=row['family'],event_types=row['event_types'],requested_strata=row['requested_strata'],length=row['length'],ready_points=len(y),first_query=320,methods=methods,RCA=record['RCA'],RCA_excluded=record['RCA_excluded'],RCA_scope='conditional true event window intersect >=320; not alarm-triggered; no TimeRCD RCA',C1_TimeRCD_passthrough=row['channels']==1,scores_SHA256=record['scores_SHA256'],target_updates=0,source_thresholds_only=True))
 return row['id']
if __name__=='__main__':
 mp.set_start_method('spawn',force=True);p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();plan=json.loads(a.plan.read_text());failed=[];done=0
 with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
  futures={pool.submit(one,(str(a.root),str(a.plan),row)):row for row in plan['rows']}
  for f in as_completed(futures):
   row=futures[f]
   try:f.result();done+=1
   except Exception:failed.append(row['id']);atomic_json(a.root/'errors'/('metric_'+row['id']+'.json'),dict(error=traceback.format_exc()))
   atomic_json(a.root/'metric_progress.json',dict(status='running',completed=done,total=len(plan['rows']),failed=failed))
 atomic_json(a.root/'metric_progress.json',dict(status='needs_attention' if failed else 'complete',completed=done,total=len(plan['rows']),failed=failed));assert not failed
