"""Fixed original100development, no adaptation/selection; primary raw-logit comparisons."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,pickle,sys,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import sha256,atomic_json
from proposals.c08.model import JointRootReadout
from proposals.c08.data import load_manifest,batch,infer,digest

def rank_info(scores,roots,c):
 ranking=np.argsort(-scores,kind='stable');ranks=[int(np.flatnonzero(ranking==int(root))[0])+1 for root in roots]
 return dict(root_ranks=ranks,Top1=int(min(ranks)==1),Top3=int(min(ranks)<=min(3,c)),MRR=1/min(ranks))

def run(a):
 import torch
 from sklearn.metrics import average_precision_score
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 saved=torch.load(a.run/'final.pth',map_location='cpu',weights_only=False);cfg=saved['config'];model=JointRootReadout(cfg['feature_dim'],cfg['width'],cfg['layers']).cuda().eval()
 model.load_state_dict(saved['model'],strict=True);before=digest(model);m=load_manifest(a.cache,saved['cache_manifest_sha256'])
 rows=[r for r in m['files'] if r['kind']=='development'];assert len(rows)==100
 sys.path.insert(0,str(a.metric_code));from evaluate_table1 import compute_metrics
 from utils.slidingWindows import find_length_rank
 results=[];current=None;pack=None;slots=0;c1error=0
 for r in rows:
  assert sha256(a.cache/r['path'])==r['SHA256'];b=batch(a.cache,[r],saved['arm']);pred={}
  with torch.no_grad():
   for view in ['x','normal']:
    out=infer(model,b,view);pred[view+'_fused']=out['logit'][0].cpu().numpy();pred[view+'_base']=b[view+'_base'][0].cpu().numpy()
    pred[view+'_root']=out['root_log_prob'][0].exp().cpu().numpy();slots+=r['scalars']
  if r['channels']==1:
   error=max(float(np.max(abs(pred[v+'_fused']-pred[v+'_base']))) for v in ['x','normal']);c1error=max(c1error,error);assert error==0
  # Targets/events first read after predictions; never passed to model or calibration.
  row=r['row'];assert row['split']=='development'
  if current!=row['shard']:
   with (a.benchmark/'data'/row['shard']).open('rb') as f:pack=pickle.load(f)
   current=row['shard']
  sample=pack[row['offset']];assert sample['system_id']==r['key'];y=np.asarray(sample['labels'],int);period=max(1,int(find_length_rank(sample['time_series'],rank=1)));scores={}
  for name in ['base','fused']:
   score=pred['x_'+name];valid=len(np.unique(y))>1;metrics={k:None for k in ['Affiliation-F','F1_T','Standard-F1','VUS-PR']}
   if valid:
    raw=compute_metrics(score,y,period);metrics={k:float(raw[k]) for k in metrics}
   key='base_threshold' if name=='base' else 'threshold';threshold=saved['source_calibration'][key]
   # C1 protected branch inherits common base threshold, calibrated on M only; state scope.
   if r['channels']==1:threshold=saved['source_calibration']['base_threshold']
   scores[name]=dict(metrics=metrics,AP=float(average_precision_score(y,score)) if valid else None,normal_FPR=float(np.mean(pred['normal_'+name]>threshold)))
  events=[]
  if r['channels']>1:
   for event in sample['events']:
    if not event.get('applied',True):continue
    mask=np.zeros(len(y),bool)
    for start,end in event['annotation_intervals']:mask[start:end]=True
    if not mask.any():continue
    conditional=rank_info(pred['x_root'][mask].mean(0),event['root_channels'],r['channels'])
    detected=mask&(pred['x_fused']>saved['source_calibration']['threshold'])
    endtoend=rank_info(pred['x_root'][detected].mean(0),event['root_channels'],r['channels']) if detected.any() else dict(Top1=0,Top3=0,MRR=0,root_ranks=None)
    events.append(dict(event_id=event['event_id'],conditional=conditional,event_detected=bool(detected.any()),end_to_end_event_matched=endtoend))
  path=a.run/'development_scores'/(r['name']+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**pred)
  record=dict(system_id=r['key'],track=row['track'],family=row['family'],channels=r['channels'],event_types=row['event_types'],scores=scores,RCA=events,
   input_sha256=row['time_series_sha256'],normal_input_sha256=row['normal_sha256'],scores_sha256=sha256(path),C1_RCA_excluded=r['channels']==1)
  atomic_json(a.run/'development_records'/(r['name']+'.json'),record);results.append(record)
  atomic_json(a.run/'evaluation_progress.json',dict(status='running',completed=len(results),total=100,head_query_scalars=slots))
 assert digest(model)==before
 atomic_json(a.run/'development_results.json',dict(status='complete',records=results,parameter_updates=0,checkpoint_sha256=sha256(a.run/'final.pth'),
  head_query_scalars=slots,encoder_forward_scalars=0,parameter_sha_before=before,parameter_sha_after=digest(model),C1_logit_error=c1error,
  target_validation_or_locked_arrays_opened=False,score_representation='common raw logit, original4 metric formulas; no target threshold fitting',
  RCA_scope='conditional injection-window ranking and detection+root event-matched Top1 with misses=0; not free-running alarm precision'))
 atomic_json(a.run/'evaluation_progress.json',dict(status='complete',completed=100,total=100,head_query_scalars=slots))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
 p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--metric-code',type=Path,required=True);a=p.parse_args()
 try:run(a)
 except Exception:atomic_json(a.run/'evaluation_error.json',dict(error=traceback.format_exc()));raise
