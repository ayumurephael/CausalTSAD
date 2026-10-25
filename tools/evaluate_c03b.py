"""Same100development, common raw-logit original4metrics; target arrays never optimized."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,pickle,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import sha256,atomic_json
from proposals.c09.context_model import ConditionalTail,TailFusion
from proposals.c09.context_data import batch,manifest,infer,digest
def run(a):
 import torch
 from sklearn.metrics import average_precision_score
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;saved=torch.load(a.run/'final.pth',map_location='cpu',weights_only=False);cfg=saved['config']
 torch.cuda.set_per_process_memory_fraction(cfg['memory_gib_per_worker']*2**30/torch.cuda.get_device_properties(0).total_memory)
 tail=ConditionalTail(cfg['feature_dim'],cfg['width']).cuda().eval();fusion=TailFusion().cuda().eval();tail.load_state_dict(saved['tail'],strict=True);fusion.load_state_dict(saved['fusion'],strict=True)
 before=dict(tail=digest(tail),fusion=digest(fusion));assert sha256(a.cache/'manifest.json')==saved['cache_manifest_sha256'];m=manifest(a.cache);rows=[r for r in m['files'] if r['kind']=='development'];assert len(rows)==100
 sys.path.insert(0,str(a.metric_code));from evaluate_table1 import compute_metrics
 from utils.slidingWindows import find_length_rank
 results=[];slots=0;current=None;pack=None;c1error=0;warmup_error=0
 for r in rows:
  assert sha256(a.cache/r['path'])==r['SHA256'];b=batch(a.cache,[r]);pred={}
  with torch.no_grad():
   for view in ['x','normal']:
    out=infer(tail,fusion,b,view,saved['arm']);pred[view+'_fused']=out['logit'][0].cpu().numpy();pred[view+'_base']=b[view+'_base'][0].cpu().numpy();pred[view+'_node']=out['node'][0].cpu().numpy();slots+=r['scalars']
    error=float(np.max(abs(pred[view+'_fused'][:256]-pred[view+'_base'][:256])));warmup_error=max(warmup_error,error);assert error==0
    if r['channels']==1:error=float(np.max(abs(pred[view+'_fused']-pred[view+'_base'])));c1error=max(c1error,error);assert error==0
  info=r['row'];assert info['split']=='development'
  if current!=info['shard']:
   with (a.benchmark/'data'/info['shard']).open('rb') as f:pack=pickle.load(f)
   current=info['shard']
  sample=pack[info['offset']];assert sample['system_id']==r['key'];y=np.asarray(sample['labels'],int);period=max(1,int(find_length_rank(sample['time_series'],rank=1)));scores={}
  for name in ['base','fused']:
   valid=len(np.unique(y))>1;score=pred['x_'+name];metrics={k:None for k in ['Affiliation-F','F1_T','Standard-F1','VUS-PR']}
   if valid:
    values=compute_metrics(score,y,period);metrics={k:float(values[k]) for k in metrics}
   threshold=saved['source_calibration']['base_threshold' if name=='base' or r['channels']==1 else 'threshold']
   scores[name]=dict(metrics=metrics,AP=float(average_precision_score(y,score)) if valid else None,normal_FPR=float(np.mean(pred['normal_'+name]>threshold)))
  rca=[];skipped=0
  if r['channels']>1:
   for event in sample['events']:
    if not event.get('applied',True):continue
    mask=np.zeros(len(y),bool)
    for start,end in event['annotation_intervals']:mask[start:end]=True
    mask[:256]=False
    if not mask.any():skipped+=1;continue
    ranking=np.argsort(-pred['x_node'][mask].mean(0),kind='stable');ranks=[int(np.flatnonzero(ranking==root)[0])+1 for root in event['root_channels']]
    rca.append(dict(event_id=event['event_id'],root_ranks=ranks,Top1=int(min(ranks)==1),Top3=int(min(ranks)<=min(3,r['channels'])),MRR=1/min(ranks)))
  path=a.run/'development_scores'/(r['name']+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**pred)
  record=dict(system_id=r['key'],channels=r['channels'],track=info['track'],family=info['family'],event_types=info['event_types'],scores=scores,RCA=rca,
   root_window_entirely_in_warmup_excluded=skipped,C1_RCA_excluded=r['channels']==1,input_sha256=info['time_series_sha256'],normal_input_sha256=info['normal_sha256'],scores_sha256=sha256(path))
  atomic_json(a.run/'development_records'/(r['name']+'.json'),record);results.append(record);atomic_json(a.run/'evaluation_progress.json',dict(status='running',completed=len(results),total=100,head_query_scalars=slots))
 after=dict(tail=digest(tail),fusion=digest(fusion));assert before==after
 atomic_json(a.run/'development_results.json',dict(status='complete',records=results,checkpoint_sha256=sha256(a.run/'final.pth'),head_query_scalars=slots,
  parameters_before=before,parameters_after=after,parameter_updates=0,encoder_forward_scalars=0,C1_logit_error=c1error,warmup_logit_error=warmup_error,
  target_validation_or_locked_arrays_opened=False,scope='Common raw-logit original4metrics; conditional ready-region RCA only, C1 excluded'))
 atomic_json(a.run/'evaluation_progress.json',dict(status='complete',completed=100,total=100,head_query_scalars=slots))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--metric-code',type=Path,required=True);a=p.parse_args()
 try:run(a)
 except Exception:atomic_json(a.run/'evaluation_error.json',dict(error=traceback.format_exc()));raise
