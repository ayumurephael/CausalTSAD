"""Frozen causal-context and likelihood controls, source plus explored development.

No optimizer. Source augmentation is a declared observation nuisance, not a new
causal effect label. Only values and normal topology enter predict().
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,argparse,json,time,pickle,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from proposals.context_probe import predict
from proposals.causal_syndromes import SyndromePrior
from proposals.raw_context import parameter_sha
from causal_tsad.io import atomic_json,sha256,array_sha,adjacency
SEEDS=[348,349,350]
def summaries(out):
 q=(out['residual'].square()/out['variance']).cpu().numpy();v=out['variance'].cpu().numpy()
 return dict(whitened_squared_quantiles=np.quantile(q,[.5,.9,.99]).tolist(),mean_whitened_square=float(q.mean()),predictive_variance_quantiles=np.quantile(v,[.5,.9,.99]).tolist(),time_quarters=[dict(q50=float(np.median(x)),q99=float(np.quantile(x,.99))) for x in np.array_split(q,4)])
def masked_rca(roots,events,length,c):
 result=[];excluded=[]
 for event in events:
  if not event.get('applied',True):continue
  mask=np.zeros(length,bool)
  for lo,hi in event['annotation_intervals']:mask[lo:hi]=True
  ready=mask[320:]
  if not ready.any() or not event['root_channels']:excluded.append(event['event_id']);continue
  rank=np.argsort(-roots[ready].mean(0),kind='stable');ranks=[int(np.flatnonzero(rank==root)[0])+1 for root in event['root_channels']];result.append(dict(event_id=event['event_id'],Top1=int(min(ranks)==1),Top3=int(min(ranks)<=min(c,3)),MRR=1/min(ranks),ready_points=int(ready.sum())))
 return result,excluded
def apply_prior(factors,models):
 scores=[];roots=[]
 for model in models:
  out=model(factors);scores.append(out['logit'].cpu().numpy());roots.append(out['root'].cpu().numpy())
 return np.stack(scores),np.stack(roots)
def background(cases,key,name):
 if name=='plain':return cases.copy()
 assert name=='smooth';rng=np.random.RandomState(int(key)%2**32);scale=np.median(np.abs(cases[0,:256]-np.median(cases[0,:256],axis=0)),axis=0)*1.4826;scale=np.where(scale<1e-3,1,scale);t=np.arange(cases.shape[1])[:,None];phase=rng.uniform(-np.pi,np.pi,cases.shape[2]);b=4*scale*np.sin(2*np.pi*t/(cases.shape[1]*.75)+phase);return cases+b[None]
def run(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(1*2**30/torch.cuda.get_device_properties(0).total_memory);plan=json.loads(a.plan.read_text());tag=a.kind+'_'+a.variant+'_w'+str(a.worker);start=time.time();models={};original={};weightsha={};records=[];counts={k:0 for k in ['query_scalar_forwards','full_input_scalar_slots','Gaussian_covariance_factorizations','context_ridge_factorizations','cached_prior_query_applications']}
 for mode in ['true_signature','empty_signature']:
  models[mode]=[]
  for seed in SEEDS:
   name=mode+'_s'+str(seed);path=Path(plan['weights'])/(name+'.pth');assert sha256(path)==plan['weight_SHA256'][name];state=torch.load(path,map_location='cpu',weights_only=False);model=SyndromePrior(seed).cuda().eval().requires_grad_(False);model.load_state_dict(state['model'],strict=True);models[mode].append(model);original[name]=parameter_sha(model);weightsha[name]=sha256(path)
 if a.kind=='source':
  manifest=json.loads((Path(plan['source'])/'manifest.json').read_text());assert sha256(Path(plan['source'])/'manifest.json')==plan['source_manifest_SHA256'];rows=[r for r in manifest['files'] if r['split'] in ['calibration','source_validation']];assert len(rows)==48;tasks=[(r,b) for r in rows for b in ['plain','smooth']];tasks=[tasks[i] for i in np.random.RandomState(348).permutation(len(tasks))][a.worker::a.workers];cache=json.loads((Path(plan['old_source_cache'])/'manifest.json').read_text());cached={(r['mode'],r['key']):r for r in cache['records']};mode=plan['source_mode']
 else:tasks=plan['target_rows'][a.worker::a.workers]
 for task in tasks:
  assert time.time()<a.deadline
  if a.kind=='source':
   row,bg=task;path=Path(plan['source'])/row['path'];assert sha256(path)==row['SHA256']
   with np.load(path) as f:raw=f['cases'];g=f['graph']
   values=background(raw,row['key'],bg);c=row['C'];ident=row['key']+'_'+bg;used_graph=g if mode=='true_signature' else np.zeros_like(g);gg=torch.tensor(used_graph,dtype=torch.float64,device='cuda');scores=[];rootranks=[];stats=[];reused=a.variant=='fixed' and bg=='plain'
   # Exact full-column class matching survives the same deterministic nuisance.
   assert all(sorted(array_sha(values[i,:,ch]) for i in [0,1])==sorted(array_sha(values[i,:,ch]) for i in [2,3]) for ch in range(c))
   if reused:
    entry=cached[(mode,row['key'])];p=Path(plan['old_source_cache'])/entry['path'];assert sha256(p)==entry['SHA256']
    with np.load(p) as f:factors=torch.tensor(f['log_factors'],dtype=torch.float64,device='cuda')
    # Apply each frozen prior to all8 cached views together, no new raw forward.
    ss,rr=apply_prior(factors,models[mode]);scores=ss.transpose(1,0,2);rootranks=rr.transpose(1,0,2,3)
   else:
    for value in values:
     out=predict(torch.tensor(value,dtype=torch.float64,device='cuda'),gg,a.variant);s,r=apply_prior(out['log_factors'],models[mode]);scores.append(s);rootranks.append(r);stats.append(summaries(out))
     for k in counts:
      if k in out:counts[k]+=out[k]
   scores=np.asarray(scores).transpose(1,0,2);rootranks=np.asarray(rootranks).transpose(1,0,2,3);methods={}
   for si,seed in enumerate(SEEDS):
    metrics={}
    for scope,views in [('matched_measurement',[0,1,2,3]),('paired_measurement',[0,1,4,5]),('physical',[0,1,6,7])]:
     y=np.repeat([int(v>=2) for v in views],len(scores[si,0]));metrics[scope]=dict(AP=float(average_precision_score(y,scores[si,views].reshape(-1))),ROC_AUC=float(roc_auc_score(y,scores[si,views].reshape(-1))))
    rca=[]
    for view in range(2,8):
     order=np.argsort(-rootranks[si,view].mean(0),kind='stable');rank=int(np.flatnonzero(order==row['root'])[0])+1;rca.append(dict(view=view,kind=row['case_kind'][view],Top1=int(rank==1),Top3=int(rank<=3),MRR=1/rank))
    methods[str(seed)]=dict(metrics=metrics,RCA=rca)
   record=dict(key=row['key'],split=row['split'],C=c,family=row['family'],background=bg,mode=mode,variant=a.variant,source_file_SHA256=row['SHA256'],input_SHA256=array_sha(values),graph_SHA256=array_sha(used_graph),matched_quartet_exact=True,methods=methods,normal_prediction_stats=stats[:2],old_raw_evidence_reused=reused)
   counts['cached_prior_query_applications']+=3*8*(len(values[0])-320)*c
   output=a.root/'source'/a.variant/mode/ident
  else:
   row=task;assert row['split']=='development' and row['pilot100'] and row['channels']>1;ident=row['id'];c=row['channels']
   with (Path(plan['benchmark'])/'data'/row['shard']).open('rb') as f:sample=pickle.load(f)[row['offset']]
   assert sample['system_id']==row['system_id'];x=np.asarray(sample['time_series']);n=np.asarray(sample['normal_time_series']);assert array_sha(x)==row['time_series_sha256'] and array_sha(n)==row['normal_sha256'];g=adjacency(sample['graph'],c)
   for mode in ['true_signature','empty_signature']:
    gg=torch.tensor(g if mode=='true_signature' else np.zeros_like(g),dtype=torch.float64,device='cuda');scores=[];roots=None;stats=[]
    for value in [x,n]:
     out=predict(torch.tensor(value,dtype=torch.float64,device='cuda'),gg,a.variant);s,r=apply_prior(out['log_factors'],models[mode]);scores.append(s);stats.append(summaries(out));roots=r if roots is None else roots
     for k in counts:
      if k in out:counts[k]+=out[k]
     counts['cached_prior_query_applications']+=3*(len(value)-320)*c
    scores=np.stack(scores,axis=1);y=np.asarray(sample['labels'],int)[320:];two=len(np.unique(y))==2;methods={}
    for si,seed in enumerate(SEEDS):
     rca,excluded=masked_rca(roots[si],sample['events'],len(x),c);methods[str(seed)]=dict(AP=float(average_precision_score(y,scores[si,0])) if two else None,ROC_AUC=float(roc_auc_score(y,scores[si,0])) if two else None,RCA=rca,RCA_excluded=excluded)
    output=a.root/'target'/a.variant/mode/ident;output.parent.mkdir(parents=True,exist_ok=True);assert not output.with_suffix('.npz').exists();np.savez_compressed(output.with_suffix('.npz'),scores=scores);record=dict(system_id=row['system_id'],id=ident,track=row['track'],C=c,family=row['family'],length=len(x),requested_strata=row['requested_strata'],mode=mode,variant=a.variant,methods=methods,input_observed_SHA256=array_sha(x),input_normal_SHA256=array_sha(n),graph_SHA256=array_sha(gg.cpu().numpy()),normal_prediction_stats=stats[1],observed_prediction_stats=stats[0],scores_SHA256=sha256(output.with_suffix('.npz')),target_updates=0)
    atomic_json(output.with_suffix('.json'),record);records.append(str(output.with_suffix('.json').relative_to(a.root)))
   atomic_json(a.root/(tag+'_progress.json'),dict(status='running',completed=len(records),**counts));continue
  output.parent.mkdir(parents=True,exist_ok=True);assert not output.with_suffix('.npz').exists();np.savez_compressed(output.with_suffix('.npz'),scores=scores);record['scores_SHA256']=sha256(output.with_suffix('.npz'));atomic_json(output.with_suffix('.json'),record);records.append(str(output.with_suffix('.json').relative_to(a.root)));atomic_json(a.root/(tag+'_progress.json'),dict(status='running',completed=len(records),**counts))
 assert all(parameter_sha(models[mode][i])==original[mode+'_s'+str(seed)] for mode in models for i,seed in enumerate(SEEDS));atomic_json(a.root/(tag+'_manifest.json'),dict(status='complete',records=records,**counts,weight_SHA256=weightsha,parameters_unchanged=True,target_updates=0,optimizer_updates=0,source_training_arrays_opened=False,validation_or_locked_arrays_opened=False,plan_SHA256=sha256(a.plan),seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated()));atomic_json(a.root/(tag+'_progress.json'),dict(status='complete',completed=len(records),**counts))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['root','plan']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--kind',choices=['source','target'],required=True);p.add_argument('--variant',choices=['fixed','rolling','student'],required=True);p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
 try:run(a)
 except Exception:atomic_json(a.root/'errors'/(a.kind+'_'+a.variant+'_w'+str(a.worker)+'.json'),dict(error=traceback.format_exc()));raise
