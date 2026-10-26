"""Frozen raw causal signatures on the historical 100 development systems.

Only X and normal topology enter evidence(). Prefix latent regression is forward
conditioning, not target SGD. The normal twin has its own independent context.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,json,pickle,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from proposals.causal_syndromes import condition,evidence,SyndromePrior,FIRST
from proposals.matched_pairs import rewire
from proposals.raw_context import parameter_sha
from causal_tsad.io import atomic_json,sha256,array_sha,adjacency
MODES=['true_signature','true_self','wrong_signature','empty_signature']
SEEDS=[348,349,350]

def run(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(1*2**30/torch.cuda.get_device_properties(0).total_memory)
 plan=json.loads(a.plan.read_text());models={};hashes={};thresholds={};weightsha={};out=a.root/'development';tasks=plan['rows'][a.worker::a.workers];query=slots=factors=conditioning=0;records=[];start=time.time()
 for mode in MODES:
  for seed in SEEDS:
   name=mode+'_s'+str(seed);p=Path(plan['weights'])/(name+'.pth');assert sha256(p)==plan['weight_SHA256'][name];s=torch.load(p,map_location='cpu',weights_only=False);assert s['mode']==mode and s['seed']==seed
   m=SyndromePrior(seed).cuda().eval().requires_grad_(False);m.load_state_dict(s['model'],strict=True);models[name]=m;hashes[name]=parameter_sha(m);thresholds[name]=float(s['source_threshold']);weightsha[name]=sha256(p)
 for row in tasks:
  assert time.time()<a.deadline and row['split']=='development' and row['pilot100'];c=row['channels'];ident=row['id'];old=Path(plan['reference'])/'development'/('s'+str(plan['reference_seed']));meta=json.loads((old/'inference'/(ident+'.json')).read_text());p=old/'scores'/(ident+'.npz');assert sha256(p)==meta['scores_SHA256']
  with np.load(p) as f:base=f['B0'].copy();normal=f['B0_normal'].copy()
  with (Path(plan['benchmark'])/'data'/row['shard']).open('rb') as f:sample=pickle.load(f)[row['offset']]
  assert sample['system_id']==row['system_id'];x=np.asarray(sample['time_series']);n=np.asarray(sample['normal_time_series']);assert x.dtype==n.dtype==np.float64;assert array_sha(x)==row['time_series_sha256'] and array_sha(n)==row['normal_sha256'];assert len(base)==len(x)-FIRST and len(normal)==len(base)
  scores=dict(B0=base,B0_normal=normal);roots={};t=dict(thresholds);used_graphs={};bt=float(plan['B0_matched_source_threshold'])
  if c==1:
   bt=float(plan['original_source_thresholds']['selftrained_uni/logit']);t={name:bt for name in models}
   for name in models:scores[name]=base.copy();scores[name+'_normal']=normal.copy()
  else:
   g=torch.tensor(adjacency(sample['graph'],c),dtype=torch.float64,device='cuda')
   for view,value in [('observed',x),('normal',n)]:
    v=torch.tensor(value,dtype=torch.float64,device='cuda');true_state=condition(v[:256],g);conditioning+=3*c
    for mode in MODES:
     gg=rewire(g,row['system_id']) if mode=='wrong_signature' else torch.zeros_like(g) if mode=='empty_signature' else g
     state=true_state if mode in ['true_signature','true_self'] else condition(v[:256],gg)
     if mode not in ['true_signature','true_self']:conditioning+=3*c
     ev=evidence(v,gg,signature='self' if mode=='true_self' else 'full',state=state);query+=len(ev['log_factors'])*c;slots+=v.numel();factors+=ev['factorizations'];used_graphs[mode]=array_sha(gg.cpu().numpy())
     for seed in SEEDS:
      name=mode+'_s'+str(seed);result=models[name](ev['log_factors']);scores[name+('_normal' if view=='normal' else '')]=result['logit'].cpu().numpy()
      if view=='observed':roots[name]=result['root'].cpu().numpy()
  # Audit supervision is accessed after model calls and never passed to evidence.
  rca={};excluded=[]
  for event in sample['events']:
   if not event.get('applied',True):continue
   mask=np.zeros(len(x),bool)
   for lo,hi in event['annotation_intervals']:mask[lo:hi]=True
   ready=mask[FIRST:]
   if c==1 or not ready.any() or not event['root_channels']:
    excluded.append(dict(event_id=event['event_id'],reason='C1' if c==1 else 'no_ready_root_window'));continue
   for name,value in roots.items():
    rank=np.argsort(-value[ready].mean(0),kind='stable');ranks=[int(np.flatnonzero(rank==root)[0])+1 for root in event['root_channels']];rca.setdefault(name,[]).append(dict(event_id=event['event_id'],root_ranks=ranks,Top1=int(min(ranks)==1),Top3=int(min(ranks)<=min(3,c)),MRR=1/min(ranks),ready_points=int(ready.sum())))
  p=out/'scores'/(ident+'.npz');p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();np.savez_compressed(p,**scores,**{'root_'+k:v for k,v in roots.items()})
  record=dict(system_id=row['system_id'],scores_SHA256=sha256(p),input_observed_SHA256=array_sha(x),input_normal_SHA256=array_sha(n),B0_reference_file_SHA256=meta['scores_SHA256'],B0_score_array_SHA256=array_sha(base),used_graph_SHA256=used_graphs,weight_SHA256=weightsha,first_query=FIRST,thresholds=dict(t,B0=bt),RCA=rca,RCA_excluded=excluded,C1_TimeRCD_passthrough=c==1,target_updates=0,context='each view own observed prefix; robust closed-form latent regression, no normal-twin repair')
  atomic_json(out/'inference'/(ident+'.json'),record);records.append(dict(id=ident,scores_SHA256=record['scores_SHA256']));atomic_json(a.root/('worker'+str(a.worker)+'_progress.json'),dict(status='running',completed=len(records),total=len(tasks),query_scalar_forwards=query,full_input_scalar_slots=slots,Gaussian_covariance_factorizations=factors,context_ridge_factorizations=conditioning,target_updates=0,TimeRCD_new_forwards=0))
 assert all(parameter_sha(m)==hashes[k] for k,m in models.items())
 atomic_json(a.root/('worker'+str(a.worker)+'_manifest.json'),dict(status='complete',records=records,parameters_unchanged=True,query_scalar_forwards=query,full_input_scalar_slots=slots,Gaussian_covariance_factorizations=factors,context_ridge_factorizations=conditioning,global_parameter_SHA256=hashes,weight_SHA256=weightsha,plan_SHA256=sha256(a.plan),target_updates=0,TimeRCD_new_forwards=0,validation_or_locked_arrays_opened=False,peak_cuda_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-start))
 atomic_json(a.root/('worker'+str(a.worker)+'_progress.json'),dict(status='complete',completed=len(records),total=len(tasks),query_scalar_forwards=query,full_input_scalar_slots=slots,parameters_unchanged=True,target_updates=0))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ['root','plan']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
 try:run(a)
 except Exception:atomic_json(a.root/'errors'/('worker'+str(a.worker)+'.json'),dict(error=traceback.format_exc()));raise
