"""C04 as a separate detector, shared ready suffix with cached TimeRCD, no fusion."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,json,pickle,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from proposals.causal_contrast import CausalContrast
from proposals.matched_pairs import rewire
from proposals.protected_fusion import causal_features,parameter_sha
from causal_tsad.io import atomic_json,sha256,array_sha,adjacency
def run(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(1*2**30/torch.cuda.get_device_properties(0).total_memory)
 plan=json.loads(a.jobs.read_text());seed=plan['seed'];base_root=Path(plan['base_root']);out=a.root/'development'/('s'+str(seed));models={};hashes={};thresholds={};weightsha={};query=0;current=None;start=time.monotonic()
 for mode in ['degree','true','wrong']:
  p=Path(plan['weights'])/(mode+'_s'+str(seed)+'.pth');assert sha256(p)==plan['weight_SHA256'][mode];s=torch.load(p,map_location='cpu',weights_only=False);m=CausalContrast().cuda().eval().requires_grad_(False);m.load_state_dict(s['model'],strict=True);models[mode]=m;hashes[mode]=parameter_sha(m);thresholds[mode]=s['source_calibration']['source_threshold'];weightsha[mode]=sha256(p)
 for index,row in enumerate(plan['rows']):
  assert time.monotonic()-start<3*3600 and row['split']=='development';c=row['channels'];ident=row['id'];path=base_root/row['path'];assert sha256(path)==row['SHA256']
  with np.load(path) as f:base=np.array(f['base'][320:]);normal=np.array(f['normal'][320:])
  if current!=row['shard']:
   with (a.benchmark/'data'/row['shard']).open('rb') as f:pack=pickle.load(f)
   current=row['shard']
  sample=pack[row['offset']];assert sample['system_id']==row['system_id'];x=np.asarray(sample['time_series']);n=np.asarray(sample['normal_time_series']);assert array_sha(x)==row['time_series_sha256'] and array_sha(n)==row['normal_sha256'];g=torch.from_numpy(adjacency(sample['graph'],c)).cuda();scores=dict(B0=base,B0_normal=normal);roots={};t=dict(thresholds);bt=plan['B0_matched_source_threshold']
  if c==1:
   bt=plan['original_source_thresholds']['selftrained_uni/logit'];t={mode:bt for mode in models}
   for mode in models:scores[mode]=base.copy();scores[mode+'_normal']=normal.copy()
  else:
   for view,value in [('observed',x),('normal',n)]:
    for mode,m in models.items():
     gg=rewire(g,row['system_id']) if mode=='wrong' else g;features,root=causal_features(m,torch.tensor(value,dtype=torch.float32,device='cuda'),gg,mode,chunk=256);scores[mode+('_normal' if view=='normal' else '')]=features[:,0].cpu().numpy();query+=len(features)*c
     if view=='observed':roots[mode]=root.cpu().numpy()
  # All targets/roots/normal twins below are audit outputs, not detector inputs.
  rca={};excluded=[]
  for event in sample['events']:
   if not event.get('applied',True):continue
   mask=np.zeros(len(x),bool)
   for lo,hi in event['annotation_intervals']:mask[lo:hi]=True
   ready=mask[320:]
   if c==1 or not ready.any() or not event['root_channels']:
    excluded.append(dict(event_id=event['event_id'],reason='C1' if c==1 else 'no_ready_root_window'));continue
   for mode,value in roots.items():
    rank=np.argsort(-value[ready].mean(0),kind='stable');ranks=[int(np.flatnonzero(rank==root)[0])+1 for root in event['root_channels']];rca.setdefault(mode,[]).append(dict(event_id=event['event_id'],root_ranks=ranks,Top1=int(min(ranks)==1),Top3=int(min(ranks)<=min(3,c)),MRR=1/min(ranks),ready_points=int(ready.sum())))
  p=out/'scores'/(ident+'.npz');p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();np.savez_compressed(p,**scores,**{'root_'+k:v for k,v in roots.items()})
  atomic_json(out/'inference'/(ident+'.json'),dict(system_id=row['system_id'],scores_SHA256=sha256(p),B0_array_SHA256=row['SHA256'],weight_SHA256=weightsha,first_query=320,thresholds=dict(t,B0=bt),RCA=rca,RCA_excluded=excluded,C1_TimeRCD_passthrough=c==1,target_updates=0))
  atomic_json(out/(a.jobs.stem+'_progress.json'),dict(status='running',completed=index+1,total=len(plan['rows']),frozen_query_scalar_forwards=query,TimeRCD_new_forwards=0,target_updates=0))
 assert all(parameter_sha(m)==hashes[k] for k,m in models.items())
 atomic_json(out/(a.jobs.stem+'_progress.json'),dict(status='complete',completed=len(plan['rows']),frozen_query_scalar_forwards=query,parameters_unchanged=True,TimeRCD_new_forwards=0,target_updates=0,validation_or_locked_arrays_opened=False,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['root','benchmark','jobs']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args()
 try:run(a)
 except Exception:atomic_json(a.root/'errors'/(a.jobs.stem+'.json'),dict(error=traceback.format_exc()));raise
