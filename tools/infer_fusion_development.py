"""Fixed final heads, full development only; input whitelist separate from RCA audit."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,json,pickle,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from proposals.causal_contrast import CausalContrast
from proposals.matched_pairs import rewire
from proposals.protected_fusion import ProtectedFusion,causal_features,parameter_sha
from causal_tsad.io import atomic_json,sha256,array_sha,adjacency
ARMS=['basecal','degree','true','wrong']

def run(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(1*2**30/torch.cuda.get_device_properties(0).total_memory)
 plan=json.loads(a.jobs.read_text());assert plan['seed']==a.seed;rows=plan['rows'];base_root=Path(plan['base_root']);models={};heads={};hashes={};head_hashes={};run_sha={}
 for arm in ARMS:
  run=a.root/'runs'/(arm+'_s'+str(a.seed));progress=json.loads((run/'progress.json').read_text());assert progress['status']=='complete' and progress['updates']==568 and progress['optimized_ready_query_scalar_exposures']==15561216
  assert sha256(run/'final.pth')==progress['checkpoint_SHA256'];s=torch.load(run/'final.pth',map_location='cpu',weights_only=False);state=s['model'];h=ProtectedFusion(state['mean'],state['std'],state['base_scale']).cuda().eval().requires_grad_(False);h.load_state_dict(state,strict=True);heads[arm]=(h,s['source_calibration']);head_hashes[arm]=parameter_sha(h);run_sha[arm]=progress['checkpoint_SHA256']
  if arm=='basecal':continue
  path=a.root/'frozen_c04'/(arm+'_s'+str(a.seed)+'.pth');assert sha256(path)==s['teacher_checkpoint_SHA256'];saved=torch.load(path,map_location='cpu',weights_only=False);m=CausalContrast().cuda().eval().requires_grad_(False);m.load_state_dict(saved['model'],strict=True);models[arm]=m;hashes[arm]=parameter_sha(m)
 output=a.root/'development'/('s'+str(a.seed));start=time.monotonic();query=0;headslots=0;current=None;pack=None
 for index,row in enumerate(rows):
  assert time.monotonic()-start<3*3600;assert row['split']=='development';ident=row['id'];dest=output/'inference'/(ident+'.json');assert not dest.exists()
  path=base_root/row['path'];assert sha256(path)==row['SHA256']
  with np.load(path) as f:bases={k:np.array(f[k]) for k in ['base','normal']}
  if current!=row['shard']:
   with (a.benchmark/'data'/row['shard']).open('rb') as f:pack=pickle.load(f)
   current=row['shard']
  sample=pack[row['offset']];assert sample['system_id']==row['system_id'];c=row['channels'];length=row['length'];values={view:np.asarray(sample[field]) for view,field in [('base','time_series'),('normal','normal_time_series')]}
  assert array_sha(values['base'])==row['time_series_sha256'] and array_sha(values['normal'])==row['normal_sha256']
  graph=adjacency(sample['graph'],c);g=torch.from_numpy(graph).cuda();predicted={};roots={};thresholds={};base_thresholds={}
  with torch.inference_mode():
   for view,x in values.items():
    base=torch.from_numpy(bases[view]).cuda()
    if c==1:
     for arm in ARMS:predicted[arm+'__'+view]=bases[view].copy()
     continue
    features={}
    for mode,m in models.items():
     gg=rewire(g,row['system_id']) if mode=='wrong' else g;f,r=causal_features(m,torch.tensor(x,dtype=torch.float32,device='cuda'),gg,mode,chunk=256)
     features[mode]=torch.cat([base[320:,None],f,torch.tensor([float(torch.log1p(gg.sum(0).mean())),float(np.log(c))],device='cuda')[None].expand(len(f),-1)],-1)
     query+=len(f)*c
     if view=='base':roots[mode]=r.cpu().numpy()
    for arm,(head,cal) in heads.items():
     f=features['degree' if arm=='basecal' else arm];out,delta=head.full_scores(base,f,arm,c);assert torch.equal(out[:320],base[:320]);assert torch.isfinite(out).all()
     predicted[arm+'__'+view]=out.cpu().numpy();thresholds[arm]=cal['source_threshold'];base_thresholds[arm]=cal['base_threshold'];headslots+=len(f)
  # Ground-truth events are used only after all model calls have finished.
  if c==1:
   theta=plan['original_source_thresholds']['selftrained_uni/logit'];thresholds={arm:theta for arm in ARMS};base_thresholds=thresholds.copy()
  rca={};excluded=[]
  for event in sample['events']:
   if not event.get('applied',True):continue
   mask=np.zeros(length,bool)
   for lo,hi in event['annotation_intervals']:mask[lo:hi]=True
   eligible=mask[320:]
   if c==1 or not eligible.any() or not event['root_channels']:
    excluded.append(dict(event_id=event['event_id'],reason='C1' if c==1 else 'no_ready_root_window'));continue
   for mode,r in roots.items():
    rank=np.argsort(-r[eligible].mean(0),kind='stable');ranks=[int(np.flatnonzero(rank==root)[0])+1 for root in event['root_channels']]
    rca.setdefault(mode,[]).append(dict(event_id=event['event_id'],root_ranks=ranks,Top1=int(min(ranks)==1),Top3=int(min(ranks)<=min(3,c)),MRR=1/min(ranks),ready_window_points=int(eligible.sum())))
  score_path=output/'scores'/(ident+'.npz');score_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(score_path,**predicted,**{'root_'+k:v for k,v in roots.items()})
  atomic_json(dest,dict(system_id=row['system_id'],seed=a.seed,score_path=str(score_path.relative_to(a.root)),scores_SHA256=sha256(score_path),B0_array_SHA256=row['SHA256'],head_checkpoint_SHA256=run_sha,source_thresholds=thresholds,paired_base_source_thresholds=base_thresholds,RCA=rca,RCA_excluded=excluded,RCA_scope='conditional annotation window intersect t>=320; not alarm-triggered; no TimeRCD RCA',original_parameters_unchanged=True,target_updates=0))
  atomic_json(output/(a.jobs.stem+'_progress.json'),dict(status='running',completed=index+1,total=len(rows),frozen_C04_query_scalars=query,head_query_time_points=headslots,TimeRCD_new_forwards=0,target_updates=0,validation_or_locked_arrays_opened=False))
 assert all(parameter_sha(m)==hashes[k] for k,m in models.items());assert all(parameter_sha(h[0])==head_hashes[k] for k,h in heads.items())
 atomic_json(output/(a.jobs.stem+'_progress.json'),dict(status='complete',completed=len(rows),frozen_C04_query_scalars=query,head_query_time_points=headslots,TimeRCD_new_forwards=0,target_updates=0,parameters_unchanged=True,validation_or_locked_arrays_opened=False,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['root','benchmark','jobs']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--seed',type=int,required=True);a=p.parse_args()
 try:run(a)
 except Exception:
  atomic_json(a.root/'development'/('s'+str(a.seed))/(a.jobs.stem+'_error.json'),dict(error=traceback.format_exc()));raise
