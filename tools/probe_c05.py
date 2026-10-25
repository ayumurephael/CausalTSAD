"""Source-only measurement-isolation probe, no learned repair or physical do claim."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,hashlib,time,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import sha256,atomic_json,array_sha

def state_sha(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def probe(a):
 import torch
 from sklearn.metrics import average_precision_score
 from dependencies.c02_model import ParentTiming,loss_and_score
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;cfg=json.loads(a.config.read_text())
 torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(0).total_memory)
 assert not a.output.exists();a.output.mkdir(parents=True);expected=cfg['teachers'][f'{a.mode}_s{a.seed}'];weight=a.weights/expected['file'];assert sha256(weight)==expected['SHA256']
 saved=torch.load(weight,map_location='cpu',weights_only=False);model=ParentTiming().cuda().eval().requires_grad_(False);model.load_state_dict(saved['model'],strict=True);before=state_sha(model)
 assert sha256(a.source/'manifest.json')==cfg['source_manifest_sha256'];manifest=json.loads((a.source/'manifest.json').read_text());rows=[];slots=0;original_refs=saved['source_calibration']['records'];threshold=saved['source_calibration']['threshold'];reference_errors=[]
 for packrow in manifest['files']:
  if packrow['split']!='source_validation' or packrow['channels']==1:continue
  assert sha256(a.source/packrow['path'])==packrow['SHA256']
  with np.load(a.source/packrow['path']) as z:pack={k:np.array(z[k]) for k in z.files}
  c=packrow['channels']
  for i in range(len(pack['x'])):
   x=pack['x'][i];normal=pack['normal'][i];root=int(pack['root'][i]);graph=pack['adjacency'][i];event=pack['root_window'][i].astype(bool);ready=np.arange(len(x))>=64;mask=event&ready
   isolated=normal.copy();isolated[:,root]=x[:,root];assert np.array_equal(isolated[:,root],x[:,root])
   assert np.array_equal(isolated[:,np.arange(c)!=root],normal[:,np.arange(c)!=root])
   actual_graph=graph if a.mode=='parents_current' else np.zeros_like(graph);pred={}
   with torch.no_grad():
    for name,values in [('normal',normal),('original',x),('measurement_only',isolated)]:
     b=dict(values=torch.from_numpy(values)[None].cuda(),graph=torch.from_numpy(actual_graph)[None].cuda(),timing='current' if a.mode=='parents_current' else 'past')
     _,score,node=loss_and_score(model,b);pred[name]=dict(score=score[0].cpu().numpy(),node=node[0].cpu().numpy());slots+=values.size
   children=np.flatnonzero(graph[root]);others=np.flatnonzero((np.arange(c)!=root)&(graph[root]==0));direct_delta=(pred['measurement_only']['node']-pred['normal']['node'])
   unrelated_error=float(np.max(abs(direct_delta[:,others]))) if len(others) else 0.
   assert unrelated_error<1e-6,'A measured-root change must not affect teacher scores of nonchildren/nonroot'
   own_only_error=float(np.max(abs(direct_delta[:,np.arange(c)!=root]))) if a.mode=='none' else None
   if own_only_error is not None:assert own_only_error<1e-6
   metrics={}
   for name,predicted in pred.items():
    full_rank=None;ready_rank=None
    if event.any():full_rank=int(np.flatnonzero(np.argsort(-predicted['node'][event].mean(0),kind='stable')==root)[0])+1
    if mask.any():ready_rank=int(np.flatnonzero(np.argsort(-predicted['node'][mask].mean(0),kind='stable')==root)[0])+1
    metrics[name]=dict(root_rank_full=full_rank,root_rank_ready=ready_rank,Top1_ready=int(ready_rank==1) if ready_rank is not None else None,
     child_score_mean=float(predicted['node'][mask][:,children].mean()) if mask.any() and len(children) else None,
     child_threshold_exceedance=float(np.mean(predicted['node'][mask][:,children]>threshold)) if mask.any() and len(children) else None,
     root_score_mean=float(predicted['node'][mask,root].mean()) if mask.any() else None)
   reference=original_refs[f'c{c}/{i}'];old_ap=reference['observed_AP'];new_ap=float(average_precision_score(pack['labels'][i],pred['original']['score']))
   assert old_ap is None or abs(old_ap-new_ap)<1e-7;reference_errors.append(0. if old_ap is None else abs(old_ap-new_ap))
   if reference['conditional_RCA'] is not None:assert int(metrics['original']['root_rank_full']==1)==reference['conditional_RCA']['Top1']
   parents=np.flatnonzero(graph[:,root]);parents_identical=bool(np.array_equal(x[:,parents],normal[:,parents]))
   out=dict(system_id=f'c{c}/{i}',channels=c,root=root,outdegree=len(children),root_has_children=bool(len(children)),
    source_seed=int(pack['seed'][i]),source_array_SHA256=array_sha(x),normal_array_SHA256=array_sha(normal),measurement_only_array_SHA256=array_sha(isolated),
    root_observation_exactly_matched=True,all_nonroot_observations_exactly_normal=True,root_parents_factual_equal_normal=parents_identical,
    original_has_any_nonroot_response=bool(np.any(x[:,np.arange(c)!=root]!=normal[:,np.arange(c)!=root])),
    metrics=metrics,unrelated_score_null_error=unrelated_error,own_only_nonroot_null_error=own_only_error,
    child_score_change_from_bad_parent=float(direct_delta[mask][:,children].mean()) if mask.any() and len(children) else None,
    child_exceedance_delta=(metrics['measurement_only']['child_threshold_exceedance']-metrics['normal']['child_threshold_exceedance']) if mask.any() and len(children) else None)
   rows.append(out);atomic_json(a.output/'progress.json',dict(status='running',completed=len(rows),total=123,query_scalar_forwards=slots,optimizer_updates=0))
 assert len(rows)==123 and state_sha(model)==before
 atomic_json(a.output/'result.json',dict(status='complete',seed=a.seed,mode=a.mode,records=rows,parameter_sha_before=before,parameter_sha_after=state_sha(model),
  teacher_checkpoint_sha256=sha256(weight),source_manifest_sha256=cfg['source_manifest_sha256'],frozen_query_scalar_forwards=slots,optimizer_updates=0,
  original_AP_null_max_error=max(reference_errors),target_arrays_opened=False,local_commit=a.local_commit,github_commit=a.github_commit,
  scope='Measurement-only synthetic readout fault construction, not a physical state intervention; original may include differing fault types. RCA windows assessment only; no new1D/effect labels'))
 atomic_json(a.output/'progress.json',dict(status='complete',completed=123,total=123,query_scalar_forwards=slots,optimizer_updates=0))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['none','parents_current'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--local-commit',required=True);p.add_argument('--github-commit',required=True);a=p.parse_args()
 try:probe(a)
 except Exception:atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
