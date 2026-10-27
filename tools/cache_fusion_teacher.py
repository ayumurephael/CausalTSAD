"""Frozen C04 source features; same input/B0 SHA for every graph control."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from proposals.causal_contrast import CausalContrast
from proposals.matched_pairs import rewire
from proposals.protected_fusion import causal_features,parameter_sha
from causal_tsad.io import atomic_json,sha256
from fusion_source import raw_group
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['root','plan','checkpoint','output']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(1*2**30/torch.cuda.get_device_properties(0).total_memory)
 plan=json.loads(a.plan.read_text());saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m=CausalContrast().cuda().eval().requires_grad_(False);m.load_state_dict(saved['model'],strict=True);before=parameter_sha(m);packs={};records=[];query=0;start=time.monotonic()
 for i,row in enumerate(plan['rows']):
  assert time.monotonic()-start<2*3600
  cases,labels,roots,window,graph=raw_group(plan,row,packs);g=torch.from_numpy(graph).cuda()
  if saved['mode']=='wrong':g=rewire(g,row['key'])
  bp=a.root/'base/arrays'/(row['id']+'.npz');brec=json.loads((a.root/'base/records'/(row['id']+'.json')).read_text());assert sha256(bp)==brec['SHA256']
  with np.load(bp) as f:base=np.array(f['base']);assert np.array_equal(labels,f['labels'])
  arrays=[];root_logits=[]
  for j,v in enumerate(cases):
   x=torch.tensor(v,dtype=torch.float32,device='cuda');f,r=causal_features(m,x,g,saved['mode'],chunk=256)
   feature=torch.cat([torch.from_numpy(base[j]).cuda()[:,None],f,torch.tensor([float(torch.log1p(g.sum(0).mean())),float(np.log(row['channels']))],device='cuda')[None].expand(len(f),-1)],-1)
   assert torch.isfinite(feature).all();arrays.append(feature.cpu().numpy());root_logits.append(r.cpu().numpy());query+=len(f)*row['channels']
  path=a.output/'arrays'/(row['id']+'.npz');path.parent.mkdir(exist_ok=True);data=dict(features=np.stack(arrays),base=base,labels=labels,root=roots,root_window=window)
  if row['split'].startswith('val'):data['root_logits']=np.stack(root_logits)
  np.savez_compressed(path,**data);records.append(dict(row,path=str(path.relative_to(a.output)),SHA256=sha256(path),base_SHA256=brec['SHA256']))
  atomic_json(a.output/'progress.json',dict(status='running',completed=i+1,total=len(plan['rows']),frozen_query_predictions=query,optimizer_updates=0,target_arrays_opened=False))
 assert parameter_sha(m)==before
 atomic_json(a.output/'manifest.json',dict(status='complete',rows=records,mode=saved['mode'],seed=saved['seed'],teacher_checkpoint_SHA256=sha256(a.checkpoint),teacher_parameter_SHA256=before,
  plan_SHA256=sha256(a.plan),frozen_query_predictions=query,parameters_unchanged=True,target_arrays_opened=False,optimizer_updates=0))
 atomic_json(a.output/'progress.json',dict(status='complete',completed=len(records),frozen_query_predictions=query,parameters_unchanged=True,optimizer_updates=0,target_arrays_opened=False,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
