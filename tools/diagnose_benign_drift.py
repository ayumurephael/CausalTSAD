"""Frozen source stress: matched class trajectories survive shared channel nuisance."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,argparse,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from sklearn.metrics import average_precision_score
from tools.train_causal_contrast import load_pool
from proposals.causal_contrast import CausalContrast
from proposals.protected_fusion import causal_features,parameter_sha
from causal_tsad.io import atomic_json,sha256
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['checkpoint','source','output']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--source-sha256',required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(1*2**30/torch.cuda.get_device_properties(0).total_memory)
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert saved['mode']=='true';model=CausalContrast().cuda().eval().requires_grad_(False);model.load_state_dict(saved['model'],strict=True);before=parameter_sha(model);data=load_pool(a.source,a.source_sha256);clean=saved['source_calibration']['records'];theta=saved['source_calibration']['source_threshold'];records=[];queries=0;start=time.monotonic()
 for row in data['source_validation']:
  assert time.monotonic()-start<2*3600;x=row['cases'];c=x.shape[-1];scale=x[:2,:256].reshape(-1,c).std(0);rng=np.random.default_rng(int(row['key']));sign=rng.choice([-1,1],c);ramp=np.arange(1024,dtype=np.float64)/1023
  for strength in [2.,8.]:
   shift=strength*ramp[:,None]*scale[None]*sign[None];z=x+shift[None]
   for j in range(c):
    lhs=sorted(hashlib.sha256(np.ascontiguousarray(v[:,j]).tobytes()).hexdigest() for v in z[:2]);rhs=sorted(hashlib.sha256(np.ascontiguousarray(v[:,j]).tobytes()).hexdigest() for v in z[2:]);assert lhs==rhs
   score=[];roots=[];normal=[]
   for view in range(4):
    f,r=causal_features(model,torch.tensor(z[view],dtype=torch.float32,device='cuda'),torch.from_numpy(row['graph']).cuda(),'true');s=f[:,0].cpu().numpy();score.append(s);queries+=704*c
    if view<2:normal.extend(s.tolist())
    else:roots.append(int(r.mean(0).argmax())==row['root'])
   records.append(dict(key=row['key'],C=c,family=row['family'],strength=strength,paired_AP=float(average_precision_score(np.repeat([0,0,1,1],704),np.concatenate(score))),normal_FPR_clean_threshold=float(np.mean(np.asarray(normal)>theta)),conditional_root_Top1=float(np.mean(roots)),clean=clean[row['key']],whole_channel_class_matching=True))
  atomic_json(a.output/'progress.json',dict(status='running',completed=len(records)//2,total=128,frozen_query_scalars=queries,optimizer_updates=0,target_arrays_opened=False))
 assert parameter_sha(model)==before;atomic_json(a.output/'results.json',dict(status='complete',seed=saved['seed'],records=records,checkpoint_SHA256=sha256(a.checkpoint),source_manifest_SHA256=a.source_sha256,frozen_query_scalars=queries,optimizer_updates=0,target_arrays_opened=False,parameters_unchanged=True,nuisance='same deterministic channelwise affine-in-time shift on all4views; time-dependent SCM, no extra graph edges',scope='prescribed source stress, not proof this explains every target failure'))
 atomic_json(a.output/'progress.json',dict(status='complete',completed=128,frozen_query_scalars=queries,optimizer_updates=0,target_arrays_opened=False,parameters_unchanged=True))
