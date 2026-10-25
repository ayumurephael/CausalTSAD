"""GPU inference separated from CPU metrics so CPUs do not reserve GPU queue slots."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,json,pickle,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from causal_tsad.io import atomic_json,sha256,array_sha
from causal_tsad.frozen_detector import FrozenDetector,methods,normalize

def main(a):
 import torch
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
 model=FrozenDetector(a.original_code,a.weights);jobs=json.loads(a.jobs.read_text());status=dict(status='running',completed=0,total=len(jobs),optimizer_updates=0,pid=os.getpid());current=None;pack=None
 start_time=time.monotonic();atomic_json(a.root/'progress'/(a.jobs.stem+'.json'),status)
 for row in jobs:
  assert row['split']=='development'
  if time.monotonic()-start_time>6*3600:raise TimeoutError('Frozen inference6h endpoint')
  if current!=row['shard']:
   with (a.benchmark/'data'/row['shard']).open('rb') as f:pack=pickle.load(f)
   current=row['shard']
  sample=pack[row['offset']];assert sample['system_id']==row['system_id'];results={};ident=row['id'];reuse=[]
  assert array_sha(np.asarray(sample['time_series']))==row['time_series_sha256'] and array_sha(np.asarray(sample['normal_time_series']))==row['normal_sha256']
  for method,weight in methods(row['channels']):
   for view,field in [('observed','time_series'),('normal','normal_time_series')]:
    x=np.asarray(sample[field]);assert x.ndim==2 and np.isfinite(x).all()
    pred=model.predict(x,weight,'native')
    for representation,value in pred.items():results[f'{method}__native__{view}__{representation}']=value
    # Exact input identity is a safe compute reuse, never a tolerance-based skip.
    equal=np.array_equal(normalize(x,'native').astype(np.float32),normalize(x,'causal_compat').astype(np.float32))
    alt=pred if equal else model.predict(x,weight,'causal_compat')
    if equal:reuse.append(method+'/'+view)
    for representation,value in alt.items():results[f'{method}__causal_compat__{view}__{representation}']=value
  p=a.root/'scores'/(ident+'.npz');p.parent.mkdir(exist_ok=True);assert not p.exists();np.savez_compressed(p,**results)
  atomic_json(a.root/'inference'/(ident+'.json'),dict(system_id=row['system_id'],scores_SHA256=sha256(p),observed_input_SHA=row['time_series_sha256'],normal_input_SHA=row['normal_sha256'],
    exact_normalized_input_reuses=reuse,checkpoint_SHA256={k:sha256(a.weights/(k+'.pth')) for _,k in methods(row['channels'])},RCA_evaluated=False))
  status.update(completed=status['completed']+1,actual_input_scalars=model.actual_input_scalars,padded_forward_scalars=model.padded_forward_scalars,encoder_calls=model.encoder_calls);atomic_json(a.root/'progress'/(a.jobs.stem+'.json'),status)
 model.verify_frozen();status.update(status='complete',parameter_SHA_unchanged=True,peak_cuda_bytes=torch.cuda.max_memory_allocated());atomic_json(a.root/'progress'/(a.jobs.stem+'.json'),status)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['root','benchmark','original-code','weights','jobs']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 try:main(a)
 except Exception:atomic_json(a.root/'progress'/(a.jobs.stem+'_error.json'),dict(status='needs_attention',error=traceback.format_exc()));raise
