"""Only missing C04 source train/calibration B0 forwards; old/validation caches reused."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from causal_tsad.frozen_detector import FrozenDetector
from causal_tsad.io import atomic_json,sha256
from fusion_source import raw_group
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['plan','root','original-code','weights']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(1*2**30/torch.cuda.get_device_properties(0).total_memory)
 plan=json.loads(a.plan.read_text());rows=[r for r in plan['rows'] if r['base_missing']][a.worker::a.workers];packs={};model=FrozenDetector(a.original_code,a.weights)
 for i,row in enumerate(rows):
  cases,labels,roots,window,graph=raw_group(plan,row,packs);values=[]
  for v in cases:values.append(model.predict(v,'selftrained_multi','native')['logit'][320:])
  path=a.root/'base/arrays'/(row['id']+'.npz');assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,base=np.stack(values),labels=labels,root=roots,root_window=window,graph=graph)
  atomic_json(a.root/'base/records'/(row['id']+'.json'),dict(id=row['id'],path=str(path.relative_to(a.root)),SHA256=sha256(path),reused=False))
  atomic_json(a.root/'base'/('worker_'+str(a.worker)+'.json'),dict(status='running',completed=i+1,total=len(rows),actual_input_scalars=model.actual_input_scalars,encoder_calls=model.encoder_calls))
 model.verify_frozen();atomic_json(a.root/'base'/('worker_'+str(a.worker)+'.json'),dict(status='complete',completed=len(rows),actual_input_scalars=model.actual_input_scalars,padded_forward_scalars=model.padded_forward_scalars,encoder_calls=model.encoder_calls,
  parameter_SHA_unchanged=True,optimizer_updates=0,target_arrays_opened=False,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
