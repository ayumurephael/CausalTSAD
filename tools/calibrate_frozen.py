"""Source normal validation only; no target labels or target normal fitting."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from causal_tsad.io import atomic_json,sha256
from causal_tsad.frozen_detector import FrozenDetector

def main(a):
 import torch
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
 manifest=json.loads((a.source/'manifest.json').read_text());assert sha256(a.source/'manifest.json')=='bd24005add8469c106b89ee69a68bf1f6a20e7070d4dcf7f6a4e3b74dd4c2ac6'
 model=FrozenDetector(a.original_code,a.weights);values={};cases=0
 for row in manifest['files']:
  if row['split']!='source_validation':continue
  assert sha256(a.source/row['path'])==row['SHA256']
  with np.load(a.source/row['path']) as f:normal=np.array(f['normal'])
  assert normal.dtype==np.float32
  for x in normal:
   for group in ['official','selftrained']:
    name=group+('_uni' if x.shape[1]==1 else '_multi');pred=model.predict(x,name,'native')
    for representation,v in pred.items():values.setdefault(name+'/'+representation,[]).append(v)
   cases+=1
   if cases%20==0:atomic_json(a.root/'calibration_progress.json',dict(status='running',cases=cases,target_arrays_opened=False))
 thresholds={k:float(np.quantile(np.concatenate(v),.99)) for k,v in values.items()};model.verify_frozen()
 atomic_json(a.root/'calibration.json',dict(status='complete',thresholds=thresholds,source_validation_systems=cases,source_manifest_SHA=sha256(a.source/'manifest.json'),
   source_normal_inputs_float32=True,native_and_compat_source_preprocessing_exact=True,checkpoint_SHA256={name:sha256(a.weights/(name+'.pth')) for name in model.models},
   actual_input_scalars=model.actual_input_scalars,padded_forward_scalars=model.padded_forward_scalars,encoder_calls=model.encoder_calls,parameter_SHA_unchanged=True,target_arrays_opened=False,optimizer_updates=0))
 atomic_json(a.root/'calibration_progress.json',dict(status='complete',cases=cases,target_arrays_opened=False));print('source calibration complete')

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['root','source','original-code','weights']:p.add_argument('--'+k,type=Path,required=True)
 main(p.parse_args())
