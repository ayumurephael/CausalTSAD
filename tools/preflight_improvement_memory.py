"""Runtime-only recovery: identical output under larger cap, padded and repeated E4 inputs."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from causal_tsad.io import atomic_json
from causal_tsad.frozen_detector import FrozenDetector
def main(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 total=torch.cuda.get_device_properties(0).total_memory
 detector=FrozenDetector(a.detector_code,a.detector_weights)
 x=np.random.RandomState(79283).normal(size=(5000,40))
 torch.cuda.set_per_process_memory_fraction(8*2**30/total)
 old=detector.predict(x[:128,:6],'selftrained_multi','native')
 torch.cuda.set_per_process_memory_fraction(48*2**30/total)
 new=detector.predict(x[:128,:6],'selftrained_multi','native')
 error=float(np.max(np.abs(old['probability']-new['probability'])))
 assert error==0
 for length in [5000,5001,10000]:
  full=detector.predict(np.tile(x,(2,1))[:length],'selftrained_multi','native')
  assert len(full['logit'])==length and np.isfinite(full['logit']).all()
 detector.verify_frozen()
 atomic_json(a.output,dict(status='passed',memory_cap_probability_error=error,tested_shapes=[[5000,40],[5001,40],[10000,40]],peak_cuda_bytes=torch.cuda.max_memory_allocated(),technical_TimeRCD_scalars=detector.actual_input_scalars,technical_TimeRCD_calls=detector.encoder_calls,global_updates=0,benchmark_opened=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--detector-code',type=Path,required=True);p.add_argument('--detector-weights',type=Path,required=True);main(p.parse_args())
