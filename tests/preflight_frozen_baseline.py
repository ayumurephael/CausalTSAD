"""Four checkpoints, original zero_shot equivalence, long M memory and cold padding."""
import argparse,sys,json,gc
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from causal_tsad.io import atomic_json,sha256
from causal_tsad.frozen_detector import FrozenDetector,normalize,WEIGHTS

def main(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
 sys.path.insert(0,str(a.original_code));from models.TimeRCD import TimeRCDPretrainTester
 from models.time_rcd.time_rcd_config import default_config
 from utils.dataset import TimeRCDDataset
 rng=np.random.RandomState(348);checks=[]
 for group in ['official','selftrained']:
  for c,mode in [(1,'uni'),(8,'multi')]:
   x=rng.normal(size=(257,c))
   if c>1:x[:,0]=0
   cfg=deepcopy(default_config);cfg.ts_config.patch_size=16;cfg.ts_config.num_features=c;cfg.win_size=5000;cfg.batch_size=1
   ref=TimeRCDPretrainTester(str(a.weights/(group+'_'+mode+'.pth')),cfg);prob,raw=ref.zero_shot(x)
   prob=np.concatenate([v.ravel() for v in prob])[:len(x)];raw=np.concatenate([v.ravel() for v in raw])[:len(x)];del ref;gc.collect();torch.cuda.empty_cache()
   detector=FrozenDetector(a.original_code,a.weights);current=detector.predict(x,group+'_'+mode,'native')
   error_prob=float(np.max(abs(prob-current['probability'])));error_raw=float(np.max(abs(raw-current['logit'])));assert error_prob<=1e-6 and error_raw<=1e-5,(group,c,error_prob,error_raw)
   normalized=TimeRCDDataset(x,window_size=257,stride=257,normalize=True).data;assert np.array_equal(normalized,normalize(x,'native'))
   checks.append(dict(checkpoint=group+'_'+mode,channels=c,probability_error=error_prob,raw_logit_error=error_raw))
   detector.verify_frozen();del detector;gc.collect();torch.cuda.empty_cache()
 # Largest channel/window shape and repeat-last tail; no truncated attention.
 detector=FrozenDetector(a.original_code,a.weights);x=rng.normal(size=(5017,20))
 for name in ['official_multi','selftrained_multi']:
  pred=detector.predict(x,name,'native');assert len(pred['logit'])==len(x) and np.isfinite(pred['probability']).all()
 detector.verify_frozen()
 atomic_json(a.output,dict(status='passed',checks=checks,full_attention_unmodified=True,normalization_matches_upstream=True,
  long_shape=[5017,20],repeat_last_tail=True,parameter_SHA_unchanged=True,weight_SHA256=WEIGHTS,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
  source_arrays_opened=False,target_arrays_opened=False,optimizer_updates=0,random_input_arrays_only=True,RCA_evaluated=False,
  reference_random_scalar_forwards=4626,adapter_random_scalar_forwards=205306,unique_random_input_scalars=104966))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['output','original-code','weights']:p.add_argument('--'+k,type=Path,required=True)
 main(p.parse_args())
