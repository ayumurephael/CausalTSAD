"""Original official/selftrained TimeRCD, source paired TSAD only, no RCA adaptation."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from causal_tsad.frozen_detector import FrozenDetector
from causal_tsad.io import atomic_json,sha256
from train_causal_contrast import load_pool
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['source','output','original-code','weights']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--source-sha256',required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(0).total_memory)
 data=load_pool(a.source,a.source_sha256);model=FrozenDetector(a.original_code,a.weights);results={};start=time.monotonic()
 for name in ['official_multi','selftrained_multi']:
  normal={rep:[] for rep in ['logit','probability']};records={}
  for row in data['calibration']:
   for view in [0,1]:
    out=model.predict(row['cases'][view],name,'native')
    for rep in normal:normal[rep].extend(out[rep][320:].tolist())
  thresholds={rep:float(np.quantile(v,.99)) for rep,v in normal.items()}
  for i,row in enumerate(data['source_validation']):
   assert time.monotonic()-start<4*3600;stored={};metrics={}
   for view in range(4):
    pred=model.predict(row['cases'][view],name,'native')
    for rep,v in pred.items():stored[rep+'_'+str(view)]=v[320:]
   for rep in normal:
    score=np.concatenate([stored[rep+'_'+str(k)] for k in range(4)]);y=np.repeat([0,0,1,1],704)
    metrics[rep]=dict(paired_AP=float(average_precision_score(y,score)),paired_ROC_AUC=float(roc_auc_score(y,score)),
     normal_FPR=float(np.mean(np.concatenate([stored[rep+'_0'],stored[rep+'_1']])>thresholds[rep])))
   arrays=a.output/'scores'/name/(row['key']+'.npz');arrays.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(arrays,**stored)
   records[row['key']]=dict(channels=row['cases'].shape[-1],family=row['family'],prefix_contaminated=row['prefix_contaminated'],scores_SHA256=sha256(arrays),**metrics)
   atomic_json(a.output/'progress.json',dict(status='running',weight=name,completed=i+1,total=128,encoder_calls=model.encoder_calls,actual_input_scalars=model.actual_input_scalars))
  model.verify_frozen();results[name]=dict(records=records,source_thresholds=thresholds,RCA_evaluated=False)
  atomic_json(a.output/(name+'.json'),results[name])
 atomic_json(a.output/'progress.json',dict(status='complete',checkpoints=list(results),RCA_evaluated=False,parameter_SHA_unchanged=True,
  encoder_calls=model.encoder_calls,actual_input_scalars=model.actual_input_scalars,padded_forward_scalars=model.padded_forward_scalars,
  source_manifest_SHA256=a.source_sha256,target_arrays_opened=False,optimizer_updates=0,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
  scope='Source paired normal/fault query AP prior50%, not a replacement for completed original4 benchmark table'))
