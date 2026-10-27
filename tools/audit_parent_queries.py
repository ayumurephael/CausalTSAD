"""Frozen source normal diagnostics: alter current parent queries, preserve learned history."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from proposals.raw_context import RawContextPredictor,prepare,PARENT_LAGS,losses,parameter_sha
from causal_tsad.io import atomic_json,sha256
from train_raw_context import load

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ['source','config','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();assert not a.output.exists();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert saved['arm']=='true_mean'
 cfg=json.loads(a.config.read_text());m=RawContextPredictor().cuda().eval().requires_grad_(False);m.load_state_dict(saved['model'],strict=True);before=parameter_sha(m)
 rows=load(a.source,cfg)['source_validation'];records={};count=0;max_error=0
 with torch.no_grad():
  for row in rows:
   v=torch.from_numpy(row['normal']).cuda();graph=torch.from_numpy(row['graph']).cuda();inp,y,times,scale=prepare(v)
   center=v[:256].mean(0);z=(v-center)/scale
   stale=z[(times-128)[:,None]-torch.tensor(PARENT_LAGS,device='cuda')[None]].permute(0,2,1)
   out={}
   for name,parent in [('correct',inp['parent']),('stale128',stale),('zero_query',torch.zeros_like(inp['parent']))]:
    f=dict(inp,parent=parent);mu,lv=m.predict(f,graph,'true_mean');nll,mse,_=losses(mu,lv,y);out[name]=dict(normal_NLL=float(nll.mean()),normal_MSE=float(mse.mean()))
    count+=y.numel()
   old=saved['source_calibration']['records'][row['key']]
   error=max(abs(old[k]-out['correct'][k]) for k in ['normal_NLL','normal_MSE']);max_error=max(max_error,error);assert error<2e-5
   records[row['key']]=dict(channels=row['normal'].shape[1],**out)
 assert parameter_sha(m)==before
 atomic_json(a.output,dict(status='complete',seed=saved['seed'],checkpoint_SHA256=sha256(a.checkpoint),parameter_SHA_unchanged=True,
  baseline_reference_max_error=max_error,tolerance=2e-5,records=records,source_systems=len(rows),extra_frozen_query_predictions=count,
  optimizer_updates=0,target_arrays_opened=False,scope='Stale/zero parent query sensitivity with the same fixed learned context and target; diagnostic input corruption, not SCM-generated counterfactual or intervention effect identification'))
