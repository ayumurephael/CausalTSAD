"""Source-only numerical equivalence/null/no-own-current/feature-future checks."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from causal_tsad.io import atomic_json,sha256
from causal_tsad.frozen_detector import parameter_sha
from proposals.causal_meta_basis import CausalMetaBasis
from proposals.context_feature_oracle import diagnose
from proposals.context_crossfit import configurations,design,blocks,context_mask

def main(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 p=json.loads(a.plan.read_text());task=p['tasks'][0];row=task['row'];seed=task['seed']
 torch.cuda.set_per_process_memory_fraction(p['worker_GiB']*2**30/torch.cuda.get_device_properties(0).total_memory)
 with np.load(Path(p['source'])/row['path']) as z:x=z['values'];g=z['graph']
 with np.load(Path(p['prior_root'])/'context_cache'/(task['id']+'.npz')) as z:cache={k:z[k] for k in z.files}
 model=CausalMetaBasis(seed).cuda().eval().requires_grad_(False)
 model.load_state_dict(torch.load(Path(p['weights'])/f'normal_only_s{seed}.pth',map_location='cpu',weights_only=False)['model'])
 before=parameter_sha(model);graph=torch.tensor(g,dtype=torch.float64,device='cuda');errors={};null=0.;budget={}
 for view in [0,6]:
  o,bud=diagnose(model,torch.tensor(x[view],device='cuda'),torch.tensor(x[0],device='cuda'),graph,configurations()['xfit'])
  for k,v in bud.items():
   if isinstance(v,(int,float)) and not isinstance(v,bool):budget[k]=budget.get(k,0)+v
  for k in ['residual','variance']:
   errors[f'view{view}_{k}']=float(np.max(np.abs(o['OO'][k].cpu().numpy()-cache['true__'+k][view,288:])))
   if view==0:
    null=max(null,max(float((v[k]-o['OO'][k]).abs().max()) for v in o.values()))
 assert max(errors.values())<1e-8 and null<1e-12
 # Check the feature constructor alone, with fixed normalization, not the
 # transductive context estimator (which is explicitly allowed future context).
 z=torch.tensor(x[0],device='cuda');t=torch.tensor([512],device='cuda');maxown=maxfuture=0.
 for node in range(row['C']):
  pa=torch.where(graph[:,node]>0)[0].tolist();f,_=design(model,z,node,pa,t)
  zz=z.clone();zz[512,node]+=100;f2,_=design(model,zz,node,pa,t);maxown=max(maxown,float((f-f2).abs().max()))
  zz=z.clone();zz[513:]+=100;f2,_=design(model,zz,node,pa,t);maxfuture=max(maxfuture,float((f-f2).abs().max()))
 assert maxown==0 and maxfuture==0
 tt=np.arange(32,len(x[0]))
 for lo,hi in blocks(len(x[0]),8,'crossfit'):
  kept=tt[context_mask(len(x[0]),lo,hi,32,list(range(1,33)),'crossfit')]
  for lag in range(33):assert not (((kept-lag)>=lo)&((kept-lag)<hi)).any()
 assert parameter_sha(model)==before
 atomic_json(a.output,dict(status='passed',baseline_max_errors=errors,null_max_error=null,no_own_current_feature=maxown,no_future_query_feature=maxfuture,
  purged_fit_verified=True,parameters_unchanged=True,oracle_only=True,benchmark_arrays_opened=False,budget=budget,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
