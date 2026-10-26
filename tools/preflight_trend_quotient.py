"""Check a trend-null is exact in algebra but retains a known level-fault blind spot."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from proposals.trend_quotient import second_difference,TrendQuotientContrast
from proposals.benign_nuisance import transform
from proposals.raw_context import parameter_sha
from causal_tsad.io import atomic_json
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory);rng=np.random.default_rng(348);x=rng.normal(size=(2,1024,20));x[1,:256]=x[0,:256];cases=np.concatenate([x,x.copy()]);cases[2,256:,3]=x[1,256:,3];cases[3,256:,3]=x[0,256:,3];row=dict(cases=cases,key='348');z=torch.tensor(cases[0],dtype=torch.float64,device='cuda');drift=torch.tensor(transform(row,8)['cases'][0],dtype=torch.float64,device='cuda');error=float((second_difference(z)-second_difference(drift)).abs().max());assert error<1e-10
 m=TrendQuotientContrast().cuda();m.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model']);before=parameter_sha(m);g=torch.zeros(20,20,device='cuda');g[torch.arange(19),torch.arange(1,20)]=1
 with torch.no_grad():
  base=m(z,g,'true');linear=m(drift,g,'true');prob=float((base['logit'].sigmoid()-linear['logit'].sigmoid()).abs().max());assert prob<1e-4;future=z.clone();future[900:]+=1e6;changed=m(future,g,'true');assert torch.equal(base['logit'][:580],changed['logit'][:580])
  shifted=z.clone();shifted[384:,3]+=10;assert torch.allclose(second_difference(shifted)[386:],second_difference(z)[386:],atol=1e-12,rtol=0)
 # Complete-trajectory class matching must hold after filtering, including boundary terms.
 filtered=[second_difference(torch.tensor(v,dtype=torch.float64,device='cuda')).float().cpu().numpy() for v in cases]
 for j in range(20):
  import hashlib
  assert sorted(hashlib.sha256(v[:,j].tobytes()).hexdigest() for v in filtered[:2])==sorted(hashlib.sha256(v[:,j].tobytes()).hexdigest() for v in filtered[2:])
 o=m(z,g,'true');(o['logit'].square().mean()+o['root_logits'].square().mean()+.05*o['normal_NLL'].mean()).backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);assert parameter_sha(m)==before
 atomic_json(a.output,dict(status='passed',linear_frontend_max_error=error,linear_probability_max_error=prob,whole_trajectory_class_matching=True,future_isolation_exact=True,long_level_shift_interior_removed=True,parameters_unchanged=True,optimizer_updates=0,benchmark_arrays_opened=False,technical_forward_query_scalars=4*704*20,technical_backward_query_scalars=704*20,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
