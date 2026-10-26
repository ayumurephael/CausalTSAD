"""Nuisance exact class matching and finite gradients, no optimizer/target arrays."""
import sys,argparse,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from torch.nn import functional as F
from proposals.causal_contrast import CausalContrast
from proposals.benign_nuisance import transform
from proposals.raw_context import parameter_sha
from causal_tsad.io import atomic_json,sha256
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(0).total_memory);rng=np.random.default_rng(348);x=rng.normal(size=(2,1024,20));x[1,:256]=x[0,:256];cases=np.concatenate([x,x.copy()]);cases[2,256:,3]=x[1,256:,3];cases[3,256:,3]=x[0,256:,3];row=dict(cases=cases,key='348');aug=transform(row,8)
 for dtype in [np.float64,np.float32]:
  z=aug['cases'].astype(dtype)
  for j in range(20):assert sorted(hashlib.sha256(np.ascontiguousarray(v[:,j]).tobytes()).hexdigest() for v in z[:2])==sorted(hashlib.sha256(np.ascontiguousarray(v[:,j]).tobytes()).hexdigest() for v in z[2:])
 assert np.array_equal(transform(row,0)['cases'],cases);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert all(k in saved['config'] for k in ['normal_NLL_weight','root_loss_weight','weight_decay']);m=CausalContrast().cuda();m.load_state_dict(saved['model'],strict=True);before=parameter_sha(m);g=torch.zeros(20,20,device='cuda');g[torch.arange(19),torch.arange(1,20)]=1;loss=0
 for data,k in [(cases,0),(aug['cases'],0),(cases,2),(aug['cases'],2)]:
  out=m(torch.tensor(data[k],dtype=torch.float32,device='cuda'),g,'true');loss+=F.binary_cross_entropy_with_logits(out['logit'],torch.full_like(out['logit'],float(k>=2)))+.05*out['normal_NLL'].mean()
 assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None);assert parameter_sha(m)==before
 atomic_json(a.output,dict(status='passed',FP64_FP32_whole_channel_class_matching=True,identity_exact=True,finite_gradients=True,parameters_unchanged=True,checkpoint_SHA256=sha256(a.checkpoint),optimizer_updates=0,source_or_target_arrays_opened=False,technical_forward_query_scalars=56320,technical_backward_query_scalars=56320,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
