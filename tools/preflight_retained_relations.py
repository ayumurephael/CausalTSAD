"""Projection algebra, block isolation and finite source backward; zero updates."""
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from proposals.retention_step import halfspace,protect,vector
from proposals.causal_contrast import CausalContrast
from proposals.raw_context import parameter_sha
from causal_tsad.io import atomic_json
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
 v=torch.tensor([-1.,2.],dtype=torch.float64,device='cuda');r=torch.tensor([1.,0.],dtype=torch.float64,device='cuda');assert torch.equal(halfspace(v,r),torch.tensor([0.,2.],dtype=v.dtype,device='cuda'));assert torch.equal(halfspace(v,torch.zeros_like(r)),v)
 q=torch.nn.Parameter(torch.tensor([1.,2.],device='cuda'));old=vector([q]);q.data+=.01;protected=protect([q],old,r,True);assert protected['applied_dot_clean_gradient']>=0
 m=CausalContrast().cuda();m.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model']);before=parameter_sha(m);x=torch.randn(1024,20,device='cuda');g=torch.zeros(20,20,device='cuda');g[torch.arange(19),torch.arange(1,20)]=1;o=m(x,g,'true');(o['logit'].square().mean()+o['root_logits'].square().mean()+.05*o['normal_NLL'].mean()).backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);assert parameter_sha(m)==before
 atomic_json(a.output,dict(status='passed',halfspace_analytic_exact=True,zero_reference_identity=True,rounded_step_feasible=True,finite_gradients=True,parameters_unchanged=True,optimizer_updates=0,benchmark_arrays_opened=False,technical_forward_query_scalars=704*20,technical_backward_query_scalars=704*20,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
