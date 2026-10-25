"""G0: query/parent/normalization isolation, tail monotonicity, mass and frozen-base identity."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from causal_tsad.io import atomic_json
from proposals.c09.model import ConditionalTail,TailFusion,ARMS
from dependencies.c02_model import ParentTiming
from tools.cache_c09 import legal_views
def main(output):
 torch.set_num_threads(1);torch.manual_seed(348);torch.backends.cuda.matmul.allow_tf32=False
 teacher=ParentTiming().cuda().eval();tail=ConditionalTail().cuda();fusion=TailFusion().cuda();x=torch.randn(1,192,4,device='cuda');graph=torch.zeros(1,4,4,device='cuda');graph[:,0,1]=1
 context,residual,mu,lv=legal_views(teacher,x,graph);t=128
 changed=x.clone();changed[:,t,1]+=100;ctx,_,mu2,_=legal_views(teacher,changed,graph)
 own=max(float((ctx[:,t,1]-context[:,t,1]).abs().max()),float((mu2[:,t,1]-mu[:,t,1]).abs().max()));assert own<1e-6
 changed=x.clone();changed[:,t+1:]+=100;ctx,_,_,_=legal_views(teacher,changed,graph);future=float((ctx[:,:t+1]-context[:,:t+1]).abs().max());assert future<1e-6
 changed=x.clone();changed[:,:,3]+=100;ctx,_,_,_=legal_views(teacher,changed,graph);nonparent=float((ctx[:,:,1]-context[:,:,1]).abs().max());assert nonparent<1e-6
 perm=torch.tensor([2,0,3,1],device='cuda');ctx,r,_,_=legal_views(teacher,x[:,:,perm],graph[:,perm][:,:,perm]);equiv=float((ctx-context[:,:,perm]).abs().max());assert equiv<1e-5
 base=torch.randn(1,192,device='cuda');ready=torch.arange(192,device='cuda')[None]>=64
 checks={}
 for arm in ARMS:
  node,out=tail.evidence(context.detach(),residual.detach(),arm);assert torch.isfinite(node).all() and (out['quantiles'].diff(dim=-1)>0).all()
  q=out['quantiles'];point=tail(context.detach(),arm)['quantiles'];assert torch.isfinite(point).all()
  fused=fusion(node,base,ready,arm);err=float((fused-base).abs().max());assert err==0
  loss,_=tail.objective(context.detach(),residual.detach(),arm,ready);tail.zero_grad();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in tail.parameters())
  checks[arm]=dict(zero_fusion_error=err,finite_objective=float(loss),strictly_monotonic=True)
 # Extrapolation must stay finite and distinguish extreme values, without probability saturation.
 ctx=torch.zeros(1,2,1,12,device='cuda');extreme=torch.tensor([[[100.],[1000.]]],device='cuda')
 for arm in ARMS:
  e,_=tail.evidence(ctx,extreme,arm);assert torch.isfinite(e).all() and e[0,1,0]>e[0,0,0]
 with torch.no_grad():torch.nn.init.normal_(fusion.net[-1].weight)
 node=tail.evidence(context.detach(),residual.detach(),'conditional_quantile')[0]
 out=fusion(node,base,ready,'conditional_quantile');assert torch.equal(out[:,:64],base[:,:64])
 assert torch.equal(fusion(node[:,:,:1],base,ready,'conditional_quantile'),base)
 # Legal-context swap does not get normal twin or true parameters; no benchmark arrays used.
 atomic_json(output,dict(status='passed',own_query_context_error=own,future_context_error=future,nonparent_context_error=nonparent,channel_equivariance_error=equiv,
  strictly_monotonic_quantiles=True,finite_unsaturated_extreme_scores=True,warmup_identity=True,C1_identity=True,checks=checks,
  tail_parameters=sum(p.numel() for p in tail.parameters()),fusion_parameters=sum(p.numel() for p in fusion.parameters()),
  source_or_target_arrays_opened=False,optimizer_updates=0,random_observed_scalars=768,random_base_scalars=192,
  scope='Same-time direct parents allowed, own query/future excluded from predictive distribution; fusion remains original offline detector plus evidence'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
