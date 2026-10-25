"""Non-vacuous context isolation, causal guard, active-history and capacity checks."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from causal_tsad.io import atomic_json
from proposals.c09.context_model import ConditionalTail,TailFusion,ARMS

def main(output):
 torch.set_num_threads(1);torch.manual_seed(348);torch.backends.cuda.matmul.allow_tf32=False
 model=ConditionalTail().cuda().eval();fusion=TailFusion().cuda();x=torch.randn(1,384,4,12,device='cuda');r=torch.randn(1,384,4,device='cuda');t=320
 # Nonzero decoder prevents a vacuous null test at zero residual initialization.
 with torch.no_grad():torch.nn.init.normal_(model.density.net[-1].weight,std=.05)
 errors={}
 for arm in ARMS:
  q=model(x,arm,r)['quantiles'];z=r.clone();z[:,t]+=100
  own=float((model(x,arm,z)['quantiles'][:,t]-q[:,t]).abs().max());assert own<1e-6
  z=r.clone();z[:,t+1:]+=100;v=x.clone();v[:,t+1:]+=100
  future=float((model(v,arm,z)['quantiles'][:,:t+1]-q[:,:t+1]).abs().max());assert future<1e-6
  z=r.clone();z[:,t-16:]+=100
  guard=float((model(x,arm,z)['quantiles'][:,t]-q[:,t]).abs().max());assert guard<1e-6
  z=r.clone();z[:,t-100:t-50,0]+=10
  q2=model(x,arm,z)['quantiles'];nonself=float((q2[:,:,:,][...,1:,:]-q[...,1:,:]).abs().max());assert nonself<1e-6
  active=float((q2[:,t,0]-q[:,t,0]).abs().max())
  if arm=='no_context':assert active==0
  else:assert active>1e-5,'Guarded historical context must actually affect predictive quantiles'
  perm=torch.tensor([2,0,3,1],device='cuda');equiv=float((model(x[:,:,perm],arm,r[:,:,perm])['quantiles']-q[:,:,perm]).abs().max());assert equiv<1e-5
  ready=torch.arange(x.shape[1],device='cuda')[None]>=256;loss,_=model.objective(x,r,arm,ready);model.zero_grad();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
  node,_=model.evidence(x,r,arm);assert torch.isfinite(node).all() and (q.diff(dim=-1)>0).all()
  base=torch.randn(1,x.shape[1],device='cuda');out=fusion(node,base,ready,arm);assert torch.equal(out,base)
  with torch.no_grad():torch.nn.init.normal_(fusion.net[-1].weight,std=.01)
  assert torch.equal(fusion(node,base,ready,arm)[:,:256],base[:,:256]);assert torch.equal(fusion(node[:,:,:1],base,ready,arm),base)
  with torch.no_grad():fusion.net[-1].weight.zero_()
  errors[arm]=dict(own_query_error=own,future_error=future,guard_error=guard,unrelated_channel_error=nonself,permutation_error=equiv,active_history_effect=active)
 atomic_json(output,dict(status='passed',checks=errors,tail_parameters=sum(p.numel() for p in model.parameters()),fusion_parameters=sum(p.numel() for p in fusion.parameters()),
  target_arrays_opened=False,source_arrays_opened=False,optimizer_updates=0,random_predictive_feature_scalars=x.numel(),random_history_residual_scalars=r.numel(),
  current_or_future_residual_not_used_for_own_distribution=True,C1_identity=True,warmup256_identity=True,
  scope='C09 upstream raw-input parent/query/future invariance inherited with exact cache SHA; this test covers new learned context operator'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
