"""Adversarial input changes check the raw predictive API before source optimization."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from proposals.raw_context import RawContextPredictor,prepare,ARMS,parameter_sha
from causal_tsad.io import atomic_json

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(348);torch.cuda.manual_seed_all(348);torch.backends.cuda.matmul.allow_tf32=False
 m=RawContextPredictor().cuda().eval();before=parameter_sha(m);x=torch.randn(384,4,device='cuda');g=torch.zeros(4,4,device='cuda');g[0,1]=1;g[0,2]=1;g[1,3]=1;results={};counts={'forward_queries':0,'backward_queries':0}
 def pred(x,g,arm,valid=None):
  f,y,_,_=prepare(x,valid);mu,lv=m.predict(f,g,arm);counts['forward_queries']+=y.numel();return torch.stack([mu,lv],-1)
 with torch.no_grad():
  for arm in ARMS:
   b=pred(x,g,arm);i=340-320
   own=x.clone();own[340,1]+=100;future=x.clone();future[341:]+=100
   other=x.clone();other[:,2]+=100;invalid=torch.ones(384,dtype=torch.bool,device='cuda');invalid[100]=False;v=x.clone();v[100]+=1000
   changed=x.clone();changed[340,0]+=10;perm=torch.tensor([2,0,3,1],device='cuda')
   r=dict(own_query_error=float((b[i,1]-pred(own,g,arm)[i,1]).abs().max()),future_error=float((b[i]-pred(future,g,arm)[i]).abs().max()),
    nonparent_error=float((b[:,1]-pred(other,g,arm)[:,1]).abs().max()),invalid_error=float((pred(x,g,arm,invalid)-pred(v,g,arm,invalid)).abs().max()),
    permutation_error=float((b[:,perm]-pred(x[:,perm],g[perm][:,perm],arm)).abs().max()),parent_current_effect=float((b[i,1]-pred(changed,g,arm)[i,1]).abs().max()))
   assert max(r[k] for k in ['own_query_error','future_error','nonparent_error','invalid_error'])==0
   assert r['permutation_error']<1e-5
   if not arm.startswith('degree'):assert r['parent_current_effect']>1e-8
   prefix=x.clone();prefix[64:256,0]*=-1
   r['historical_context_effect']=float((b[i,1]-pred(prefix,g,arm)[i,1]).abs().max())
   results[arm]=r
 assert parameter_sha(m)==before
 m.train();inputs,y,_,_=prepare(x);mu,lv=m.predict(inputs,g,'true_attention');loss=((mu-y).square()*torch.exp(-lv)+lv).mean();loss.backward()
 counts['forward_queries']+=y.numel();counts['backward_queries']+=y.numel()
 assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 m.zero_grad(set_to_none=True);long=torch.randn(1024,20,device='cuda');lg=torch.zeros(20,20,device='cuda');lg[torch.arange(19),torch.arange(1,20)]=1
 inputs,y,_,_=prepare(long);mu,lv=m.predict(inputs,lg,'true_attention');loss=((mu-y).square()*torch.exp(-lv)+lv).mean();loss.backward()
 counts['forward_queries']+=y.numel();counts['backward_queries']+=y.numel()
 assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 atomic_json(a.output,dict(status='passed',checks=results,parameters=sum(p.numel() for p in m.parameters()),optimizer_updates=0,source_or_target_arrays_opened=False,random_unique_scalars=x.numel()+long.numel(),extra_counts=counts,predictor_requires_no_query_targets=True,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
