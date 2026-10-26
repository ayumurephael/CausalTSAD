"""Numerically compare rank-one integration to its dense Gaussian formula."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from proposals.sparse_edge_alternatives import rank1_bf,window_sum,heads
from proposals.conditional_change_evidence import integrated_bf,energy_bf
from causal_tsad.io import atomic_json
def run(output):
    torch.set_num_threads(1);torch.manual_seed(348);dtype=torch.float64
    r=torch.randn(2,64,device='cuda',dtype=dtype);v=torch.rand_like(r)+.5;f=torch.randn(2,64,1,device='cuda',dtype=dtype)
    error=float((rank1_bf(r,v,f,16)-integrated_bf(r,v,f)).abs().max());assert error<1e-9
    x=torch.randn(2,384,4,device='cuda',dtype=dtype);g=torch.zeros(4,4,device='cuda',dtype=dtype);g[0,1]=g[1,2]=1
    rr=torch.randn(2,64,4,device='cuda',dtype=dtype);vv=torch.rand_like(rr)+.5
    with torch.no_grad():a,au=heads(x,g,rr,vv,'348');changed=x.clone();changed[:,360:]+=100;b,_=heads(changed,g,rr,vv,'348')
    future=max(float((a[k]['score'][:,:40]-b[k]['score'][:,:40]).abs().max()) for k in a);assert future==0
    replay=float((a['energy16']['score']-(torch.logsumexp(energy_bf(rr.transpose(1,2).reshape(-1,64),vv.transpose(1,2).reshape(-1,64)).reshape(2,4,64).transpose(1,2),-1)-__import__('math').log(4))).abs().max());assert replay<1e-9
    assert torch.equal(rank1_bf(r,v,torch.zeros_like(f),16),torch.zeros_like(r))
    atomic_json(output,dict(status='passed',rank1_dense_single_feature_max_error=error,energy16_old_replay_max_error=replay,future_invariance_error=future,zero_feature_BF_exact=True,technical_reused_query_node_time=2*64*4*2,new_backbone_forwards=0,optimizer_updates=0,target_arrays_opened=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
