"""G0 of deterministic causal context and normal predictor; random arrays only."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json
from proposals.c03.data import encode


def main(a):
    import torch
    from proposals.c03.model import ContextMechanism,loss_and_score
    torch.set_num_threads(1);torch.manual_seed(348)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    x=np.random.RandomState(348).normal(size=(512,4)).astype(np.float32)
    data,count=encode(x);changed=x.copy();changed[256:]+=123;other,_=encode(changed)
    for key in ['raw','fraction','mean','median']:
        assert np.array_equal(data[key][:17],other[key][:17]),key+' leaked query/future'
    changed=x.copy();changed[:,3]+=np.arange(len(x))*10;other,_=encode(changed)
    for key in ['raw','fraction']:assert np.allclose(data[key][:,:3],other[key][:,:3],atol=1e-6,rtol=0)
    for key in ['mean','median']:assert np.allclose(data[key][:,:3,:3],other[key][:,:3,:3],atol=1e-6,rtol=0)
    raw=torch.from_numpy(data['raw'])[None].cuda();fraction=torch.from_numpy(data['fraction'])[None].cuda()
    ctx=torch.from_numpy(data['mean'])[None].cuda();graph=torch.zeros(1,4,4,device='cuda');graph[:,0,1]=1;graph[:,1,2]=1
    model=ContextMechanism().cuda();mu,lv=model(raw,fraction,graph,ctx)
    other_raw=torch.from_numpy(other['raw'])[None].cuda();other_ctx=torch.from_numpy(other['mean'])[None].cuda()
    changed_mu=model(other_raw,fraction,graph,other_ctx)[0]
    assert torch.allclose(mu[:,:,:,1],changed_mu[:,:,:,1],atol=1e-6,rtol=0)
    p=torch.tensor([3,0,2,1],device='cuda')
    new=model(raw[:,:,p],fraction[:,:,p],graph[:,p][:,:,p],ctx[:,:,p][:,:,:,p])[0]
    assert torch.allclose(mu[:,:,:,p],new,atol=1e-5,rtol=0)
    (mu.square().mean()+lv.square().mean()).backward()
    assert all(t.grad is None or torch.isfinite(t.grad).all() for t in model.parameters())
    constant,_=encode(np.ones((1024,1),np.float32))
    values={k:torch.from_numpy(constant[k])[None].cuda() for k in ['raw','fraction','query','valid']}
    values.update(context=torch.from_numpy(constant['mean'])[None].cuda(),graph=torch.zeros(1,1,1,device='cuda'))
    nll,score=loss_and_score(model,values);assert torch.isfinite(nll).all() and torch.isfinite(score).all()
    atomic_json(a.output,dict(status='passed',no_source_or_target_arrays=True,optimizer_updates=0,
        query_and_future_blind=True,nonparent_invariance_end_to_end=True,channel_equivariance=True,
        finite_backward=True,C1_constant_empty_graph_finite=True,parameters=sum(p.numel() for p in model.parameters()),
        technical_random_scalars=x.size,context_construction_calls=4,additional_constant_scalars=1024,
        query_sets_checked=17,normal_predictor_does_not_accept_query=True,peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
