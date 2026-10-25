"""Per-target query blindness even with same-time parent observations."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json


def main(a):
    import torch
    from proposals.c02.model import ParentTiming,predict
    torch.set_num_threads(1);torch.manual_seed(348)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=ParentTiming().cuda();values=torch.randn(1,320,4,device='cuda')
    graph=torch.zeros(1,4,4,device='cuda');graph[:,0,1]=1;graph[:,1,2]=1
    query_t=256;errors={}
    for timing in ['block','past','current']:
        mu,lv,_=predict(model,values,graph,timing)
        changed=values.clone();changed[:,query_t,1]+=100
        other=predict(model,changed,graph,timing)[0]
        own_error=float((mu[:,query_t,1]-other[:,query_t,1]).abs().max());assert own_error<1e-6
        future=values.clone();future[:,query_t+1:]+=100
        assert torch.allclose(mu[:,:query_t+1],predict(model,future,graph,timing)[0][:,:query_t+1],atol=1e-6,rtol=0)
        nonparent=values.clone();nonparent[:,:,3]+=30
        assert torch.allclose(mu[:,:,1],predict(model,nonparent,graph,timing)[0][:,:,1],atol=1e-6,rtol=0)
        p=torch.tensor([2,0,3,1],device='cuda')
        assert torch.allclose(mu[:,:,p],predict(model,values[:,:,p],graph[:,p][:,:,p],timing)[0],atol=1e-5,rtol=0)
        errors[timing]=own_error
    empty=torch.zeros_like(graph)
    outputs=[predict(model,values,empty,t)[0] for t in ['block','past','current']]
    assert all(torch.equal(outputs[0],v) for v in outputs[1:]),'No-graph timing is an exact null; do not retrain duplicate placebo arms'
    mu,lv,_=predict(model,values,graph,'current');(mu.square().mean()+lv.square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    constant=torch.ones(1,128,1,device='cuda');empty1=torch.zeros(1,1,1,device='cuda')
    for timing in ['block','past','current']:assert all(torch.isfinite(v).all() for v in predict(model,constant,empty1,timing))
    atomic_json(a.output,dict(status='passed',own_current_query_max_errors=errors,future_blind=True,
        nonparent_invariance=True,channel_equivariance=True,empty_graph_timing_exact_null=True,
        finite_backward=True,C1_constant_finite=True,parameters=sum(p.numel() for p in model.parameters()),
        optimizer_updates=0,random_array_scalars=1280,additional_constant_scalars=128,
        technical_random_forward_calls=19,technical_constant_forward_calls=3,
        technical_raw_forward_scalar_uses=19*1280+3*128,
        technical_logical_head_history_scalar_slots=2*(19*1280+3*128)*64,
        source_or_target_arrays_read=False,scope='same-time parent branch only; own query remains excluded'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
