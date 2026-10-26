"""Hard input/gradient/closed-form checks on independent technical random data."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from proposals.causal_meta_basis import CausalMetaBasis,relation_loss
from causal_tsad.io import atomic_json
def run(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(349);m=CausalMetaBasis().cuda();x=torch.randn(4,400,3,device='cuda',dtype=torch.float64);g=torch.tensor([[0,1,0],[0,0,1],[0,0,0]],device='cuda',dtype=torch.float64)
    out=m(x,g);loss=out['normal_nll'][:2].mean()+.25*relation_loss(out['score']);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    changes={n:float(p.grad.abs().max()) for n,p in m.named_parameters()};assert max(changes.values())>0
    m.zero_grad();q=350;node=1;y=x.clone();y[:,q:,node]+=17
    # Mean at q is allowed to see parents at q, never own q or future.
    with torch.no_grad():
        changed=m(y,g);own=float((changed['mean'][:,q-320,node]-out['mean'][:,q-320,node]).abs().max());assert own==0
        y=x.clone();y[:,q+1:]+=17;future=float((m(y,g)['score'][:,:q-320+1]-out['score'][:,:q-320+1]).abs().max());assert future==0
        perm=torch.tensor([2,0,1],device='cuda');other=m(x[:,:,perm],g[perm][:,perm]);error=float((other['score']-out['score']).abs().max());assert error<1e-8
        # A non-parent's current value cannot affect this node's normal prediction.
        y=x.clone();y[:,q,2]+=11;nonparent=float((m(y,g)['mean'][:,q-320,1]-out['mean'][:,q-320,1]).abs().max());assert nonparent==0
        gc=torch.zeros(20,20,device='cuda',dtype=torch.float64)
        for child in range(20):gc[max(0,child-4):child,child]=1
        z=torch.randn(2,340,20,device='cuda',dtype=torch.float64);assert torch.isfinite(m(z,gc)['score']).all()
    # Independent batch ridge solve agrees with Cholesky implementation.
    f=torch.randn(2,24,7,device='cuda',dtype=torch.float64);yy=torch.randn(2,24,1,device='cuda',dtype=torch.float64);a0=torch.eye(7,device='cuda',dtype=torch.float64)+f.transpose(1,2)@f;b=f.transpose(1,2)@yy;ce=float((torch.linalg.solve(a0,b)-torch.cholesky_solve(b,torch.linalg.cholesky(a0))).abs().max());assert ce<1e-10
    receipt=dict(status='passed',own_current_mean_error=own,future_score_error=future,nonparent_current_mean_error=nonparent,permutation_error=error,closed_form_solve_error=ce,gradients=changes,technical_input_slots=int(4*400*3*5+2*340*20),technical_backward_input_slots=4*400*3,benchmark_arrays_opened=False)
    atomic_json(a.output,receipt);print(json.dumps(receipt))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args())
