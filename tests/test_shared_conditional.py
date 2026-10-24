"""Information-boundary and inference tests, not implementation mirrors."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baselines.shared_conditional import SharedConditional


def run(device='cpu'):
    torch.manual_seed(348)
    model=SharedConditional(history=8).to(device).eval()
    x=torch.randn(2,24,4,device=device)
    a=torch.zeros(2,4,4,device=device);a[:,0,1]=1;a[:,1,2]=1
    out=model(x,a)
    # Predictions for target 1 cannot read nonparent 3, its query value, or future.
    z=x.clone();z[:,:,3]=z[:,:,3]*90+10
    assert torch.allclose(out['mu'][:,:,1],model(z,a)['mu'][:,:,1],atol=1e-6,rtol=0)
    z=x.clone();z[:,10:,1]+=30
    assert torch.allclose(out['mu'][:,:11,1],model(z,a)['mu'][:,:11,1],atol=1e-6,rtol=0)
    z=x.clone();z[:,11:]+=20
    assert torch.allclose(out['mu'][:,:11],model(z,a)['mu'][:,:11],atol=1e-6,rtol=0)
    # Strict equivariance under a simultaneous channel/graph permutation.
    p=torch.tensor([3,1,0,2],device=device)
    other=model(x[:,:,p],a[:,p][:,:,p])
    assert torch.allclose(out['mu'][:,:,p],other['mu'],atol=2e-6,rtol=0)
    assert torch.allclose(out['logit'],other['logit'],atol=2e-6,rtol=0)
    # C1 and an empty-parent DAG remain finite, including at the first point.
    one=model(torch.ones(2,12,1,device=device),torch.zeros(2,1,1,device=device))
    assert all(torch.isfinite(v).all() for v in one.values())
    # CE cannot train the normal mechanism through a shortcut.
    model.zero_grad();out['logit'].square().mean().backward()
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('detector.'))
    assert any(p.grad is not None for n,p in model.named_parameters() if n.startswith('detector.'))
    return dict(nonparent_invariance=True,query_blind_prediction=True,future_blind=True,
                channel_equivariance=True,C1_constant_finite=True,CE_density_gradient_isolation=True,
                parameters=sum(p.numel() for p in model.parameters()))


if __name__=='__main__':
    import json
    print(json.dumps(run(sys.argv[1] if len(sys.argv)>1 else 'cpu')))
