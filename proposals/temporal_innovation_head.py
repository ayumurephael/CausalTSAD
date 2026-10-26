"""Small shared source-trained readout; frozen causal normal predictor external."""
import math
import torch
from torch import nn
import torch.nn.functional as F
WINDOWS=(1,4,16,64,256)

def trailing_mean(x,w):
    cs=F.pad(x.cumsum(1),(0,0,1,0));t=torch.arange(x.shape[1],device=x.device)
    lo=(t+1-w).clamp_min(0);n=(t+1-lo).to(x.dtype)
    return (cs[:,t+1]-cs[:,lo])/n[None,:,None]

def features(residual,variance):
    r=residual/variance.sqrt();out=[]
    for w in WINDOWS:
        m=trailing_mean(r,w);ma=trailing_mean(r.abs(),w);q=trailing_mean(r.square(),w);v=trailing_mean(variance,w)
        out.extend([m.sign()*torch.log1p(m.abs()),torch.log1p(ma),torch.log1p(q.clamp_min(0)),v.clamp_min(1e-12).log()])
    x=torch.stack(out,-1);assert torch.isfinite(x).all();return x.float()

class TemporalInnovationHead(nn.Module):
    def __init__(self,mode,seed):
        super().__init__();assert mode in ['point','fixed_mix','multiscale'];self.mode=mode
        with torch.random.fork_rng():
            torch.manual_seed(seed);self.network=nn.Sequential(nn.Linear(20,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
            nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
    def forward(self,x):
        # Common compressed point evidence. It is an explicit additional baseline,
        # not numerically equivalent to the old Gaussian-energy point readout.
        base=x[...,2]-math.log(3)
        shaped=x.reshape(*x.shape[:-1],5,4)
        if self.mode=='point':shaped=shaped[...,:1,:].expand_as(shaped)
        elif self.mode=='fixed_mix':shaped=shaped.mean(-2,keepdim=True).expand_as(shaped)
        logits=base+self.network(shaped.reshape_as(x)).squeeze(-1)
        return dict(logits=logits,score=torch.logsumexp(logits,-1)-math.log(logits.shape[-1]),root=logits.softmax(-1))

def objective(output,labels,roots):
    positive=labels>0;negative=~positive;assert positive.any() and negative.any()
    loss=.5*(F.softplus(-output['score'][positive]).mean()+F.softplus(output['score'][negative]).mean())
    target=roots[:,None].expand_as(labels)[positive]
    root=F.cross_entropy(output['logits'][positive],target)
    return loss+.25*root,loss,root
