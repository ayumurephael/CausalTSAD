"""I09: learned normal-model mixing from observed protected context only.

No labels, twins, event metadata or true parameters enter components/selector.
Normal twins are used solely as source-training responses by the caller.
Offline context inference is disclosed; this is not an online predictor.
"""
import math
import torch
from proposals.context_crossfit import context_mask,blocks,design

FEATURES=('log_count','log_dimension','log_noise','mean_irls_weight',
          'design_energy','inverse_trace','own_beta_norm','intercept',
          'parent_count','parent_beta_norm','parent_fitted_variance','parent_response_covariance')

class ContextSelector(torch.nn.Module):
    def __init__(self,seed,hide_parent=False):
        super().__init__();torch.manual_seed(seed)
        self.net=torch.nn.Sequential(torch.nn.Linear(12,32),torch.nn.Tanh(),torch.nn.Linear(32,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
        self.hide_parent=hide_parent
    def forward(self,features,valid):
        x=features.clone()
        if self.hide_parent:x[...,8:]=0
        logits=self.net(x).squeeze(-1).masked_fill(~valid,-torch.inf)
        return torch.log_softmax(logits,-1)

@torch.inference_mode()
def components(model,values,graph):
    assert values.ndim==2 and values.dtype==torch.float64
    T,C=values.shape;device=values.device;t=torch.arange(32,T,device=device)
    mu=torch.zeros((T-320,C,5),device=device,dtype=values.dtype);var=torch.ones_like(mu)
    y=torch.zeros((T-320,C),device=device,dtype=values.dtype);center_out=y.clone();scale_out=torch.ones_like(y)
    # Features/weights constant within each protected query block, never
    # recomputed from its response. Arbitrary component order carries no ID.
    features=torch.zeros((*mu.shape,12),device=device,dtype=values.dtype);valid=torch.zeros_like(mu,dtype=torch.bool)
    counts=dict(input_scalars=values.numel(),feature_node_rows=0,factorizations=0,query_node_outputs=0)
    for node in range(C):
        pa=torch.where(graph[:,node]>0)[0].tolist()
        for lo,hi in blocks(T,8,'crossfit'):
            if hi<=320:continue
            m=torch.as_tensor(context_mask(T,lo,hi,32,list(range(1,33)),'crossfit'),device=device)
            query=(t>=max(lo,320))&(t<hi);outidx=t[query]-320
            assert m.sum()>=64
            center=values[t[m]].median(0).values
            scale=(values[t[m]]-center).abs().median(0).values*1.4826
            scale=torch.where(scale<1e-3,torch.ones_like(scale),scale)
            z=(values-center)/scale;raw,ps=design(model,z,node,pa,t)
            fm=raw[m].mean(0);fs=raw[m].std(0,unbiased=False).clamp_min(1e-3);fm[0]=0;fs[0]=1
            f=(raw-fm)/fs;q=f[query];y[outidx,node]=z[t[query],node]
            center_out[outidx,node]=center[node];scale_out[outidx,node]=scale[node]
            masks=[m]+[m&(t>=a)&(t<b) for a,b in [(32,280),(280,528),(528,776),(776,T)]]
            for k,mask in enumerate(masks):
                if mask.sum()<64:continue
                cf=f[mask];cy=z[t[mask],node];w=torch.ones_like(cy)
                eye=torch.eye(f.shape[-1],device=device,dtype=values.dtype)
                for _ in range(3):
                    chol=torch.linalg.cholesky(eye+cf.T@(w[:,None]*cf))
                    beta=torch.cholesky_solve((cf.T@(w*cy))[:,None],chol)[:,0]
                    e=cy-cf@beta;noise=((w*e.square()).sum()+beta.square().sum()+1)/(w.sum()+2)
                    noise=noise.clamp_min(1e-4);w=5/(4+e.square()/noise)
                inv=torch.cholesky_solve(q.T,chol).T
                mu[outidx,node,k]=q@beta;var[outidx,node,k]=noise*(1+(q*inv).sum(-1));valid[outidx,node,k]=True
                pv=cf[:,ps:]@beta[ps:] if pa else torch.zeros_like(cy)
                def signed_log(v):return v.sign()*torch.log1p(v.abs())
                st=torch.stack([torch.log(mask.sum().to(values.dtype)),torch.log(torch.as_tensor(f.shape[-1],device=device,dtype=values.dtype)),
                    noise.log(),w.mean(),torch.log1p(cf.square().mean()),torch.log1p(torch.cholesky_inverse(chol).trace()),
                    torch.log1p(beta[1:ps].norm()),signed_log(beta[0]),torch.as_tensor(len(pa),device=device,dtype=values.dtype),
                    torch.log1p(beta[ps:].norm()),torch.log1p(pv.var(unbiased=False)),signed_log(((pv-pv.mean())*(cy-cy.mean())).mean())]).clamp(-20,20)
                features[outidx,node,k]=st
                counts['factorizations']+=3;counts['query_node_outputs']+=int(query.sum())
            counts['feature_node_rows']+=len(t)
    assert valid[...,0].all() and torch.isfinite(mu).all() and torch.isfinite(var).all() and (var>0).all()
    return dict(mu=mu,variance=var,response=y,center=center_out,scale=scale_out,features=features,valid=valid,audit=counts)

def mixture_logpdf(response,mu,variance,log_weights):
    terms=-.5*(math.log(2*math.pi)+variance.log()+(response[...,None]-mu).square()/variance)
    return torch.logsumexp(log_weights+terms,-1)

def node_evidence(response,mu,variance,log_weights):
    # Same variance-inflation likelihood ratio as C04/I06 for one component.
    return mixture_logpdf(response,mu,3*variance,log_weights)-mixture_logpdf(response,mu,variance,log_weights)

def aggregate(evidence,temperature,bias):
    return temperature*(torch.logsumexp(evidence/temperature,-1)-math.log(evidence.shape[-1]))+bias
