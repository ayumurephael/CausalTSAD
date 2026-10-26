"""Frozen working-Gaussian coefficient-change evidence; not a true SCM posterior.

Residuals/variances and the normal mean are identical for every head. Actual
coefficients, noise, lag and fault roots are never inputs. Current parent values
are permitted. Normal parameter fitting uses only the first256 observed points.
"""
import hashlib, math
import numpy as np
import torch
import torch.nn.functional as F

FIRST=320
WINDOW=16
TAU2=(.0025,.04,.64)
BETA=(.5,2.,8.)

def rolling(x,w=WINDOW):
    # [B,Q,D] -> [B,Q,w,D]; first incomplete windows have zero evidence.
    return F.pad(x.transpose(1,2),(w-1,0)).unfold(-1,w,1).permute(0,2,3,1)

def integrated_bf(r,var,features):
    """Log p(r|D+tau2 ZZ')/p(r|D), uniformly integrated over three tau2."""
    rw=rolling(r[...,None]/var.sqrt()[...,None]).squeeze(-1)
    u=rolling(features/var.sqrt()[...,None])
    eye=torch.eye(WINDOW,device=r.device,dtype=r.dtype)
    gram=u@u.transpose(-1,-2);terms=[]
    for tau2 in TAU2:
        chol=torch.linalg.cholesky(eye+tau2*gram)
        solution=torch.cholesky_solve(rw[...,None],chol).squeeze(-1)
        logdet=2*chol.diagonal(dim1=-2,dim2=-1).log().sum(-1)
        terms.append(.5*((rw.square()).sum(-1)-(rw*solution).sum(-1)-logdet))
    return torch.logsumexp(torch.stack(terms),0)-math.log(len(terms))

def energy_bf(r,var):
    rw=rolling(r[...,None]/var.sqrt()[...,None]).squeeze(-1)
    count=torch.arange(1,r.shape[1]+1,device=r.device,dtype=r.dtype).clamp_max(WINDOW)
    energy=rw.square().sum(-1)
    terms=[.5*(b/(1+b)*energy-count*math.log1p(b)) for b in BETA]
    return torch.logsumexp(torch.stack(terms),0)-math.log(len(terms))

def partial_features(z,node,partners):
    t=torch.arange(8,z.shape[1],device=z.device)
    own=torch.stack([torch.ones_like(z[:,t,node])]+[z[:,t-k,node] for k in range(1,9)],-1)
    parent=torch.stack([z[:,t-k,p] for p in partners for k in range(9)],-1)
    a,b=own[:,:248],parent[:,:248]
    gram=a.transpose(1,2)@a+torch.eye(a.shape[-1],device=z.device,dtype=z.dtype)
    gamma=torch.cholesky_solve(a.transpose(1,2)@b,torch.linalg.cholesky(gram))
    residual=parent-own@gamma
    scale=residual[:,:248].std(1,unbiased=False).clamp_min(1e-3)
    return (residual/scale[:,None])[:,FIRST-8:]

def evidence_heads(values,graph,residual,variance,key):
    center=values[:,:256].median(1).values
    scale=(values[:,:256]-center[:,None]).abs().median(1).values*1.4826
    scale=torch.where(scale<1e-3,torch.ones_like(scale),scale)
    z=(values-center[:,None])/scale[:,None]
    energy=energy_bf(residual.transpose(1,2).reshape(-1,residual.shape[1]),
                     variance.transpose(1,2).reshape(-1,variance.shape[1]))
    energy=energy.reshape(len(values),values.shape[-1],-1).transpose(1,2)
    true=energy.clone();wrong=energy.clone();partners_audit={};factorizations=0
    for node in range(values.shape[-1]):
        parents=torch.where(graph[:,node]>0)[0].tolist()
        if not parents:continue
        candidates=[p for p in range(values.shape[-1]) if p!=node and p not in parents]
        assert len(candidates)>=len(parents),'Matched nonparent control unavailable'
        seed=int.from_bytes(hashlib.sha256((key+'/'+str(node)).encode()).digest()[:8],'little')
        nonparents=np.random.default_rng(seed).choice(candidates,len(parents),replace=False).tolist()
        partners_audit[str(node)]=dict(parents=parents,nonparents=nonparents)
        for dest,partners in [(true,parents),(wrong,nonparents)]:
            f=partial_features(z,node,partners)
            bf=integrated_bf(residual[:,:,node],variance[:,:,node],f)
            dest[:,:,node]=torch.logaddexp(energy[:,:,node],bf)-math.log(2)
            factorizations+=3*len(values)*residual.shape[1]+len(values)
    out={}
    for name,e in [('energy',energy),('parent',true),('nonparent',wrong)]:
        out[name]=dict(score=torch.logsumexp(e,-1)-math.log(e.shape[-1]),root=e.softmax(-1))
    return out,dict(partners=partners_audit,working_temporal_diagonal_covariance=True,
        factorization_count=factorizations,lag_bank=list(range(9)),WINDOW=WINDOW,TAU2=TAU2,BETA=BETA,
        global_parameter_updates=0,interpretation='working likelihood evidence, not calibrated causal posterior')
