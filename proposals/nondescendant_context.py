"""Candidate-specific negative-control context, not oracle root selection."""
import numpy as np
import torch

def donors(graph,node,mode):
    C=len(graph);seen={node};todo=[node]
    while todo:
        k=todo.pop()
        for j in torch.where(graph[k]>0)[0].tolist():
            if j not in seen:seen.add(j);todo.append(j)
    nd=[j for j in range(C) if j not in seen]
    others=[j for j in range(C) if j!=node]
    if mode=='nondesc':return nd
    if mode=='all':return others
    if mode=='sham':return np.random.default_rng(348+node).permutation(others)[:len(nd)].tolist()
    raise ValueError(mode)

def past_average(z,width=65):
    cs=torch.cat([torch.zeros_like(z[:1]),z.cumsum(0)],0)
    t=torch.arange(len(z),device=z.device)
    return (cs[t]-cs[(t-width).clamp_min(0)])/t.clamp(min=1,max=width)[:,None]

def context_basis(z,graph,node,t,mask,mode):
    ids=donors(graph,node,mode)
    if len(ids)<3:return torch.zeros((len(t),2),dtype=z.dtype,device=z.device)
    low=past_average(z)[:,ids][t]
    # Spatial loadings are estimated on purged CONTEXT rows; query covariates
    # use only donor history before t. No normal twin or true root is consulted.
    _,_,vh=torch.linalg.svd(low[mask],full_matrices=False)
    return low@vh[:2].T

def preflight():
    g=torch.zeros((5,5),dtype=torch.float64,device='cuda');g[0,1]=1;g[1,2]=1;g[3,4]=1
    z=torch.randn(400,5,dtype=torch.float64,device='cuda');t=torch.arange(32,400,device='cuda')
    m=t<180;x=context_basis(z,g,1,t,m,'nondesc')
    altered=z.clone();altered[:,1:3]+=100
    assert torch.equal(x,context_basis(altered,g,1,t,m,'nondesc'))
    # With fixed context, future donor changes cannot affect query feature t=200.
    altered=z.clone();altered[201:]+=100
    assert torch.equal(x[168],context_basis(altered,g,1,t,m,'nondesc')[168])
    assert donors(g,1,'nondesc')==[0,3,4]
    assert torch.allclose(past_average(z)[200],z[135:200].mean(0),atol=1e-12,rtol=0)
    return dict(status='passed',descendant_invariance=True,fixed_context_future_query_invariance=True,
                donor_features_strictly_past=True,rank=2,width=65)
