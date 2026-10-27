"""I19: fixed, source-normal calibration and graph incidence readouts.

Empirical tail ranks are scores, not guaranteed p-values for dependent series.
All windows are trailing and divide by their actual available point count.
"""
import math
import numpy as np
import torch
from proposals.improvement3.evidence import window_sum

WINDOWS=(1,8,32,128)

@torch.no_grad()
def statistics(z):
    r=z['response']-z['mu'][...,0];v=z['variance'][...,0]
    e=r.square()/(3*v)-.5*math.log(3)
    edges=z['edges'].long();g=z['contribution'][...,0]
    child=edges[:,1]; node={};edge={};family={}
    for w in WINDOWS:
        count=torch.arange(1,len(r)+1,device=r.device,dtype=r.dtype).clamp(max=w)
        n=window_sum(e,w)/count.sqrt()[:,None]
        b=.5*window_sum(r[:,child]*g/v[:,child],w).square()/window_sum(g.square()/v[:,child],w).clamp_min(1e-8)
        node[w]=n.cpu().numpy();edge[w]=b.cpu().numpy()
        family['node'+str(w)]=n.max(-1).values.cpu().numpy()
        family['edge'+str(w)]=b.max(-1).values.cpu().numpy() if len(edges) else np.zeros(len(r))
    family['raw']=z['raw_logit'].cpu().numpy()
    return dict(node=node,edge=edge,family=family,edges=edges.cpu().numpy())

def tail(reference,x):
    ref=np.asarray(reference);return (1+len(ref)-np.searchsorted(ref,x,side='left'))/(len(ref)+1)

def calibrate(samples):
    # One C stratum, four independent normal systems, four views/system.
    refs={k:np.sort(np.concatenate([s['family'][k] for s in samples])) for k in samples[0]['family']}
    moments={}
    for w in WINDOWS:
        b=np.concatenate([s['edge'][w].reshape(-1) for s in samples])
        n=np.concatenate([s['node'][w].reshape(-1) for s in samples])
        moments[str(w)]=dict(edge_mean=float(b.mean()),edge_sd=float(b.std()),node_mean=float(n.mean()),node_sd=float(n.std()))
    return dict(refs=refs,moments=moments)

def scores(s,cal):
    p={k:tail(cal['refs'][k],v) for k,v in s['family'].items()}
    groups={'node':[k for k in p if k.startswith('node')],
            'node_edge':[k for k in p if k!='raw'],'node_edge_raw':list(p),'raw':['raw']}
    out={}
    for name,keys in groups.items():
        q=np.min([p[k] for k in keys],axis=0)
        out[name]=-np.log(q)
        # Keep signed scores for the monotone control; only clipping makes ties.
        out[name+'_bonf_shift']=-np.log(q*len(keys))
        out[name+'_bonf_clip']=-np.log(np.minimum(1.,q*len(keys)))
    return out

def roots(s,cal,w=32):
    n=s['node'][w];b=s['edge'][w];edges=s['edges'];C=n.shape[1]
    incidence=np.zeros((len(edges),C))
    for i,(a,c) in enumerate(edges):incidence[i,a]=incidence[i,c]=1.
    degree=incidence.sum(0).clip(1);m=cal['moments'][str(w)]
    raw=b@incidence;center=(b-m['edge_mean'])@incidence
    proxy=(b/np.maximum(np.median(b,axis=1,keepdims=True),1e-6)-1)@incidence
    nz=(n-m['node_mean'])/max(m['node_sd'],1e-6)
    cz=center/(max(m['edge_sd'],1e-6)*np.sqrt(degree))
    # Fixed edge-row permutation destroys correspondence, preserves graph degrees.
    order=np.random.default_rng(348).permutation(len(edges))
    sham=((b[:,order]-m['edge_mean'])@incidence)/(max(m['edge_sd'],1e-6)*np.sqrt(degree))
    return dict(node=n,edge_raw=raw,edge_degree=raw/degree,edge_proxy_median=proxy,
                edge_source_center=center,node_centered=nz+cz,node_sham=nz+sham)

def alarm_mask(score,high,low,gap=8):
    from proposals.improvement3.metrics import spans
    active=np.zeros(len(score),bool)
    for a,b in spans(score>low):
        if np.any(score[a:b]>high):active[a:b]=True
    ss=spans(active)
    for (_,b),(a,_) in zip(ss[:-1],ss[1:]):
        if a-b<=gap:active[b:a]=True
    return active
