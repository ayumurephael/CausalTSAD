"""Unknown single coefficient/lag alternative mixture, not an oracle posterior.

Normal prediction is frozen externally. Candidate parent lags are all0..8,
not actual generating lags. The coefficient perturbation is integrated rather
than supplied. These are working diagonal-noise likelihoods.
"""
import hashlib,math
import numpy as np
import torch
from proposals.conditional_change_evidence import partial_features,TAU2,BETA,FIRST

METHODS=('energy16','energy64','energy256','parent_partial16','nonparent_partial16',
         'parent_raw16','nonparent_raw16','parent_raw64','nonparent_raw64','parent_raw256','nonparent_raw256')

def window_sum(x,w):
    cs=torch.cat([torch.zeros_like(x[:,:1]),x.cumsum(1)],1)
    t=torch.arange(x.shape[1],device=x.device);return cs[:,t+1]-cs[:,(t+1-w).clamp_min(0)]

def rank1_bf(r,var,features,w):
    s=window_sum(features.square()/var[...,None],w).clamp_min(0)
    u=window_sum(features*r[...,None]/var[...,None],w)
    terms=[.5*(tau*u.square()/(1+tau*s)-torch.log1p(tau*s)) for tau in TAU2]
    # Uniform over candidate feature and scale; hypothesis count cannot increase
    # evidence just because the node has more parents.
    stacked=torch.stack(terms,-1).flatten(-2)
    return torch.logsumexp(stacked,-1)-math.log(stacked.shape[-1])

def energy_evidence(r,var,w):
    q=window_sum(r.square()/var,w);n=torch.arange(1,r.shape[1]+1,device=r.device,dtype=r.dtype).clamp_max(w)
    return torch.logsumexp(torch.stack([.5*(b/(1+b)*q-n[None,:,None]*math.log1p(b)) for b in BETA]),0)-math.log(len(BETA))

def heads(values,graph,residual,variance,key):
    center=values[:,:256].median(1).values
    scale=(values[:,:256]-center[:,None]).abs().median(1).values*1.4826
    scale=torch.where(scale<1e-3,torch.ones_like(scale),scale);z=(values-center[:,None])/scale[:,None]
    energies={w:energy_evidence(residual,variance,w) for w in [16,64,256]}
    nodes={name:energies[int(name.removeprefix('energy'))].clone() for name in METHODS if name.startswith('energy')}
    for name in METHODS:
        if not name.startswith('energy'):nodes[name]=energies[int(''.join(filter(str.isdigit,name)))].clone()
    audit={};candidate_applications=0;t=torch.arange(FIRST,values.shape[1],device=values.device)
    for node in range(values.shape[-1]):
        parents=torch.where(graph[:,node]>0)[0].tolist()
        if not parents:continue
        candidates=[p for p in range(values.shape[-1]) if p!=node and p not in parents];assert len(candidates)>=len(parents)
        seed=int.from_bytes(hashlib.sha256((key+'/'+str(node)).encode()).digest()[:8],'little')
        nonparents=np.random.default_rng(seed).choice(candidates,len(parents),replace=False).tolist()
        audit[str(node)]=dict(parents=parents,nonparents=nonparents)
        for tag,partners in [('parent',parents),('nonparent',nonparents)]:
            raw=torch.stack([z[:,t-lag,p] for p in partners for lag in range(9)],-1)
            partial=partial_features(z,node,partners)
            for orth,f,windows in [('raw',raw,[16,64,256]),('partial',partial,[16])]:
                for w in windows:
                    bf=rank1_bf(residual[:,:,node],variance[:,:,node],f,w)
                    nodes[tag+'_'+orth+str(w)][:,:,node]=torch.logaddexp(energies[w][:,:,node],bf)-math.log(2)
                    candidate_applications+=len(values)*residual.shape[1]*f.shape[-1]*len(TAU2)
    out={name:dict(score=torch.logsumexp(e,-1)-math.log(values.shape[-1]),root=e.softmax(-1)) for name,e in nodes.items()}
    return out,dict(partners=audit,candidate_scale_time_applications=candidate_applications,lag_bank=list(range(9)),TAU2=TAU2,BETA=BETA,normal_weights_updated=False,oracle_parameters_input=False)
