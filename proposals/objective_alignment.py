"""Three source objectives on the identical frozen-context selector interface.

Roots and normal responses are loss targets; predict() cannot access them.
"""
import math
import torch
from proposals.context_mixture import mixture_logpdf,node_evidence,aggregate

def predict(selector,z,temperature,bias):
    lw=selector(z['features'],z['valid'])
    evidence=(z['response'][...,None]-z['mu']).square()/(3*z['variance'])-.5*math.log(3)
    mean=(lw.exp()*evidence).sum(-1)
    density=node_evidence(z['response'],z['mu'],z['variance'],lw)
    return dict(lw=lw,mean=mean,density=density,full=evidence[...,0],
                score_mean=aggregate(mean,temperature,bias),score_density=aggregate(density,temperature,bias),
                score_full=aggregate(evidence[...,0],temperature,bias))

def objective(selector,normal,fault,normal_target,fault_target,root,onset,kind,temperature,bias):
    n=predict(selector,normal,temperature,bias);f=predict(selector,fault,temperature,bias)
    nlln=-mixture_logpdf(normal_target,normal['mu'],normal['variance'],n['lw'])
    nllf=-mixture_logpdf(fault_target,fault['mu'],fault['variance'],f['lw'])
    assert nlln.ndim==2 and nlln.shape==nllf.shape and nllf.shape[1]>1
    mask=torch.arange(nllf.shape[0],device=nllf.device)+320>=onset
    base=.5*(nlln.mean()+nllf.mean())
    other=(nllf.sum(-1)-nllf[:,root])/(nllf.shape[1]-1)
    balanced=torch.where(mask,.5*nllf[:,root]+.5*other,nllf.mean(-1))
    root_loss=.5*(nlln.mean()+balanced.mean())
    rank=torch.nn.functional.softplus(n['score_mean'][mask]-f['score_mean'][mask]).mean()
    if kind=='all_nll':loss=base
    elif kind=='root_nll':loss=root_loss
    elif kind=='relation_rank':loss=base+rank
    else:raise ValueError(kind)
    return loss,dict(all_nll=float(base.detach()),root_balanced_nll=float(root_loss.detach()),rank_loss=float(rank.detach()),
                     root_normal_target_nll=float(nllf[mask,root].mean().detach()))
