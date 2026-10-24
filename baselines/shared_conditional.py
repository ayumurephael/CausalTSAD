"""Shared, strictly historical parent-conditional density and detector baseline.

No TimeRCD weights; no target fitting. Dense pair messages keep capacity and
computed slots fixed across graph controls. Normal mechanisms train only on
source normal views. Detection gradients are isolated from the density model.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SharedConditional(nn.Module):
    def __init__(self, history=64, width=64, message_width=32):
        super().__init__()
        self.history = history
        self.own = nn.Sequential(nn.Linear(history+1, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.message = nn.Sequential(nn.Linear(2*history, message_width), nn.GELU(),
                                     nn.Linear(message_width, message_width), nn.GELU())
        self.predictor = nn.Sequential(nn.Linear(width+message_width+1, width), nn.GELU(), nn.Linear(width, 2))
        self.detector = nn.Sequential(nn.Linear(width+4, width), nn.GELU(), nn.Linear(width, 1))

    def histories(self, x):
        b, t, c = x.shape; h = self.history
        windows = F.pad(x.transpose(1,2), (h,0)).unfold(-1,h+1,1).permute(0,2,1,3)
        past = windows[...,:h]
        mask = (torch.arange(t,device=x.device)[:,None] + torch.arange(h,device=x.device)[None] - h >= 0)
        mask = mask[None,:,None,:]
        count = mask.sum(-1).clamp_min(1)
        center = (past*mask).sum(-1)/count
        variance = ((past-center[...,None]).square()*mask).sum(-1)/count
        scale = variance.sqrt().clamp_min(1e-3)
        scale = torch.where(mask.any(-1),scale,torch.ones_like(scale))
        normalized = (past-center[...,None])/scale[...,None]*mask
        fraction = mask.float().mean(-1).expand(b,t,c)
        return normalized, (x-center)/scale, fraction

    def forward(self, x, adjacency):
        assert x.ndim == 3 and adjacency.shape == (len(x),x.shape[-1],x.shape[-1])
        h, y, fraction = self.histories(x)
        b,t,c,_ = h.shape
        own = self.own(torch.cat([h,fraction[...,None]],-1))
        # Axis i is target; axis j is source. Nothing is normalized across nodes.
        hi = h[:,:,:,None,:].expand(-1,-1,-1,c,-1)
        hj = h[:,:,None,:,:].expand(-1,-1,c,-1,-1)
        pair = self.message(torch.cat([hi,hj],-1))
        incoming = adjacency.transpose(1,2)[:,None,:,:,None]
        degree = incoming.sum(-2)
        aggregate = (pair*incoming).sum(-2)/degree.clamp_min(1)
        degree_feature = torch.log1p(degree).expand(b,t,c,1)
        stats = self.predictor(torch.cat([own,aggregate,degree_feature],-1))
        mu, logvar = stats.unbind(-1); logvar = logvar.clamp(-6,6)
        error = (y-mu)*torch.exp(-.5*logvar)
        nll = .5*(math.log(2*math.pi)+logvar+error.square())
        detached = torch.stack([torch.sign(error)*torch.log1p(error.abs()),
            torch.log1p(error.abs()),torch.log1p(nll.clamp_min(0)),logvar],-1).detach()
        node_logits = self.detector(torch.cat([own.detach(),detached],-1)).squeeze(-1)
        logit = torch.logsumexp(node_logits,-1)-math.log(c)
        return dict(mu=mu,logvar=logvar,nll=nll,node_logits=node_logits,logit=logit,
                    innovation=error.abs(),density_score=torch.logsumexp(nll,-1)-math.log(c))
