"""Source-only readout controls with an exact frozen Gaussian-energy anchor.

This module never receives source root/labels as features. Two-head comparisons
hold the frozen predictor and residual features fixed; they do not compare two
end-to-end trained foundation models.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

ARMS=('point_joint','fixed_mix_joint','multi_joint','multi_det','multi_shared2','multi_dual')

def network(seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        net=nn.Sequential(nn.Linear(20,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
        nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
    return net

class RawAnchorHeads(nn.Module):
    def __init__(self,arm,seed):
        super().__init__();assert arm in ARMS;self.arm=arm
        self.net_a=network(seed)
        self.net_b=network(seed+1000) if arm in ['multi_shared2','multi_dual'] else None
    def forward(self,x,energy,temperature,bias):
        shaped=x.reshape(*x.shape[:-1],5,4)
        if self.arm=='point_joint':shaped=shaped[...,:1,:].expand_as(shaped)
        elif self.arm=='fixed_mix_joint':shaped=shaped.mean(-2,keepdim=True).expand_as(shaped)
        x=shaped.reshape_as(x);a=self.net_a(x).squeeze(-1).double()
        b=self.net_b(x).squeeze(-1).double() if self.net_b is not None else a
        if self.arm=='multi_dual':detection=energy+a;root=energy+b
        else:detection=energy+(.5*(a+b) if self.net_b is not None else a);root=detection
        score=temperature*(torch.logsumexp(detection/temperature,-1)-math.log(energy.shape[-1]))+bias
        return dict(score=score,root=root.softmax(-1),root_logits=root,detection_logits=detection)

def objective(o,labels,roots,arm):
    pos=labels>0;neg=~pos;assert pos.any() and neg.any()
    det=.5*(F.softplus(-o['score'][pos]).mean()+F.softplus(o['score'][neg]).mean())
    target=roots[:,None].expand_as(labels)[pos]
    rca=F.cross_entropy(o['root_logits'][pos],target)
    return det if arm=='multi_det' else det+.25*rca,det,rca
