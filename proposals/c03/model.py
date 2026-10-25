"""Equal-capacity parent-only normal mechanism head with explicit context."""
import torch
from torch import nn


class ContextMechanism(nn.Module):
    def __init__(self,history=64,horizon=16,width=64,message_width=32):
        super().__init__();self.horizon=horizon
        self.own=nn.Sequential(nn.Linear(history+1,width),nn.GELU())
        self.message=nn.Sequential(nn.Linear(2*history+8,message_width),nn.GELU(),nn.Linear(message_width,message_width),nn.GELU())
        self.distribution=nn.Sequential(nn.Linear(width+message_width+1,width),nn.GELU(),nn.Linear(width,2*horizon))

    def forward(self,raw,fraction,graph,context):
        b,q,c,h=raw.shape
        own=self.own(torch.cat([raw,fraction[...,None]],-1))
        target=raw[:,:,:,None,:].expand(-1,-1,-1,c,-1)
        parent=raw[:,:,None,:,:].expand(-1,-1,c,-1,-1)
        messages=self.message(torch.cat([target,parent,context],-1))
        mask=graph.transpose(1,2)[:,None,:,:,None];degree=mask.sum(-2)
        pooled=(messages*mask).sum(-2)/degree.clamp_min(1)
        parameters=self.distribution(torch.cat([own,pooled,torch.log1p(degree).expand(b,q,c,1)],-1))
        mu,lv=parameters.reshape(b,q,c,self.horizon,2).unbind(-1)
        return mu.transpose(-1,-2),lv.transpose(-1,-2).clamp(-6,6)


def loss_and_score(model,b):
    mu,lv=model(b['raw'],b['fraction'],b['graph'],b['context'])
    squared=(b['query']-mu).square()*torch.exp(-lv)
    nll=.5*(lv+squared+1.8378770664093453)
    innovation=torch.log1p(squared)
    return nll,innovation.amax(-1)
