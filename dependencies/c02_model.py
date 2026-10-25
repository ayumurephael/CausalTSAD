"""C02 timing probe: same own history, normalization, capacity and dense FLOPs."""
import torch
from torch import nn
from torch.nn import functional as F


def inputs(values,history=64,block=16):
    b,t,c=values.shape
    past=F.pad(values,(0,0,history,0)).unfold(1,history,1)[:,:t]
    current=F.pad(values,(0,0,history-1,0)).unfold(1,history,1)
    times=torch.arange(t,device=values.device);positions=torch.arange(history,device=values.device)
    valid=positions[None,:]>=history-times.clamp(max=history)[:,None]
    valid_current=positions[None,:]>=history-(times+1).clamp(max=history)[:,None]
    starts=(times//block)*block
    block_past=past[:,starts]
    block_valid=valid[starts][None,:,None,:]
    count=block_valid.sum(-1,keepdim=True).clamp_min(1)
    center=(block_past*block_valid).sum(-1,keepdim=True)/count
    scale=(((block_past-center).square()*block_valid).sum(-1,keepdim=True)/count).sqrt().clamp_min(1e-3)
    scale=torch.where(block_valid.any(-1,keepdim=True),scale,torch.ones_like(scale))
    own=(past-center)/scale*valid[None,:,None,:]
    # All three views are constructed in all arms to match preprocessing work.
    parents=dict(block=(block_past-center)/scale*block_valid,
                 past=own,current=(current-center)/scale*valid_current[None,:,None,:])
    query=(values-center.squeeze(-1))/scale.squeeze(-1)
    fraction=(times.clamp(max=history)/history)[None,:,None].expand(b,t,c)
    return own,parents,fraction,query,times%block


class ParentTiming(nn.Module):
    def __init__(self,history=64,block=16,width=64,message_width=32):
        super().__init__();self.history=history;self.block=block
        self.own=nn.Sequential(nn.Linear(history+1,width),nn.GELU())
        self.message=nn.Sequential(nn.Linear(2*history,message_width),nn.GELU(),nn.Linear(message_width,message_width),nn.GELU())
        self.distribution=nn.Sequential(nn.Linear(width+message_width+1,width),nn.GELU(),nn.Linear(width,2*block))

    def forward(self,own_values,parent_values,fraction,graph,phase):
        b,t,c,h=own_values.shape
        own=self.own(torch.cat([own_values,fraction[...,None]],-1))
        target=own_values[:,:,:,None,:].expand(-1,-1,-1,c,-1)
        parent=parent_values[:,:,None,:,:].expand(-1,-1,c,-1,-1)
        message=self.message(torch.cat([target,parent],-1))
        mask=graph.transpose(1,2)[:,None,:,:,None];degree=mask.sum(-2)
        pooled=(message*mask).sum(-2)/degree.clamp_min(1)
        parameters=self.distribution(torch.cat([own,pooled,torch.log1p(degree).expand(b,t,c,1)],-1)).reshape(b,t,c,self.block,2)
        index=phase[None,:,None,None,None].expand(b,t,c,1,2)
        selected=parameters.gather(-2,index).squeeze(-2)
        mu,lv=selected.unbind(-1)
        return mu,lv.clamp(-6,6)


def predict(model,values,graph,timing):
    own,parent,fraction,query,phase=inputs(values,model.history,model.block)
    mu,lv=model(own,parent[timing],fraction,graph,phase)
    return mu,lv,query


def loss_and_score(model,b,timing=None):
    mu,lv,query=predict(model,b['values'],b['graph'],timing or b['timing'])
    squared=(query-mu).square()*torch.exp(-lv)
    nll=.5*(lv+squared+1.8378770664093453)
    innovation=torch.log1p(squared)
    return nll,innovation.amax(-1),innovation
