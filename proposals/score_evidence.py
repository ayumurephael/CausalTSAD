"""Eight-parameter controlled evidence composition; not a new density theorem."""
import torch
from torch import nn
class EvidenceHead(nn.Module):
 def __init__(self,mean,std,base_scale):
  super().__init__();self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32).clamp_min(1e-3));self.register_buffer('base_scale',torch.as_tensor(base_scale,dtype=torch.float32));self.linear=nn.Linear(7,1);nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)
 def forward(self,base,features,arm):
  z=(features-self.mean)/self.std
  if arm=='basecal':z=z.clone();z[...,1:5]=0
  delta=4*self.base_scale*torch.tanh(self.linear(z).squeeze(-1));return base+delta,delta
 @torch.no_grad()
 def project(self,positive):
  self.linear.weight[:,0].clamp_(min=0)
  if positive:self.linear.weight[:,1].clamp_(min=0)
 def full_scores(self,base,features,arm,channels,first_query=320):
  if channels==1:return base,torch.zeros_like(base)
  out=base.clone();delta=torch.zeros_like(base)
  if len(base)>first_query:out[first_query:],delta[first_query:]=self(base[first_query:],features,arm)
  return out,delta
