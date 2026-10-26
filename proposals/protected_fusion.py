"""Frozen causal features, small bounded residual head, identity for C1/warmup."""
import hashlib
import torch
from torch import nn
from proposals.raw_context import prepare,signed_log

def parameter_sha(model):
 h=hashlib.sha256()
 for k,v in sorted(model.named_parameters()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

@torch.inference_mode()
def causal_features(model,values,graph,mode,chunk=256):
 inputs,y,times,scale=prepare(values);features=[];roots=[]
 for start in range(0,len(times),chunk):
  end=min(len(times),start+chunk);sub=dict(inputs)
  for key in ['own','parent']:sub[key]=inputs[key][start:end]
  out=model.forward_prepared(sub,y[start:end],times[start:end],graph,mode)
  # Source-normalized later. No root confidence/entropy enters detection features.
  features.append(torch.stack([out['logit'],signed_log(out['normal_NLL'].mean(-1)),torch.log1p(out['normal_MSE'].mean(-1)),torch.log1p(out['normal_MSE'].max(-1).values)],-1))
  roots.append(out['root_logits'])
 return torch.cat(features),torch.cat(roots)

class ProtectedFusion(nn.Module):
 def __init__(self,mean,std,base_scale):
  super().__init__();self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32).clamp_min(1e-3));self.register_buffer('base_scale',torch.as_tensor(base_scale,dtype=torch.float32))
  self.residual=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,1))
  self.gate=nn.Sequential(nn.Linear(7,16),nn.GELU(),nn.Linear(16,1))
  nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)

 def forward(self,base,features,arm):
  z=(features-self.mean)/self.std
  if arm=='basecal':z=z.clone();z[...,1:5]=0
  delta=4*self.base_scale*self.residual(z).squeeze(-1).tanh()*self.gate(z).squeeze(-1).sigmoid()
  return base+delta,delta

 def full_scores(self,base,features,arm,channels,first_query=320):
  if channels==1:return base,torch.zeros_like(base)
  out=base.clone();delta=torch.zeros_like(base)
  if len(base)>first_query:out[first_query:],delta[first_query:]=self(base[first_query:],features,arm)
  return out,delta
