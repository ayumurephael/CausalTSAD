"""C03v2: learned past input/residual pairs, not supplied SCM coefficients.

Uses the already audited C09 legal predictive features. Context summaries end
16 samples before each query; current/future residuals cannot set their own tail.
Mean/median are CNP-style deterministic summaries, not a Bayesian posterior.
"""
import torch
from torch import nn
from proposals.c09.model import ConditionalTail as BaseTail, TailFusion

ARMS = ['no_context', 'learned_mean', 'learned_median']
CONTEXT_WIDTH = 32
GUARD = 16
BLOCK = 64
READY = 256


class ConditionalTail(nn.Module):
 def __init__(self, feature_dim=12, width=64):
  super().__init__()
  self.encoder = nn.Sequential(nn.Linear(feature_dim+1,32),nn.GELU(),nn.Linear(32,CONTEXT_WIDTH))
  self.density = BaseTail(feature_dim+CONTEXT_WIDTH,width)

 @property
 def taus(self): return self.density.taus

 def summary(self, context, residual, arm):
  assert arm in ARMS
  b,t,c,_=context.shape
  history=torch.sign(residual)*torch.log1p(residual.abs())
  embedded=self.encoder(torch.cat([context,history.clamp(-20,20)[...,None]],-1))
  prefix=torch.cat([torch.zeros_like(embedded[:,:1]),embedded.cumsum(1)],1)
  times=torch.arange(t,device=context.device)
  blocks=[]
  for j in range(3):
   end=(times-GUARD-j*BLOCK).clamp(0,t)
   start=(end-BLOCK).clamp_min(0)
   count=(end-start).clamp_min(1)[None,:,None,None]
   blocks.append((prefix[:,end]-prefix[:,start])/count)
  stack=torch.stack(blocks,-2)
  out=stack.median(-2).values if arm=='learned_median' else stack.mean(-2)
  out=out*(times>=READY)[None,:,None,None]
  # All slots and encoder computations retained, but no active context in control.
  if arm=='no_context': out=out*0
  return out

 def augment(self,context,residual,arm,context_source=None):
  features,history=(context,residual) if context_source is None else context_source
  return torch.cat([context,self.summary(features,history,arm)],-1)

 def forward(self,context,arm,residual,context_source=None):
  return self.density(self.augment(context,residual,arm,context_source),'conditional_quantile')

 def objective(self,context,residual,arm,ready):
  return self.density.objective(self.augment(context,residual,arm),residual,'conditional_quantile',ready)

 def evidence(self,context,residual,arm):
  return self.density.evidence(self.augment(context,residual,arm),residual,'conditional_quantile')


def linear_flops(t,c,width=64):
 # Dense Linear forward only. Prefix/median/nonlinear/backward costs separate.
 density=2*t*c*(13*32+32*32+44*width+width*width+width*9)
 return density,2*t*(5*16+16)
