"""C09: source-trained conditional residual tails, independent from C08 root supervision."""
import math
import torch
from torch import nn
from torch.nn import functional as F
TAUS=[.001,.01,.05,.1,.5,.9,.95,.99,.999]
Z=[-3.090232306,-2.326347874,-1.644853627,-1.281551566,0.,1.281551566,1.644853627,2.326347874,3.090232306]
ARMS=['gaussian','marginal_quantile','conditional_quantile','base_cal']

class ConditionalTail(nn.Module):
 def __init__(self,feature_dim=12,width=64):
  super().__init__();self.net=nn.Sequential(nn.Linear(feature_dim,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,9))
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
  self.register_buffer('taus',torch.tensor(TAUS));self.register_buffer('z',torch.tensor(Z))
  gaps=torch.tensor([Z[i+1]-Z[i] for i in range(8)])
  self.register_buffer('gap_bias',torch.log(torch.expm1(gaps)))

 def forward(self,context,arm):
  if arm=='marginal_quantile':context=torch.zeros_like(context)
  raw=self.net(context)
  if arm=='gaussian':
   location=raw[...,4];logscale=raw[...,0].clamp(-3,3);quantiles=location[...,None]+logscale.exp()[...,None]*self.z
  else:
   location=raw[...,4];lower=F.softplus(raw[...,:4]+self.gap_bias[:4])+1e-5
   upper=F.softplus(raw[...,5:]+self.gap_bias[4:])+1e-5
   left=location[...,None]-torch.flip(torch.cumsum(torch.flip(lower,[-1]),-1),[-1]);right=location[...,None]+torch.cumsum(upper,-1)
   quantiles=torch.cat([left,location[...,None],right],-1);logscale=None
  return dict(quantiles=quantiles,location=location,logscale=logscale)

 def objective(self,context,residual,arm,ready):
  out=self(context,arm);r=residual
  if arm=='gaussian':loss=out['logscale']+.5*((r-out['location'])*torch.exp(-out['logscale'])).square()+.9189385332046727
  else:
   error=r[...,None]-out['quantiles'];loss=torch.maximum(self.taus*error,(self.taus-1)*error).mean(-1)
  return loss[ready[:,:,None].expand_as(r)].mean(),out

 def evidence(self,context,residual,arm):
  out=self(context,arm);r=residual
  if arm=='gaussian':
   z=(r-out['location'])*torch.exp(-out['logscale']);logp=torch.special.log_ndtr(z);logs=torch.special.log_ndtr(-z)
  else:
   q=out['quantiles'];idx=(r[...,None]>=q).sum(-1).clamp(1,8);lo=q.gather(-1,(idx-1)[...,None]).squeeze(-1);hi=q.gather(-1,idx[...,None]).squeeze(-1)
   p=self.taus[idx-1]+(self.taus[idx]-self.taus[idx-1])*(r-lo)/(hi-lo).clamp_min(1e-6);p=p.clamp(1e-12,1-1e-7)
   logp=torch.log(p);logs=torch.log1p(-p)
   low_scale=(q[...,1]-q[...,0])/math.log(TAUS[1]/TAUS[0]);high_scale=(q[...,-1]-q[...,-2])/math.log((1-TAUS[-2])/(1-TAUS[-1]))
   low_log=math.log(TAUS[0])+(r-q[...,0])/low_scale.clamp_min(1e-6)
   high_log=math.log(1-TAUS[-1])-(r-q[...,-1])/high_scale.clamp_min(1e-6)
   below=r<q[...,0];above=r>q[...,-1]
   logp=torch.where(below,low_log,torch.where(above,torch.log1p(-torch.exp(high_log.clamp_max(-1e-7))),logp))
   logs=torch.where(above,high_log,torch.where(below,torch.log1p(-torch.exp(low_log.clamp_max(-1e-7))),logs))
  surprise=(-torch.minimum(logp,logs)-math.log(2)).clamp_min(0)
  return torch.log1p(surprise),out

class TailFusion(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(5,16),nn.GELU(),nn.Linear(16,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,node,base,ready,arm):
  c=node.shape[-1]
  if arm=='base_cal':node=torch.zeros_like(node)
  features=torch.stack([node.mean(-1),node.amax(-1),node.std(-1,unbiased=False),base,torch.full_like(base,math.log(c))],-1)
  residual=self.net(features).squeeze(-1)
  return base+residual*ready if c>1 else base

def linear_flops(t,c,width=64):return 2*t*c*(12*width+width*width+width*9),2*t*(5*16+16)
