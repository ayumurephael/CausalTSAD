"""Raw-series causal innovation signatures with a source-learned finite fault prior.

Context coefficients are forward-inferred latent states, not saved simulator
parameters or target-trained network weights. Local Gaussian/diagonal temporal
approximations are explicit; this is not an exact FKF or a complete foundation model.
"""
import math
import numpy as np
import torch
from torch import nn
L=8;W=16;FIRST=320;SCALES=(.5,2.,8.)
def _features(x,center,scale,node,parents):
 t=torch.arange(L,len(x),device=x.device);z=(x-center)/scale;columns=[torch.ones(len(t),dtype=x.dtype,device=x.device),t.to(x.dtype)/256];descriptors=[None,None]
 for lag in range(1,L+1):columns.append(z[t-lag,node]);descriptors.append((node,lag,'own'))
 for parent in parents:
  for lag in range(L+1):
   raw=x[t-lag,parent];columns.extend([raw,raw.tanh()]);descriptors.extend([(parent,lag,'raw'),(parent,lag,'tanh')])
 return torch.stack(columns,-1),descriptors
@torch.no_grad()
def condition(prefix,graph):
 assert prefix.shape[0]==256 and prefix.dtype==torch.float64 and torch.isfinite(prefix).all();c=prefix.shape[1];center=prefix.median(0).values;scale=(prefix-center).abs().median(0).values*1.4826;scale=torch.where(scale<1e-3,torch.ones_like(scale),scale);states=[]
 for node in range(c):
  parents=torch.where(graph[:,node]>0)[0].tolist();f,desc=_features(prefix,center,scale,node,parents);fm=f.mean(0);fs=f.std(0,unbiased=False).clamp_min(1e-3);fm[0]=0;fs[0]=1;f=(f-fm)/fs;y=((prefix-center)/scale)[L:,node];w=torch.ones_like(y)
  for _ in range(3):
   a=torch.eye(f.shape[1],dtype=f.dtype,device=f.device)+f.T@(w[:,None]*f);chol=torch.linalg.cholesky(a);mean=torch.cholesky_solve((f.T@(w*y))[:,None],chol).squeeze(-1);res=y-f@mean;s2=((w*res.square()).sum()+mean.square().sum()+1)/(w.sum()+2);w=5/(4+res.square()/s2.clamp_min(1e-4))
  states.append(dict(parents=parents,center=fm,scale=fs,mean=mean,inverse=torch.cholesky_inverse(chol),noise=s2.clamp_min(1e-4),descriptors=desc))
 return dict(center=center,scale=scale,states=states,graph=graph)
@torch.no_grad()
def innovations(values,state):
 assert values.dtype==torch.float64;c=values.shape[1];t=len(values);mus=[];variances=[];a=torch.zeros((t,c,c,L+1),dtype=values.dtype,device=values.device)
 for node,s in enumerate(state['states']):
  raw,desc=_features(values,state['center'],state['scale'],node,s['parents']);f=(raw-s['center'])/s['scale'];mu=f@s['mean'];v=s['noise']*(1+(f@s['inverse']*f).sum(-1));mus.append(mu);variances.append(v);a[L:,node,node,0]=1
  for j,item in enumerate(desc):
   if item is None:continue
   parent,lag,kind=item;derivative=torch.ones_like(mu) if kind=='own' else state['scale'][parent]*torch.ones_like(mu) if kind=='raw' else state['scale'][parent]*(1-values[L-lag:t-lag,parent].tanh().square());a[L:,node,parent,lag]-=s['mean'][j]/s['scale'][j]*derivative
 mu=torch.stack(mus,-1);variance=torch.stack(variances,-1);residual=(values[L:]-state['center'])/state['scale']-mu
 return residual,variance,a[L:],mu
def covariance_log_ratio(residual,variance,matrix,amplitude,nu=4.):
 # R20 covariance parametrization: cov=K, Mahalanobis denominator nu-2.
 # This is a finite joint t likelihood ratio, not a complete Student-t process.
 white=residual/variance.sqrt();u=matrix/variance.sqrt()[...,None];q=u.transpose(-1,-2)@u;v=u.transpose(-1,-2)@white[...,None];eye=torch.eye(matrix.shape[-1],dtype=matrix.dtype,device=matrix.device);chol=torch.linalg.cholesky(eye+amplitude*q);solution=torch.cholesky_solve(v,chol);gain=amplitude*(v*solution).sum((-2,-1));remaining=white-amplitude*(u@solution).squeeze(-1);quad=remaining.square().sum(-1)+amplitude*solution.square().sum((-2,-1));logdet=2*chol.diagonal(dim1=-2,dim2=-1).log().sum(-1)
 return -.5*logdet+.5*(nu+residual.shape[-1])*torch.log1p(gain/(nu-2+quad))
@torch.no_grad()
def evidence(values,graph,signature='full',state=None):
 state=condition(values[:256],graph) if state is None else state;res,var,a,mu=innovations(values,state);c=values.shape[1];times=torch.arange(FIRST,len(values),device=values.device);factors=torch.empty((len(times),c,6),dtype=values.dtype,device=values.device);row_time=torch.arange(W,device=values.device);lag_ids=torch.arange(L+1,device=values.device);context_factors=0
 for root in range(c):
  for start in range(0,len(times),64):
   tt=times[start:start+64];idx=tt[:,None]-W+1+row_time[None]-L;rr=res[idx].reshape(len(tt),W*c);vv=var[idx].reshape(len(tt),W*c);mat=torch.zeros((len(tt),W,c,W+L),dtype=values.dtype,device=values.device);physical=torch.zeros_like(mat)
   for j in range(W):
    physical[:,j,root,j+L]=1
    if signature=='full':
     for lag in range(L+1):mat[:,j,:,j+L-lag]=a[idx[:,j],:,root,lag]
    elif signature=='self':
     for lag in range(L+1):mat[:,j,root,j+L-lag]=a[idx[:,j],root,root,lag]
    else:mat[:,j,root,j+L]=1
   mats=[physical.reshape(len(tt),W*c,W+L),mat.reshape(len(tt),W*c,W+L)]
   for mode,m in enumerate(mats):
    for k,beta in enumerate(SCALES):factors[start:start+len(tt),root,mode*3+k]=covariance_log_ratio(rr,vv,m,beta)
   context_factors+=len(tt)*6
 assert torch.isfinite(factors).all()
 return dict(log_factors=factors,normal_NLL=.5*(var.log()+res.square()/var+math.log(2*math.pi))[FIRST-L:].mean(),factorizations=context_factors,normal_mean=mu,context_state=state,ready_residual=res[FIRST-L:],ready_variance=var[FIRST-L:])
class SyndromePrior(nn.Module):
 def __init__(self,seed=348):
  super().__init__();g=torch.Generator().manual_seed(seed);self.weights=nn.Parameter(torch.randn(6,generator=g,dtype=torch.float64)*.01);self.bias=nn.Parameter(torch.zeros((),dtype=torch.float64))
 def forward(self,factors):
  component=factors+self.weights.log_softmax(0);root_evidence=component.logsumexp(-1);logit=root_evidence.logsumexp(-1)-math.log(factors.shape[-2])+self.bias;return dict(logit=logit,root=root_evidence.softmax(-1),component=component)
