"""C08b: distinguish root-confidence readout coupling from multitask gradient conflict."""
import math
import torch
from .model import JointRootReadout

class RepairedJointRootReadout(JointRootReadout):
 def __init__(self,*args,readout='lse',**kwargs):super().__init__(*args,**kwargs);self.readout=readout
 def forward(self,features,base_logit,graph,valid):
  out=super().forward(features,base_logit,graph,valid);c=features.shape[-2]
  # Exact identity: logmeanexp(r)=mean(r)+KL(U||softmax(r)).
  kl=-out['root_log_prob'].mean(-1)-math.log(c);out['root_confidence_KL']=kl
  if self.readout=='mean' and c>1:
   out['logit']=torch.where(valid,out['logit']-kl,out['logit'])
   out['joint_root_log_prob']=torch.nn.functional.logsigmoid(out['logit'])[...,None]+out['root_log_prob']
   out['null_log_prob']=torch.nn.functional.logsigmoid(-out['logit'])
  return out

def merged_gradients(detector_loss,root_loss,parameters,method):
 """Two-task PCGrad on shared coordinates only; private parameters retain own gradients."""
 params=list(parameters);gd=torch.autograd.grad(detector_loss,params,retain_graph=True,allow_unused=True)
 gr=torch.autograd.grad(root_loss,params,allow_unused=True);return project_gradients(gd,gr,params,method)

def project_gradients(gd,gr,params,method='pcgrad'):
 shared=[i for i,(d,r) in enumerate(zip(gd,gr)) if d is not None and r is not None]
 d=torch.cat([gd[i].reshape(-1) for i in shared]) if shared else None
 r=torch.cat([gr[i].reshape(-1) for i in shared]) if shared else None
 dot=float(torch.dot(d,r)) if shared else 0.;conflict=False
 if method=='pcgrad' and shared and dot<0 and float(d.square().sum())>0 and float(r.square().sum())>0:
  dp=d-torch.dot(d,r)/r.square().sum()*r;rp=r-torch.dot(d,r)/d.square().sum()*d;flat=dp+rp;conflict=True
 else:flat=d+r if shared else None
 out=[];offset=0
 for i,p in enumerate(params):
  if i in shared:out.append(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
  elif gd[i] is not None:out.append(gd[i])
  elif gr[i] is not None:out.append(gr[i])
  else:out.append(None)
 return out,dict(shared_gradient_dot=dot,projected_conflict=conflict)
