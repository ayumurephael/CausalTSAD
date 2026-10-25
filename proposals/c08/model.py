"""Joint no-fault/root posterior, not a query-blind forecasting model."""
import math
import torch
from torch import nn

ARMS={
 'local_det':('none',False,'values'), 'local_joint':('none',True,'values'),
 'dag_det':('parents',False,'values'), 'dag_joint':('parents',True,'values'),
 'rewired_joint':('rewired',True,'values'), 'base_cal':('none',False,'zero'),
 'prior_joint':('parents',True,'zero'), 'dag_random_root':('parents',True,'values')}

class JointRootReadout(nn.Module):
 def __init__(self,feature_dim=12,width=32,layers=2):
  super().__init__();self.feature_dim=feature_dim;self.width=width;self.layers=layers
  self.embed=nn.Sequential(nn.Linear(feature_dim+2,width),nn.GELU())
  self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,width),nn.GELU()) for _ in range(layers)])
  self.updates=nn.ModuleList([nn.Sequential(nn.Linear(3*width+2,width),nn.GELU()) for _ in range(layers)])
  self.node=nn.Linear(width,1)
  self.calibration=nn.Sequential(nn.Linear(2,width),nn.GELU(),nn.Linear(width,1))
  for layer in [self.node,self.calibration[-1]]:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

 def forward(self,features,base_logit,graph,valid):
  # No metadata/labels/root/normal-twin arguments exist at the model boundary.
  b,t,c,d=features.shape;assert d==self.feature_dim and graph.shape==(b,c,c)
  x=torch.where(valid[:,:,None,None],features,torch.zeros_like(features))
  base=torch.where(valid,base_logit,torch.zeros_like(base_logit)).detach()
  ctx=torch.stack([torch.sign(base)*torch.log1p(base.abs()),torch.full_like(base,math.log(c))],-1)
  h=self.embed(torch.cat([x,ctx[:,:,None,:].expand(-1,-1,c,-1)],-1))
  parents=graph.transpose(1,2)[:,None,:,:,None];children=graph[:,None,:,:,None]
  indegree=parents.sum(-2);outdegree=children.sum(-2)
  for message,update in zip(self.messages,self.updates):
   pair=message(torch.cat([h[:,:,:,None,:].expand(-1,-1,-1,c,-1),h[:,:,None,:,:].expand(-1,-1,c,-1,-1)],-1))
   pa=(pair*parents).sum(-2)/indegree.clamp_min(1)
   ch=(pair*children).sum(-2)/outdegree.clamp_min(1)
   degrees=torch.cat([torch.log1p(indegree),torch.log1p(outdegree)],-1).expand(b,t,c,2)
   h=h+update(torch.cat([h,pa,ch,degrees],-1))
  node=self.node(h).squeeze(-1);root_log_prob=torch.log_softmax(node,-1)
  correction=self.calibration(ctx).squeeze(-1)+torch.logsumexp(node,-1)-math.log(c)
  logit=base+correction
  if c==1:logit=base # Protect common-multi C1 detector exactly, no C1 optimization.
  logit=torch.where(valid,logit,base)
  # p(null)=sigmoid(-logit); p(root=i)=sigmoid(logit)*softmax(node)_i.
  return dict(logit=logit,node_logits=node,root_log_prob=root_log_prob,
              joint_root_log_prob=torch.nn.functional.logsigmoid(logit)[...,None]+root_log_prob,
              null_log_prob=torch.nn.functional.logsigmoid(-logit))

def linear_flops(model,t,c):
 w=model.width;d=model.feature_dim
 return 2*t*(c*((d+2)*w+w)+model.layers*(c*c*(2*w*w+w*w)+c*(3*w+2)*w)+2*w+w)
