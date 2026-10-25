"""Frozen teacher value-graph controls, with true degree information matched.

The wrong graph is a deliberate predictive control, not a proposed causal DAG.
It preserves each target's parent count, excludes own-current, may contain cycles.
"""
import hashlib
import numpy as np
import torch
from dependencies.c02_model import ParentTiming

def value_graph(graph, mode, key):
 g=np.asarray(graph,dtype=np.float32);c=len(g)
 assert g.shape==(c,c) and np.all(np.diag(g)==0)
 if mode=='true':return g.copy()
 if mode=='degree_only':return np.zeros_like(g)
 assert mode=='wrong_degree'
 seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little')^0xC0347
 rng=np.random.RandomState(seed);wrong=np.zeros_like(g)
 for child in range(c):
  k=int(g[:,child].sum());candidates=np.flatnonzero(np.arange(c)!=child)
  if not k:continue
  selected=rng.choice(candidates,k,replace=False)
  if k<c-1 and np.array_equal(np.sort(selected),np.flatnonzero(g[:,child])):
   absent=np.setdiff1d(candidates,selected);selected[0]=rng.choice(absent)
  wrong[selected,child]=1
 assert np.array_equal(wrong.sum(0),g.sum(0)) and np.all(np.diag(wrong)==0)
 return wrong

class ValueGraphTeacher(ParentTiming):
 def forward(self,own_values,parent_values,fraction,graph,phase):
  b,t,c,h=own_values.shape
  own=self.own(torch.cat([own_values,fraction[...,None]],-1))
  target=own_values[:,:,:,None,:].expand(-1,-1,-1,c,-1)
  parent=parent_values[:,:,None,:,:].expand(-1,-1,c,-1,-1)
  message=self.message(torch.cat([target,parent],-1))
  mask=graph.transpose(1,2)[:,None,:,:,None]
  # true topology degree in ALL controls, including no-parent-values.
  degree=self.reference_graph.transpose(1,2)[:,None,:,:,None].sum(-2)
  pooled=(message*mask).sum(-2)/degree.clamp_min(1)
  parameters=self.distribution(torch.cat([own,pooled,torch.log1p(degree).expand(b,t,c,1)],-1)).reshape(b,t,c,self.block,2)
  selected=parameters.gather(-2,phase[None,:,None,None,None].expand(b,t,c,1,2)).squeeze(-2)
  mu,lv=selected.unbind(-1)
  return mu,lv.clamp(-6,6)

def views(model,values,graph,mode,key,chunk=128):
 from tools.cache_c09 import legal_views
 model.reference_graph=graph
 control=torch.from_numpy(value_graph(graph[0].detach().cpu().numpy(),mode,key))[None].to(values.device)
 context,residual,mu,lv=legal_views(model,values,control,chunk)
 context[...,10]=torch.log1p(graph.sum(1))[:,None,:]
 return context,residual,mu,lv
