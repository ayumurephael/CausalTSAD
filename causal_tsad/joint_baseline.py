"""B2: frozen raw X,G -> TimeRCD score + conditional root posterior.

Repackages measured C08b mean+PCGrad checkpoints. No new optimization,
no true event window/root/normal-twin in this inference boundary.
"""
import numpy as np
import torch
from pathlib import Path
from causal_tsad.io import sha256
from dependencies.c01_model import CausalResidualDetector
from dependencies.raw_views import FrozenViews
from proposals.c08.repair import RepairedJointRootReadout
from tools.cache_c08 import moving

BASE_SHA='9a22a4a2e42d1fa147a4f4d0d09aa0ad6be1ef15c90f7a124fa846603cbf638e'
TEACHER_SHA='47e98b418848ef8b80c8bb4216e1c729ccc74d60af1bf87c029096e57f62deac'

class JointTSADRCA:
 def __init__(self,original_code,base_checkpoint,teacher_checkpoint,head_checkpoint):
  assert torch.cuda.is_available(),'Run this measured baseline on a CUDA server'
  assert sha256(base_checkpoint)==BASE_SHA and sha256(teacher_checkpoint)==TEACHER_SHA
  torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  teacher=torch.load(teacher_checkpoint,map_location='cpu',weights_only=False);tc=teacher['config'];self.tc=tc
  self.teacher=CausalResidualDetector(teacher['feature_dim'],tc['width'],tc['message_width'],tc['horizon']).cuda().eval().requires_grad_(False)
  self.teacher.load_state_dict(teacher['model'],strict=True)
  self.views=FrozenViews(original_code,base_checkpoint,dict(tc,checkpoint_sha256=BASE_SHA))
  saved=torch.load(head_checkpoint,map_location='cpu',weights_only=False);cfg=saved['config']
  assert saved['variant']=='mean_pcgrad' and saved['readout']=='mean' and saved['arm']=='dag_joint'
  self.head=RepairedJointRootReadout(cfg['feature_dim'],cfg['width'],cfg['layers'],readout='mean').cuda().eval().requires_grad_(False)
  self.head.load_state_dict(saved['model'],strict=True)
  self.threshold=saved['source_calibration']['threshold'];self.seed=saved['seed']
  self.provenance=dict(base_checkpoint_SHA256=BASE_SHA,teacher_checkpoint_SHA256=TEACHER_SHA,head_checkpoint_SHA256=sha256(head_checkpoint),
   optimizer_seed=self.seed,head_training='C08b fixed3pass final, mean+PCGrad, no new training',threshold_scope='source-validation normal M 99th percentile')

 @torch.no_grad()
 def predict(self,values,graph,head_chunk=1024):
  x=np.ascontiguousarray(values,dtype=np.float32);g=np.ascontiguousarray(graph,dtype=np.float32)
  assert x.ndim==2 and len(x)>0 and np.isfinite(x).all();t,c=x.shape
  assert g.shape==(c,c) and ((g==0)|(g==1)).all() and (np.diag(g)==0).all()
  degree=g.sum(0).astype(int);queue=list(np.flatnonzero(degree==0));visited=0
  while queue:
   node=queue.pop();visited+=1
   for child in np.flatnonzero(g[node]):
    degree[child]-=1
    if degree[child]==0:queue.append(int(child))
  assert visited==c,'Input must be normal-system DAG, source-row -> target-column'
  if c==1:
   base,_=self.views.base_logits(x);features=np.zeros((t,1,12),np.float32)
  else:
   encoded,_,_=self.views.encode(x,self.tc);features_tensor=torch.from_numpy(encoded['features'])[None].cuda();fraction=torch.from_numpy(encoded['fraction'])[None].cuda()
   mu,lv=self.teacher.density(features_tensor,fraction,torch.zeros(1,c,c,device='cuda'))
   mu=mu[0].cpu().numpy().reshape(-1,c)[:t].astype(np.float64);lv=lv[0].cpu().numpy().reshape(-1,c)[:t].astype(np.float64)
   query=encoded['query'].reshape(-1,c)[:t].astype(np.float64);residual=(query-mu)*np.exp(-lv/2)
   signed=np.sign(residual)*np.log1p(np.abs(residual));energy=np.log1p(residual**2)
   q=np.sign(query)*np.log1p(np.abs(query));m=np.sign(mu)*np.log1p(np.abs(mu));fraction=np.repeat(encoded['fraction'],self.tc['horizon'],axis=0)[:t]
   features=np.clip(np.stack([signed,energy,lv,m,q,fraction]+[moving(energy,w) for w in [8,32,64]]+[moving(signed,w) for w in [8,32,64]],-1),-50,50).astype(np.float32)
   base=encoded['base'].ravel()[:t]
  graph_tensor=torch.from_numpy(g)[None].cuda();result={k:[] for k in ['logit','root_posterior','joint_root_probability','null_probability']}
  for start in range(0,t,head_chunk):
   end=min(t,start+head_chunk);out=self.head(torch.from_numpy(features[start:end])[None].cuda(),torch.from_numpy(base[start:end])[None].cuda(),graph_tensor,torch.ones(1,end-start,dtype=torch.bool,device='cuda'))
   for name,value in [('logit',out['logit']),('root_posterior',out['root_log_prob'].exp()),('joint_root_probability',out['joint_root_log_prob'].exp()),('null_probability',out['null_log_prob'].exp())]:result[name].append(value[0].cpu().numpy())
  result={k:np.concatenate(v) for k,v in result.items()};result['base_logit']=base
  result['anomaly_probability']=1-result['null_probability'];result['alarm']=result['logit']>self.threshold
  if c==1:assert np.array_equal(result['logit'],base)
  return result

 def alarm_events(self,prediction):
  alarm=prediction['alarm'];edges=np.diff(np.pad(alarm.astype(np.int8),(1,1)));events=[]
  for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
   posterior=prediction['root_posterior'][start:end].mean(0);ranking=np.argsort(-posterior,kind='stable')
   events.append(dict(start=int(start),end_exclusive=int(end),root_channel=int(ranking[0]),root_ranking=ranking.tolist(),mean_conditional_root_posterior=posterior.tolist()))
  return events
