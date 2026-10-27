"""Original frozen TimeRCD only: no DAG, root head, optimizer or score fusion."""
from pathlib import Path
import sys,hashlib
import numpy as np
import torch
from causal_tsad.io import sha256

WEIGHTS={
 'official_uni':'250bafa559eb45385c275f110b27c044b4975465837821f56167d8286d7048a0',
 'official_multi':'e661d60abb2200d8977b4a4c959c12ede6a11361b099d70829cf194731f335ea',
 'selftrained_uni':'c4acfc8b4002575a9dfdc8ca0dede3d93fef4285c484eb2d2f3389908500b841',
 'selftrained_multi':'9a22a4a2e42d1fa147a4f4d0d09aa0ad6be1ef15c90f7a124fa846603cbf638e'}

def parameter_sha(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def normalize(values,policy):
 assert policy in ['native','causal_compat']
 x=np.asarray(values) if policy=='native' else np.asarray(values,dtype=np.float32)
 mean=x.mean(0);std=x.std(0);std=np.where(std==0,1e-8,std)
 return (x-mean)/std

class FrozenDetector:
 def __init__(self,code,weights):
  sys.path.insert(0,str(code));self.weights=Path(weights);self.models={};self.parameter_hashes={}
  self.actual_input_scalars=0;self.padded_forward_scalars=0;self.encoder_calls=0

 def load(self,name):
  if name not in self.models:
   from training_runtime import RuntimeModel,make_config
   p=self.weights/(name+'.pth');assert sha256(p)==WEIGHTS[name]
   model=RuntimeModel(make_config(1 if name.endswith('_uni') else 20),activation_checkpointing=False).cuda().eval().requires_grad_(False)
   saved=torch.load(p,map_location='cpu',weights_only=False);model.load_state_dict(saved['model_state_dict'],strict=True)
   self.models[name]=model;self.parameter_hashes[name]=parameter_sha(model)
  return self.models[name]

 @torch.inference_mode()
 def predict(self,values,name,policy):
  z=normalize(values,policy);assert z.ndim==2 and np.isfinite(z).all();t,c=z.shape;win=min(t,5000);raw=[];prob=[]
  model=self.load(name);model.ts_encoder.num_features=c
  for start in range(0,t,win):
   part=z[start:start+win];length=len(part)
   if length<win:part=np.concatenate([part,np.repeat(part[-1:],win-length,axis=0)])
   valid=torch.arange(win,device='cuda')[None]<length;batch=torch.as_tensor(part,dtype=torch.float32,device='cuda')[None]
   logits=model.anomaly_head(model.ts_encoder(batch,valid)).mean(-2).float()
   raw.append((logits[...,1]-logits[...,0])[0,:length].cpu().numpy());prob.append(logits.softmax(-1)[0,:length,1].cpu().numpy())
   self.padded_forward_scalars+=batch.numel();self.encoder_calls+=1
  self.actual_input_scalars+=int(values.size)
  return dict(logit=np.concatenate(raw),probability=np.concatenate(prob))

 def verify_frozen(self):
  assert all(parameter_sha(m)==self.parameter_hashes[k] for k,m in self.models.items())

def methods(channels):
 for group in ['official','selftrained']:
  yield group,group+('_uni' if channels==1 else '_multi')
  if channels==1:yield group+'_common_multi',group+'_multi'
