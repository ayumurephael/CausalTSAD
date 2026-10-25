"""Independent source splits; targets never enter distribution/fusion call signatures."""
import hashlib,json
import numpy as np
from causal_tsad.io import sha256
def digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def manifest(cache):
 m=json.loads((cache/'manifest.json').read_text());assert m['status']=='complete'
 for r in m['files']:r['ready_scalars']=max(0,r['length']-256)*r['channels']
 return m
def order(rows,budget,seed):
 out=[]
 for c in [8,12,16,20]:
  group=[r for r in rows if r['channels']==c];np.random.RandomState(seed+c).shuffle(group);chunk=[];n=0
  for r in group:
   if chunk and n+r['scalars']>budget:out.append(chunk);chunk=[];n=0
   chunk.append(r);n+=r['scalars']
  if chunk:out.append(chunk)
 np.random.RandomState(seed).shuffle(out);return out
def batch(cache,rows,device='cuda'):
 import torch
 arrays=[]
 for r in rows:
  with np.load(cache/r['path']) as f:arrays.append({k:np.array(f[k]) for k in f.files})
 keys=['x_context','normal_context','x_residual','normal_residual','x_base','normal_base','ready','labels','root','root_window']
 out={k:torch.from_numpy(np.stack([z[k] for z in arrays])).to(device) for k in keys if k in arrays[0]}
 out['ready'] &= torch.arange(out['ready'].shape[1],device=device)[None]>=256
 return out
def infer(tail,fusion,b,view,arm):
 node,out=tail.evidence(b[view+'_context'].float(),b[view+'_residual'].float(),arm)
 return dict(logit=fusion(node,b[view+'_base'].float(),b['ready'].bool(),arm),node=node,quantiles=out['quantiles'])
