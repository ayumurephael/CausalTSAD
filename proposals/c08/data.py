"""Cached observation-derived innovations; source-only labels separate from inputs."""
import hashlib,json
from pathlib import Path
import numpy as np
from causal_tsad.io import sha256,graph_control
from proposals.c08.model import ARMS

def digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def load_manifest(root,expected=None):
 if expected:assert sha256(root/'manifest.json')==expected
 m=json.loads((root/'manifest.json').read_text());assert m['status']=='complete'
 return m

def order(rows,budget,seed):
 out=[]
 for c in [1,8,12,16,20]:
  group=[r for r in rows if r['channels']==c];np.random.RandomState(seed+c).shuffle(group)
  chunk=[];n=0
  for r in group:
   if chunk and n+r['scalars']>budget:out.append(chunk);chunk=[];n=0
   chunk.append(r);n+=r['scalars']
  if chunk:out.append(chunk)
 np.random.RandomState(seed).shuffle(out);return out

def batch(root,rows,arm,device='cuda'):
 import torch
 mode,_,values=ARMS[arm];loaded=[]
 for r in rows:
  with np.load(root/r['path']) as f:z={k:np.array(f[k]) for k in f.files}
  seed=int.from_bytes(hashlib.sha256(r['key'].encode()).digest()[:4],'little')^0x46113
  perm=np.random.RandomState(seed).permutation(r['channels']);z['graph']=graph_control(z['graph'],mode,perm)
  if values=='zero':
   for view in ['x','normal']:z[view+'_features']=np.zeros_like(z[view+'_features'])
  if 'root' in z:z['true_root']=np.array(z['root'],copy=True)
  if arm=='dag_random_root' and 'root' in z and r['channels']>1:
   # Random source target is fixed per system, labels/windows unchanged. Derangement root.
   z['root']=np.asarray((int(z['root'])+1+seed%(r['channels']-1))%r['channels'])
  loaded.append(z)
 keys=['x_features','normal_features','x_base','normal_base','valid','graph','labels','root','true_root','root_window']
 return {k:torch.from_numpy(np.stack([z[k] for z in loaded])).to(device) for k in keys if k in loaded[0]}

def infer(model,b,view):return model(b[view+'_features'].float(),b[view+'_base'].float(),b['graph'].float(),b['valid'].bool())
