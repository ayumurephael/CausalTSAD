"""Whitelisted independent sources; benchmark arrays are absent from these APIs."""
import json
from pathlib import Path
import numpy as np

def raw_group(plan,row,pack_cache):
 path=Path(plan['old_source'] if row['kind']=='old' else plan['new_source'])/row['pack']
 if str(path) not in pack_cache:
  with np.load(path) as f:pack_cache[str(path)]={k:np.array(f[k]) for k in f.files}
 f=pack_cache[str(path)];i=row['offset'];views=row['views']
 if row['kind']=='old':
  cases=[f['x'][i],f['normal'][i]];labels=np.stack([f['labels'][i],np.zeros_like(f['labels'][i])]);roots=np.array([f['root'][i],-1]);window=np.stack([f['root_window'][i],np.zeros_like(f['root_window'][i])]).astype(bool);graph=f['adjacency'][i]
 else:
  cases=[f['cases'][i,k] for k in views];labels=f['labels'][i,views];roots=np.array([f['root'][i] if k>=2 else -1 for k in views]);window=np.zeros_like(labels,dtype=bool);window[np.asarray(views)>=2,320:]=True;graph=f['graph'][i]
 return cases,labels[:,320:],roots,window[:,320:],graph

def load_arrays(cache,row):
 with np.load(Path(cache)/row['path']) as f:return {k:np.array(f[k]) for k in f.files}
