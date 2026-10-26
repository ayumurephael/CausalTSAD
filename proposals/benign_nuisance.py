"""Shared deterministic channel nuisance, preserving exact four-view class matching."""
import hashlib
import numpy as np
def transform(row,strength,kind='linear',seed=None):
 x=row['cases'];c=x.shape[-1];scale=x[:2,:256].reshape(-1,c).std(0);rs=np.random.default_rng(int(row['key']) if seed is None else int.from_bytes(hashlib.sha256((row['key']+'/'+str(seed)).encode()).digest()[:8],'little'));sign=rs.choice([-1,1],c);t=np.arange(1024,dtype=np.float64)/1023;curve=t if kind=='linear' else t*t
 out=dict(row);out['cases']=x+strength*curve[None,:,None]*scale[None,None]*sign[None,None];return out
