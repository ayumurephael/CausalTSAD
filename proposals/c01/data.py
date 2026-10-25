"""Fixed past/query blocks; only own legal history supplies normalization."""
from pathlib import Path
import hashlib
import json
import numpy as np
from causal_tsad.io import sha256,graph_control


def blocks(values,history=64,horizon=16):
    t,c=values.shape;q=(t+horizon-1)//horizon
    past=np.zeros((q,c,history),np.float32);valid=np.zeros_like(past,bool)
    query=np.zeros((q,horizon,c),np.float32);query_valid=np.zeros((q,horizon),bool)
    for k,start in enumerate(range(0,t,horizon)):
        begin=max(0,start-history);length=start-begin
        if length:past[k,:,-length:]=values[begin:start].T;valid[k,:,-length:]=True
        end=min(t,start+horizon);query[k,:end-start]=values[start:end];query_valid[k,:end-start]=True
    count=valid.sum(-1,keepdims=True).clip(1)
    center=(past*valid).sum(-1,keepdims=True)/count
    scale=np.sqrt(((past-center)**2*valid).sum(-1,keepdims=True)/count).clip(1e-3)
    scale=np.where(valid.any(-1,keepdims=True),scale,1.)
    raw=(past-center)/scale*valid
    normalized=(query-center.transpose(0,2,1))/scale.transpose(0,2,1)
    assert np.isfinite(normalized).all()
    return past,valid,raw.astype(np.float32),normalized.astype(np.float32),query_valid


def selected_rows(benchmark):
    rows=json.loads((benchmark/'data/system_index.json').read_text());selected=[]
    for track in ['id','mechanism_holdout']:
        for c in [1,8,12,16,20]:
            group=[r for r in rows if r['split']=='development' and r['track']==track and r['channels']==c]
            selected+=sorted(group,key=lambda r:hashlib.sha256(r['system_id'].encode()).hexdigest())[:10]
    assert len(selected)==100
    assert not {r['shard'] for r in selected}&{r['shard'] for r in rows if r['split']!='development'}
    return selected


def load_manifest(root):
    m=json.loads((root/'manifest.json').read_text());assert m['status']=='complete'
    for r in m['files']:assert sha256(root/r['path'])==r['SHA256']
    return m


def controlled_graph(graph,mode,key):
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little')^0x46113
    return graph_control(graph,mode,np.random.RandomState(seed).permutation(len(graph)))


def batches(rows,budget,seed):
    out=[]
    for c in [1,8,12,16,20]:
        group=[r for r in rows if r['channels']==c]
        rng=np.random.RandomState(seed+c);rng.shuffle(group)
        batch=[];scalars=0
        for r in group:
            if batch and scalars+r['scalars']>budget:out.append(batch);batch=[];scalars=0
            batch.append(r);scalars+=r['scalars']
        if batch:out.append(batch)
    np.random.RandomState(seed).shuffle(out)
    return out


def tensor_batch(root,rows,arm,device):
    import torch
    feature,mode=arm.split('_',1);loaded=[]
    for r in rows:
        with np.load(root/r['path']) as f:
            z={k:np.array(f[k]) for k in f.files}
        if feature=='raw':
            d=z['x_features'].shape[-1]
            for v in ['x','normal']:
                raw=z[v+'_raw'];z[v+'_features']=np.pad(raw,[(0,0)]*(raw.ndim-1)+[(0,d-raw.shape[-1])])
        z['graph']=controlled_graph(z['graph'],mode,r['key']);loaded.append(z)
    keys=['x_features','normal_features','fraction','x_fraction','normal_fraction',
          'x_query','normal_query','x_base','normal_base','query_valid','labels','graph']
    return {k:torch.from_numpy(np.stack([z[k] for z in loaded])).to(device) for k in keys if k in loaded[0]}
