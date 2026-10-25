"""Source-only white-list and fixed graph controls, no mechanism parameters."""
import numpy as np
from causal_tsad.io import graph_control


def batch_order(pool,budget,seed):
    out=[]
    for c,pack in sorted(pool.items()):
        indices=np.random.RandomState(seed+c).permutation(len(pack['normal']))
        size=max(1,budget//(1024*c));out.extend((c,indices[k:k+size]) for k in range(0,len(indices),size))
    np.random.RandomState(seed).shuffle(out);return out


def tensors(pack,indices,arm,view,device='cuda'):
    import torch
    mode,timing=('none','past') if arm=='none' else arm.split('_',1)
    graphs=[]
    for i in indices:
        permutation=np.random.RandomState(int(pack['seed'][i])^0x46113).permutation(pack['adjacency'].shape[-1])
        graphs.append(graph_control(pack['adjacency'][i],mode,permutation))
    return dict(values=torch.from_numpy(pack['normal' if view=='normal' else 'x'][indices]).to(device),
        graph=torch.from_numpy(np.stack(graphs)).to(device),timing=timing)
