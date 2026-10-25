"""Source-only C03 probe: isolated earlier context, not query-window fitting."""
import numpy as np


def correlation(target,parent):
    """Columns are channels; result[target,parent]. No cross-channel centering."""
    a=target-target.mean(0);b=parent-parent.mean(0)
    denom=np.sqrt((a*a).sum(0)[:,None]*(b*b).sum(0)[None,:])
    return np.divide(a.T@b,denom,out=np.zeros((a.shape[1],b.shape[1])),where=denom>1e-12).clip(-1,1)


def context(values,start,history=64,chunk=64,parts=3):
    c=values.shape[1];end=max(0,start-history);begin=max(0,end-parts*chunk)
    stats=[]
    for lo in range(begin,end,chunk):
        block=values[lo:min(end,lo+chunk)].astype(np.float64)
        if len(block)<16:continue
        features=[correlation(block[lag:],block[:-lag]) if lag else correlation(block,block) for lag in [0,1,2,4,8]]
        auto=np.diag(features[1]);features.extend([np.broadcast_to(auto[:,None],(c,c)),np.broadcast_to(auto[None,:],(c,c)),
            np.full((c,c),(end-begin)/(parts*chunk))])
        stats.append(np.stack(features,-1))
    if not stats:return np.zeros((c,c,8),np.float32),np.zeros((c,c,8),np.float32),end-begin
    return np.mean(stats,axis=0).astype(np.float32),np.median(stats,axis=0).astype(np.float32),end-begin


def encode(values,history=64,horizon=16):
    t,c=values.shape;q=(t+horizon-1)//horizon
    raw=np.zeros((q,c,history),np.float32);fraction=np.zeros((q,c),np.float32)
    query=np.zeros((q,horizon,c),np.float32);valid=np.zeros((q,horizon),bool)
    mean=np.zeros((q,c,c,8),np.float32);median=np.zeros_like(mean);ready=np.zeros((q,horizon),bool)
    history_slots=0;context_slots=0
    for k,start in enumerate(range(0,t,horizon)):
        past=values[max(0,start-history):start];length=len(past)
        center=past.mean(0) if length else np.zeros(c)
        scale=np.std(past,axis=0).clip(1e-3) if length else np.ones(c)
        if length:raw[k,:,-length:]=((past-center)/scale).T
        fraction[k]=length/history
        stop=min(t,start+horizon);query[k,:stop-start]=(values[start:stop]-center)/scale;valid[k,:stop-start]=True
        mean[k],median[k],used=context(values,start,history)
        ready[k,:stop-start]=used>=128
        history_slots+=length*c;context_slots+=used*c
    assert all(np.isfinite(a).all() for a in [raw,query,mean,median])
    return dict(raw=raw,fraction=fraction,query=query,valid=valid,mean=mean,median=median,ready=ready),dict(
        unique_scalars=values.size,valid_recent_history_scalar_uses=history_slots,valid_earlier_context_scalar_uses=context_slots)


def batch_order(pool,budget,seed):
    out=[]
    for c,pack in sorted(pool.items()):
        order=np.random.RandomState(seed+c).permutation(len(pack['normal_raw']))
        size=max(1,budget//(1024*c));out.extend((c,order[k:k+size]) for k in range(0,len(order),size))
    np.random.RandomState(seed).shuffle(out);return out


def tensors(pack,indices,arm,view,device='cuda',swap=False):
    import torch
    context_indices=(indices+1)%len(pack[view+'_raw']) if swap else indices
    ctx=pack[view+'_'+('median' if arm=='median' else 'mean')][context_indices]
    if arm=='zero':ctx=np.zeros_like(ctx)
    return dict(raw=torch.from_numpy(pack[view+'_raw'][indices]).to(device),
        fraction=torch.from_numpy(pack[view+'_fraction'][indices]).to(device),
        context=torch.from_numpy(ctx).to(device),graph=torch.from_numpy(pack['graph'][indices]).to(device),
        query=torch.from_numpy(pack[view+'_query'][indices]).to(device),valid=torch.from_numpy(pack[view+'_valid'][indices]).to(device),
        ready=torch.from_numpy(pack[view+'_ready'][indices]).to(device),labels=torch.from_numpy(pack['labels'][indices]).to(device))
