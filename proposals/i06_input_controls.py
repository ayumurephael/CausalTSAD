"""Frozen TSAD input/readout controls. No root rankings from TimeRCD."""
import math
import numpy as np
import torch
from causal_tsad.frozen_detector import normalize

@torch.inference_mode()
def detector_scores(detector,values,original_channels):
    z=normalize(values,'native');T,C=z.shape
    assert T<=5000 and np.isfinite(z).all()
    model=detector.load('selftrained_multi');model.ts_encoder.num_features=C
    x=torch.as_tensor(z,dtype=torch.float32,device='cuda')[None]
    valid=torch.ones((1,T),device='cuda',dtype=torch.bool)
    logits=model.anomaly_head(model.ts_encoder(x,valid)).float()
    diff=logits[...,1]-logits[...,0]
    # Exact native arithmetic for mean: reduce two logits before subtracting.
    avg=logits.mean(-2)
    out={'mean':avg[...,1]-avg[...,0], 'max':diff.max(-1).values,
         'lse':torch.logsumexp(diff,dim=-1)-math.log(C)}
    if C==2*original_channels:
        own=logits[:,:,:original_channels].mean(-2)
        out['xmean']=own[...,1]-own[...,0]
    detector.actual_input_scalars+=values.size
    detector.padded_forward_scalars+=x.numel();detector.encoder_calls+=1
    return {k:v[0].cpu().numpy() for k,v in out.items()}

def changed_residual(res,key,mode):
    if mode=='aligned':return res.copy()
    if mode=='time':return np.roll(res,len(res)//2,axis=0)
    # A cyclic nonzero displacement is a derangement, not a random permutation
    # with possible fixed points. One deterministic mapping per system.
    shift=1+int(key)%(res.shape[1]-1)
    return np.roll(res,shift,axis=1)
