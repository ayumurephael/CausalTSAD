"""Time-local versus original trailing16 innovation readout, same predictor."""
import math
import torch
import torch.nn.functional as F

def readouts(model,output):
    e=.5*((2/3)*output['residual'].square()/output['variance']-math.log(3))
    temp=model.log_temperature.exp().clamp(.1,10)
    old=F.avg_pool1d(F.pad(e.transpose(1,2),(15,0),mode='replicate'),16,stride=1).transpose(1,2)
    out={}
    for name,v in [('point',e),('trailing16',old)]:
        out[name]=dict(score=temp*(torch.logsumexp(v/temp,-1)-math.log(v.shape[-1]))+model.bias,root=v.softmax(-1))
    error=float((out['trailing16']['score']-output['score']).abs().max())
    assert error<=1e-8
    return out,error
