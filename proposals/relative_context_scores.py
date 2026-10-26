"""E3 factorial scoring: channel vs case calibration, center vs trailing windows."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.special import logsumexp,softmax
WINDOWS=[1,8,32,128]
def average(x,w,centered):
    if w==1:return x.copy()
    if centered:return uniform_filter1d(x,w,axis=0,mode='nearest')
    pad=np.concatenate([np.repeat(x[:1],w-1,axis=0),x],axis=0)
    cs=np.concatenate([np.zeros_like(pad[:1]),np.cumsum(pad,axis=0)],axis=0)
    return (cs[w:]-cs[:-w])/w
def robust(x):
    m=np.median(x,axis=0);d=1.4826*np.median(np.abs(x-m),axis=0)
    return (x-m)/np.where(d<1e-6,1.,d)
def readouts(r,v,temperature=1.,bias=0.):
    q=r*r/v;C=r.shape[-1];out={};temp=float(temperature)
    e=.5*((2/3)*q-np.log(3))
    def result(node,score=None):
        if score is None:score=temp*(logsumexp(node/temp,-1)-np.log(C))+bias
        return dict(score=score,root=softmax(node,-1))
    out['point']=result(e)
    for centered in [False,True]:
        name='center' if centered else 'trailing';nodes=[];scores=[]
        for w in WINDOWS:
            node=.5*((2/3)*average(q,w,centered)-np.log(3))*np.sqrt(w)
            o=result(node);out[f'{name}{w}']=o;nodes.append(node);scores.append(o['score'])
        # Exact window.py ordering: LSE over channels, then per-case scale MAD.
        relative=np.stack([robust(s) for s in scores]);choice=relative.argmax(0)
        selected=np.stack(nodes)[choice,np.arange(len(r))]
        out[name+'_case']=result(selected,relative.max(0))
        # The prose additionally requests per-channel calibration. Keep separate.
        node=np.max(np.stack([robust(e) for e in nodes]),axis=0)
        out[name+'_channel']=result(node)
    return out
