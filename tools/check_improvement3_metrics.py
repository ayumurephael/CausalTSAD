"""Regression for the r1 event serialization failure; no model or data access."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from proposals.improvement3 import metrics
def check():
    native=metrics.spans
    def legacy(mask):
        d=np.diff(np.r_[0,np.asarray(mask,dtype=np.int8),0])
        return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))
    def convert(x):
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    rng=np.random.default_rng(348);count=0;old_failed=False
    for case in range(100):
        y=np.zeros(100,dtype=np.int8)
        if case%3:y[10:20]=1;y[50:75]=1
        s=rng.normal(size=100);node=rng.normal(size=(100,8))
        for threshold in [-10.,0.,10.]:
            for nr in [node,None]:
                metrics.spans=legacy
                before=metrics.evaluate(s,y,threshold,nr,3)
                try:json.dumps(before,allow_nan=False)
                except TypeError:old_failed=True
                metrics.spans=native
                after=metrics.evaluate(s,y,threshold,nr,3)
                assert json.loads(json.dumps(before,default=convert,allow_nan=False))==json.loads(json.dumps(after,allow_nan=False))
                count+=1
    assert old_failed
    return dict(status='passed',regression_cases=count,legacy_failure_reproduced=True,metric_values_unchanged=True,optimizer_steps=0,model_forwards=0)
if __name__=='__main__':print(json.dumps(check()))
