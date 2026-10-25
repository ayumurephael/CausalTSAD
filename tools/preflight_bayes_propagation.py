"""Posterior algebra/future-blindness/identity tests before source evaluation."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from proposals.bayes_propagation import BayesianPropagation,T,distances
from causal_tsad.io import atomic_json
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);g=np.array([[0,1,0],[0,0,1],[0,0,0]],np.float32);x=np.random.default_rng(348).normal(size=(T,3));p=dict(mean=0,normal_sigma=1,fault_sigmas=[1.4,2.],onset_prior=(np.ones(T//2+1)/(T//2+1)).tolist(),fault_prior=.5);m=BayesianPropagation(p);out=m.infer(x,g,forecast=True);z=x.copy();z[120:]+=10;changed=m.infer(z,g);assert np.max(np.abs(out['score'][:120]-changed['score'][:120]))<1e-12
 perm=np.array([2,0,1]);other=m.infer(x[:,perm],g[perm][:,perm]);assert np.max(np.abs(out['score']-other['score']))<1e-12 and np.max(np.abs(out['root']-other['root'][:,np.argsort(perm)]))<1e-12
 null=BayesianPropagation(dict(p,fault_sigmas=[1.,1.]));q=null.infer(x,g);expected=.5*np.array([np.mean(np.arange(T//4,3*T//4+1)<=i) for i in range(T)]);assert np.max(np.abs(q['score']-expected))<1e-12
 # Independent scalar enumeration at one time, including future-onset hypotheses.
 from scipy.special import logsumexp
 weights=[];active=[];time=100;d=distances(g)
 for root in range(3):
  for mode in [0,1]:
   for delay in [1,2,4,8]:
    for sigma in [1.4,2.]:
     for onset in range(T//4,3*T//4+1):
      w=np.log(.5/3/2/4/2/(T//2+1))
      for j in range(3):
       arrival=onset+(d[root,j]*delay if d[root,j]<=3 else 10**6) if mode else onset if j==root else 10**6
       if arrival<=time:w+=float((-np.log(sigma)+.5*x[arrival:time+1,j]**2*(1-1/sigma**2)).sum())
      weights.append(w);active.append(onset<=time)
 den=logsumexp([np.log(.5)]+weights);reference=float(np.exp(np.asarray(weights)[active]-den).sum());assert abs(reference-out['score'][time])<1e-12
 assert np.isfinite(out['score']).all() and ((out['score']>=0)&(out['score']<=1+1e-12)).all();assert np.allclose(out['root'][T//4:].sum(1),1)
 atomic_json(a.output,dict(status='passed',future_observation_invariance=True,channel_graph_permutation_equivariance=True,uninformative_likelihood_equals_prior=True,scalar_enumeration_error=abs(reference-out['score'][time]),normalized_active_root=True,optimizer_updates=0,benchmark_arrays_opened=False,hypothesis_time_cells=m.hypothesis_time_cells+null.hypothesis_time_cells))
