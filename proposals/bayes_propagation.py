"""Exact hypothesis marginalization for a restricted single persistent-fault model.

This is an evidence-level feasibility model, not a raw-TS foundation model.
Unknown root, physical/sensor type, onset, lag and variance regime are marginalized.
"""
import hashlib
import numpy as np
import torch

T=192
LAGS=(1,2,4,8)
def seed_for(namespace,index):return int.from_bytes(hashlib.sha256((namespace+'/'+str(index)).encode()).digest()[:8],'little')|2**48
def distances(graph):
 c=len(graph);d=np.where(graph>0,1,c+1).astype(int);np.fill_diagonal(d,0)
 for k in range(c):d=np.minimum(d,d[:,k,None]+d[None,k,:])
 return d
def generate(index,split):
 seed=seed_for('C06-bayes-feasibility-v1/'+split,index);rng=np.random.default_rng(seed);c=[8,12,16,20][index%4];g=np.zeros((c,c),np.float32)
 for j in range(1,c):
  if rng.random()<.15:continue
  ps=rng.choice(j,size=int(rng.integers(1,min(3,j)+1)),replace=False);g[ps,j]=1
 perm=rng.permutation(c);g=g[perm][:,perm];root=int(rng.integers(c));onset=int(rng.integers(T//4,3*T//4+1));lag=int(rng.choice(LAGS));scale=[1.4,2.][(index//4)%2];d=distances(g)[root];arrival=np.where(d<=c,onset+d*lag,T+1);noise=rng.normal(size=(4,T,c));cases=[x.copy() for x in noise];labels=[np.zeros(T,np.int8) for _ in range(4)];modes=['normal']*4;oracle=[]
 for mode in ['measurement','physical']:
  active=np.zeros((T,c),bool)
  if mode=='measurement':active[onset:,root]=True
  else:active=np.arange(T)[:,None]>=arrival[None]
  for k in [0,1]:
   cases.append(noise[k]*np.where(active,scale,1));y=np.zeros(T,np.int8);y[onset:]=1;labels.append(y);modes.append(mode);oracle.append(active)
 assert np.array_equal(cases[4][:,root],cases[6][:,root]) and np.array_equal(cases[5][:,root],cases[7][:,root])
 return dict(seed=seed,graph=g,root=root,onset=onset,lag=lag,scale=scale,cases=np.asarray(cases),labels=np.asarray(labels),modes=modes,
  oracle_activation_only=oracle,labels_detail=dict(effect=None,anomaly_channel=None,anomaly_time=None),scope='Gaussian evidence-level event simulation, not retained250Mbenchmark')

def fit_source(seed):
 # Global likelihood parameters from source normal observations and known
 # injected root windows only. No affected-channel labels or oracle A are used.
 normal=[];fault=[[],[]];onsets=np.ones(T//2+1,np.float64);observed=0
 for i in range(64):
  r=generate(i,'fit'+str(seed));normal.append(r['cases'][:4].reshape(-1));f=r['cases'][4:6,r['onset']:,r['root']].reshape(-1);fault[0 if r['scale']==1.4 else 1].append(f);onsets[r['onset']-T//4]+=1;observed+=r['cases'][:4].size+f.size
 n=np.concatenate(normal);mu=float(n.mean());sn=float(np.sqrt(np.mean((n-mu)**2)));sf=[float(np.sqrt(np.mean((np.concatenate(v)-mu)**2))) for v in fault]
 assert min(sf)>sn
 return dict(mean=mu,normal_sigma=sn,fault_sigmas=sf,onset_prior=(onsets/onsets.sum()).tolist(),fault_prior=.5,mode_prior=[.5,.5],lag_prior=[.25]*4,source_fit_observed_scalars=observed,fit_seed=seed,source_graphs=64,oracle_activation_used=False,optimizer_updates=0)

class BayesianPropagation:
 def __init__(self,parameters,device='cuda'):self.p=parameters;self.device=device;self.hypothesis_time_cells=0
 @torch.inference_mode()
 def infer(self,values,graph,variant='dag_delay_mixture',oracle_lag=None,forecast=False):
  x=torch.as_tensor(values,dtype=torch.float64,device=self.device);t,c=x.shape;assert t==T and np.isfinite(graph).all();g=np.asarray(graph)
  if variant=='reverse':g=g.T
  elif variant=='wrong':
   perm=np.random.default_rng(348+c).permutation(c);g=g[perm][:,perm]
  d=distances(g)
  if variant=='independent':d=np.full((c,c),c+1,int);np.fill_diagonal(d,0)
  modes=[1] if variant=='physical_only' else [0,1];lags=[0] if variant=='instant' else [oracle_lag] if variant=='oracle_lag' else list(LAGS)
  assert None not in lags
  roots,types,delays,scales,onsets=np.meshgrid(np.arange(c),modes,lags,np.arange(2),np.arange(T//4,3*T//4+1),indexing='ij');rr,mm,dd,kk,oo=[a.ravel() for a in [roots,types,delays,scales,onsets]];h=len(rr)
  arrival=oo[:,None]+np.where(d[rr]<=c,d[rr]*dd[:,None],10**6)
  arrival[mm==0]=10**6;arrival[np.arange(h),rr]=oo
  prior=np.log(self.p['fault_prior']/c/len(modes)/len(lags)/2)+np.log(np.asarray(self.p['onset_prior'])[oo-T//4]);prior=torch.as_tensor(prior,dtype=torch.float64,device=self.device)
  sf=torch.as_tensor(self.p['fault_sigmas'],dtype=torch.float64,device=self.device);sn=self.p['normal_sigma'];e=(x-self.p['mean']).square();lr=-torch.log(sf/sn)[:,None,None]+.5*e[None]*(1/sn**2-1/sf[:,None,None]**2);cs=lr.cumsum(1);kind=torch.as_tensor(kk,device=self.device);time=torch.arange(t,device=self.device);ll=torch.zeros((h,t),dtype=torch.float64,device=self.device)
  for j in range(c):
   a=torch.as_tensor(arrival[:,j],device=self.device);before=cs[kind,(a-1).clamp(0,t-1),j];before=torch.where(a>0,before,torch.zeros_like(before));part=cs[kind,:,j]-before[:,None];ll+=torch.where(time[None]>=a[:,None],part,torch.zeros_like(part))
  logits=ll+prior[:,None];den=torch.logaddexp(torch.logsumexp(logits,0),torch.full((t,),np.log1p(-self.p['fault_prior']),dtype=torch.float64,device=self.device));q=(logits-den).exp();active=time[None]>=torch.as_tensor(oo,device=self.device)[:,None];activeq=q*active;score=activeq.sum(0);rootmass=torch.zeros((c,t),dtype=torch.float64,device=self.device);rootmass.index_add_(0,torch.as_tensor(rr,device=self.device),activeq);mode_mass=(activeq*torch.as_tensor(mm,device=self.device)[:,None]).sum(0)
  predicted=[]
  if forecast:
   for j in range(c):predicted.append((q*(time[None]+8>=torch.as_tensor(arrival[:,j],device=self.device)[:,None])).sum(0).cpu().numpy())
  self.hypothesis_time_cells+=h*t
  return dict(score=score.cpu().numpy(),root=(rootmass/score.clamp_min(1e-300)[None]).T.cpu().numpy(),physical_given_fault=(mode_mass/score.clamp_min(1e-300)).cpu().numpy(),forecast8=np.stack(predicted,1) if forecast else None,hypotheses=h,
   note='exact within finite persistent-single-fault likelihood model; no general exact-DBN or rawTSAD claim')
