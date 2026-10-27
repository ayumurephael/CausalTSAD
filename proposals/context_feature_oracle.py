"""SOURCE-ONLY privileged diagnosis. Never a deployable predictor.

Keep the observed response and observed fold normalization fixed. Separate
context coefficient/noise estimation from own/parent query-feature pollution.
The twin may retain shared prefix contamination; it removes only the controlled
late measurement intervention. No oracle input is allowed in a target model.
"""
import torch
from proposals.context_crossfit import context_mask,blocks,design

VARIANTS=('OO','CO','OC_own','OC_parent','OC_both','CC','OC_sham')

@torch.inference_mode()
def diagnose(model,observed,twin,graph,config):
 assert observed.shape==twin.shape and observed.dtype==torch.float64
 assert not config['periodic'] and not config['long_parent'] and config['mode']=='crossfit'
 T,C=observed.shape;device=observed.device;t=torch.arange(32,T,device=device)
 # Equal correction magnitude, shifted to the wrong time; oracle negative control.
 sham=observed-torch.roll(observed-twin,T//2,dims=0)
 out={k:dict(residual=torch.zeros_like(observed),variance=torch.ones_like(observed)) for k in VARIANTS}
 factors=features=query_slots=0;calibration_checks=[]
 for node in range(C):
  pa=torch.where(graph[:,node]>0)[0].tolist()
  for lo,hi in blocks(T,config['K'],'crossfit'):
   if hi<=320:continue
   m=context_mask(T,lo,hi,config['gap'],list(range(1,33)),'crossfit')
   assert m.sum()>=16
   tr=torch.as_tensor(m,device=device);query=(t>=max(lo,320))&(t<hi)
   idx=t[tr];center=observed[idx].median(0).values
   scale=(observed[idx]-center).abs().median(0).values*1.4826
   scale=torch.where(scale<1e-3,torch.ones_like(scale),scale)
   z=(observed-center)/scale;zn=(twin-center)/scale;zs=(sham-center)/scale
   ro,ps=design(model,z,node,pa,t);rn,ps2=design(model,zn,node,pa,t);rs,ps3=design(model,zs,node,pa,t)
   assert ps==ps2==ps3
   # Same feature standardization for every arm, including oracle context fit.
   ctx=ro[tr];fm=ctx.mean(0);fs=ctx.std(0,unbiased=False).clamp_min(1e-3);fm[0]=0;fs[0]=1
   fo=(ro-fm)/fs;fn=(rn-fm)/fs;fsh=(rs-fm)/fs
   latent={}
   for fit,cf,cy in [('O',fo[tr],z[idx,node]),('C',fn[tr],zn[idx,node])]:
    w=torch.ones_like(cy);eye=torch.eye(cf.shape[-1],device=device,dtype=cf.dtype)
    for _ in range(3):
     gram=eye+cf.T@(w[:,None]*cf);chol=torch.linalg.cholesky(gram)
     beta=torch.cholesky_solve((cf.T@(w*cy))[:,None],chol)[:,0]
     e=cy-cf@beta;noise=((w*e.square()).sum()+beta.square().sum()+1)/(w.sum()+2)
     noise=noise.clamp_min(1e-4);nu=config['nu'];w=(nu+1)/(nu+e.square()/noise)
    latent[fit]=(beta,chol,noise);factors+=3
   own=torch.cat([fn[:,:ps],fo[:,ps:]],-1)
   parent=torch.cat([fo[:,:ps],fn[:,ps:]],-1)
   conditions=[('OO','O',fo),('CO','C',fo),('OC_own','O',own),('OC_parent','O',parent),('OC_both','O',fn),('CC','C',fn),('OC_sham','O',fsh)]
   for name,fit,full in conditions:
    q=full[query];beta,chol,noise=latent[fit]
    mu=q@beta;inv=torch.cholesky_solve(q.T,chol).T
    out[name]['residual'][t[query],node]=z[t[query],node]-mu
    out[name]['variance'][t[query],node]=noise*(1+(q*inv).sum(-1))
    query_slots+=len(q)
   features+=3*(T-32)
 for v in out.values():
  for k in v:v[k]=v[k][320:]
  assert torch.isfinite(v['residual']).all() and torch.isfinite(v['variance']).all() and (v['variance']>0).all()
 return out,dict(context_fit_input_scalars=2*observed.numel(),feature_node_rows=features,factorizations=factors,query_node_outputs=query_slots,
  optimizer_updates=0,training_exposure=0,oracle_only=True,normalization='fixed observed protected-fold median/MAD and observed-context feature moments')
