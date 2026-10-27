"""Frozen readout controls. Labels and normal twins are absent from this interface."""
import math
import torch
from proposals.context_mixture import mixture_logpdf,node_evidence

METHODS=['component0','component1','component2','component3','component4',
 'density_uniform','density_learned','mean_uniform','mean_learned','hard_context','density_fullvariance']

@torch.inference_mode()
def readouts(z,selector):
 y,mu,v,valid=z['response'],z['mu'],z['variance'],z['valid']
 assert valid.all(),'This fixed audit requires all 5 candidates to exist'
 lw=selector(z['features'],valid);w=lw.exp();ul=torch.full_like(lw,-math.log(5))
 e=(y[...,None]-mu).square()/(3*v)-.5*math.log(3)
 out={f'component{k}':e[...,k] for k in range(5)}
 out.update(density_uniform=node_evidence(y,mu,v,ul),density_learned=node_evidence(y,mu,v,lw),
  mean_uniform=e.mean(-1),mean_learned=(w*e).sum(-1),
  hard_context=e.gather(-1,lw.argmax(-1,keepdim=True))[...,0],
  density_fullvariance=node_evidence(y,mu,v[...,0:1].expand_as(v),lw))
 lp=-.5*(math.log(2*math.pi)+v.log()+(y[...,None]-mu).square()/v)
 log_r=torch.log_softmax(lw+lp,-1)
 err=float((torch.logsumexp(log_r+e,-1)-out['density_learned']).abs().max())
 assert err<1e-8
 assert all(torch.isfinite(q).all() for q in out.values())
 return out,dict(identity_error=err,normal_responsibility=log_r.exp(),weights=w,component_log_normal=lp)

@torch.inference_mode()
def normal_target_nll(z,selector):
 target=z['training_response'];lw=selector(z['features'],z['valid']);ul=torch.full_like(lw,-math.log(5))
 per=.5*(math.log(2*math.pi)+z['variance'].log()+(target[...,None]-z['mu']).square()/z['variance'])
 uniform=-mixture_logpdf(target,z['mu'],z['variance'],ul);learned=-mixture_logpdf(target,z['mu'],z['variance'],lw)
 return dict(full=float(per[...,0].mean()),uniform=float(uniform.mean()),
  learned=float(learned.mean()),components=per.mean((0,1)).cpu().tolist(),
  per_node=dict(full=per[...,0].mean(0).cpu().tolist(),uniform=uniform.mean(0).cpu().tolist(),learned=learned.mean(0).cpu().tolist()),
  privileged_pointwise_best_component=float(per.min(-1).values.mean()))
