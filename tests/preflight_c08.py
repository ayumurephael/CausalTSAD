"""Source-free G0 checks for a posterior (current innovations allowed), not a forecaster."""
import argparse,sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from causal_tsad.io import atomic_json
from proposals.c08.model import JointRootReadout,ARMS,linear_flops
from proposals.c08.data import digest
def run(output):
 torch.set_num_threads(1);torch.manual_seed(348);torch.backends.cuda.matmul.allow_tf32=False
 model=JointRootReadout().cuda();x=torch.randn(2,19,8,12,device='cuda');base=torch.randn(2,19,device='cuda')
 graph=torch.zeros(2,8,8,device='cuda');graph[:,0,1]=1;graph[:,1,2]=1;graph[:,5,6]=1
 valid=torch.ones(2,19,dtype=torch.bool,device='cuda');valid[:,-3:]=False
 z=model(x,base,graph,valid);zero_error=float((z['logit']-torch.where(valid,base,torch.zeros_like(base))).abs().max());assert zero_error<1e-6
 # Make learned routes nonzero, otherwise invariances would be vacuous.
 with torch.no_grad():
  torch.nn.init.normal_(model.node.weight,std=.15);torch.nn.init.normal_(model.calibration[-1].weight,std=.15)
 z=model(x,base,graph,valid)
 mass=z['null_log_prob'].exp()+z['joint_root_log_prob'].exp().sum(-1);mass_error=float((mass-1).abs().max());assert mass_error<1e-6
 p=torch.randperm(8,device='cuda');per=model(x[:,:,p],base,graph[:,p][:,:,p],valid)
 equiv=max(float((per['logit']-z['logit']).abs().max()),float((per['root_log_prob']-z['root_log_prob'][:,:,p]).abs().max()));assert equiv<1e-5
 alt=x.clone();alt[:,-3:]=1e9;err=float((model(alt,base,graph,valid)['logit']-z['logit']).abs().max());assert err==0
 alt=x.clone();alt[:,12:]+=3;future_error=float((model(alt,base,graph,valid)['root_log_prob'][:,:12]-z['root_log_prob'][:,:12]).abs().max());assert future_error==0
 # A disconnected component cannot reach node0 under two graph layers.
 alt=x.clone();alt[:,:,5:7]+=3;disjoint=float((model(alt,base,graph,valid)['root_log_prob'][...,0]-z['root_log_prob'][...,0]).abs().max())
 # Root softmax globally couples probabilities; test local unnormalized evidence instead.
 evidence_error=float((model(alt,base,graph,valid)['node_logits'][...,0]-z['node_logits'][...,0]).abs().max());assert evidence_error==0
 single=model(x[:,:,:1],base,graph[:,:1,:1],valid);identity=float((single['logit']-torch.where(valid,base,torch.zeros_like(base))).abs().max());assert identity==0
 root=torch.zeros(2,19,dtype=torch.long,device='cuda');loss=-z['root_log_prob'].gather(-1,root[...,None])[valid].mean()+torch.nn.functional.binary_cross_entropy_with_logits(z['logit'][valid],torch.zeros_like(base[valid]))
 loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 assert float(model.node.weight.grad.norm())>0
 report=dict(status='passed',source_or_target_arrays_opened=False,optimizer_updates=0,parameters=sum(p.numel() for p in model.parameters()),
  zero_adapter_original_logit_error=zero_error,posterior_mass_error=mass_error,channel_equivariance_error=equiv,
  invalid_value_invariance_error=err,posterior_future_feature_invariance_error=future_error,
  disconnected_node_evidence_invariance_error=evidence_error,C1_identity_error=identity,root_and_detection_finite_backward=True,
  architecture_Linear_forward_FLOPs=linear_flops(model,19,8)*2,
  scope='posterior current evidence allowed; original offline base held fixed in equivariance/future checks, no full-model causal/equivariance claim',
  technical_random_feature_scalars=x.numel(),technical_base_scalars=base.numel(),root_softmax_disconnected_probability_change=disjoint,
  predictive_teacher_is_frozen_own_history_C01=True)
 atomic_json(output,report);print(json.dumps(report))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
