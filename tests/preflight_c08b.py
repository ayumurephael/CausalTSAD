import argparse,sys,math,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from causal_tsad.io import atomic_json
from proposals.c08.model import JointRootReadout
from proposals.c08.repair import RepairedJointRootReadout,project_gradients,merged_gradients
from proposals.c08.data import digest
def run(output):
 torch.set_num_threads(1);torch.manual_seed(348);torch.backends.cuda.matmul.allow_tf32=False
 old=JointRootReadout().cuda();new=RepairedJointRootReadout(readout='lse').cuda();new.load_state_dict(old.state_dict(),strict=True)
 x=torch.randn(1,48,8,12,device='cuda');base=torch.randn(1,48,device='cuda');g=torch.zeros(1,8,8,device='cuda');g[:,0,1]=1;v=torch.ones(1,48,dtype=torch.bool,device='cuda')
 with torch.no_grad():torch.nn.init.normal_(old.node.weight,std=.1);torch.nn.init.normal_(old.calibration[-1].weight,std=.1)
 new.load_state_dict(old.state_dict());a=old(x,base,g,v);b=new(x,base,g,v);err=float((a['logit']-b['logit']).abs().max());assert err==0
 new.readout='mean';m=new(x,base,g,v);kl=-a['root_log_prob'].mean(-1)-math.log(8);identity=float((m['logit']-(a['logit']-kl)).abs().max());assert identity==0
 mass_error=float((m['null_log_prob'].exp()+m['joint_root_log_prob'].exp().sum(-1)-1).abs().max());assert mass_error<1e-6
 assert torch.equal(new(x[:,:,:1],base,g[:,:1,:1],v)['logit'],base)
 params=[torch.empty(2),torch.empty(1)];gd=[torch.tensor([1.,0.]),torch.tensor([3.])];gr=[torch.tensor([-1.,1.]),None]
 projected,info=project_gradients(gd,gr,params,'pcgrad');assert torch.allclose(projected[0],torch.tensor([.5,1.5])) and projected[1].item()==3 and info['projected_conflict']
 summed,_=project_gradients(gd,gr,params,'sum');assert torch.equal(summed[0],gd[0]+gr[0])
 positive,_=project_gradients(gd,[torch.tensor([2.,1.]),None],params,'pcgrad');assert torch.equal(positive[0],torch.tensor([3.,1.]))
 zero,_=project_gradients(gd,[torch.zeros(2),None],params,'pcgrad');assert all(torch.isfinite(z).all() for z in zero)
 perm=torch.randperm(8,device='cuda');before=new(x,base,g,v);after=new(x[:,:,perm],base,g[:,perm][:,:,perm],v);equiv=float((before['logit']-after['logit']).abs().max());assert equiv<1e-5
 out=new(x,base,g,v);det=torch.nn.functional.binary_cross_entropy_with_logits(out['logit'],torch.zeros_like(base));root=-out['root_log_prob'][...,0].mean();merged,stats=merged_gradients(det,root,new.parameters(),'pcgrad')
 assert all(z is None or torch.isfinite(z).all() for z in merged)
 atomic_json(output,dict(status='passed',parameters=sum(p.numel() for p in new.parameters()),old_forward_null_error=err,KL_identity_error=identity,
  shared_only_projection=True,private_gradient_unchanged=True,nonconflicting_gradients_unchanged=True,zero_gradient_finite=True,C1_identity=True,
  posterior_mass_error=mass_error,full_model_projected_backward_finite=True,channel_equivariance_error=equiv,random_feature_scalars=x.numel(),random_base_scalars=base.numel(),source_or_target_arrays_opened=False,optimizer_updates=0))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
