"""Analytic directional/pairing checks, no source or target arrays, no optimizer."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from proposals.score_evidence import EvidenceHead
from causal_tsad.io import atomic_json
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(348);cfg=json.loads((Path(__file__).resolve().parents[1]/'experiments/score_evidence_v3.json').read_text());assert all(k in cfg for k in ['data_seed','pair_loss_weight','learning_rate']);m=EvidenceHead([0]*7,[1]*7,1).cuda();x=torch.randn(4,704,7,device='cuda');base=x[...,0];assert torch.equal(m(base,x,'true')[0],base)
 full=torch.randn(1024,device='cuda');score,_=m.full_scores(full,x[0],'true',20);assert torch.equal(full,score)
 with torch.no_grad():m.linear.weight[0,1]=-1;m.linear.bias.fill_(.3)
 m.project(True);assert m.linear.weight[0,1]>=0
 with torch.no_grad():m.linear.weight[0,1]=.2
 x.requires_grad_(True);s=m(base,x,'true')[0];derivative=torch.autograd.grad(s.sum(),x,retain_graph=True)[0][...,1];assert (derivative>=0).all()
 changed=x.detach().clone();changed[...,1:5]+=100;assert torch.equal(m(base,x,'basecal')[0],m(base,changed,'basecal')[0])
 rank=lambda s:F.softplus(-(s[2:,None,:]-s[None,:2,:])).mean();assert torch.allclose(rank(s),rank(s+10),atol=1e-6)
 loss=F.binary_cross_entropy_with_logits(s,torch.cat([torch.zeros_like(base[:2]),torch.ones_like(base[:2])]))+rank(s);loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters());c1,_=m.full_scores(full,x[0],'true',1);assert torch.equal(c1,full)
 atomic_json(a.output,dict(status='passed',parameters=8,zero_identity=True,C1_identity=True,monotone_teacher_derivative=True,basecal_causal_invariance=True,pair_loss_intercept_invariance=True,finite_backward=True,source_or_target_arrays_opened=False,optimizer_updates=0,synthetic_forward_time_points=4*704*7,synthetic_backward_time_points=4*704))
