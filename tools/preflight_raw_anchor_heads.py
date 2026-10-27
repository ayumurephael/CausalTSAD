"""Exact-anchor, task-private gradients and numerical finite preflight."""
import sys,json,argparse,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from proposals.raw_anchor_task_heads import ARMS,RawAnchorHeads,objective
from causal_tsad.io import atomic_json
from tools.train_meta_basis import psha
def run(output):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(348);x=torch.randn(2,64,8,20,device='cuda');e=torch.randn(2,64,8,device='cuda',dtype=torch.float64)*10
    expected=.7*(torch.logsumexp(e/.7,-1)-math.log(8))+.2;errors={};hashes={}
    for arm in ARMS:
        m=RawAnchorHeads(arm,348).cuda();o=m(x,e,.7,.2);errors[arm]=float((o['score']-expected).abs().max());assert errors[arm]==0
        assert torch.equal(o['root'],e.softmax(-1));hashes[arm]=psha(m.net_a)
        y=torch.zeros(2,64,device='cuda',dtype=torch.long);y[:,16:20]=1;root=torch.tensor([2,3],device='cuda')
        loss,_,_=objective(o,y,root,arm);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        n=sum(p.numel() for p in m.parameters());assert n==(2434 if arm in ['multi_shared2','multi_dual'] else 1217)
    assert len(set(hashes.values()))==1
    m=RawAnchorHeads('multi_dual',348).cuda();o=m(x,e,.7,.2)
    g=torch.autograd.grad(o['score'].sum(),tuple(m.net_b.parameters()),allow_unused=True,retain_graph=True);assert all(v is None for v in g)
    g=torch.autograd.grad(o['root_logits'].sum(),tuple(m.net_a.parameters()),allow_unused=True);assert all(v is None for v in g)
    atomic_json(output,dict(status='passed',zero_head_original_energy_exact=True,score_max_errors=errors,common_first_network_initialization=True,task_private_gradients=True,parameters_single=1217,parameters_double=2434,technical_head_node_time_applications=2*64*8*len(ARMS),new_backbone_forwards=0,target_arrays_opened=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
