"""Source-head G0 only; does not read development/validation/locked benchmark."""
import sys,argparse,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from proposals.temporal_innovation_head import features,trailing_mean,TemporalInnovationHead,objective
from tools.train_meta_basis import psha
from causal_tsad.io import atomic_json
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(348)
    r=torch.randn(2,384,8,dtype=torch.float64,device='cuda');var=torch.ones_like(r);ff=features(r,var);refs=[]
    for mode in ['point','fixed_mix','multiscale']:
        m=TemporalInnovationHead(mode,348).cuda();assert sum(v.numel() for v in m.parameters())==1217;refs.append(psha(m));o=m(ff);base=ff[...,2]-math.log(3);expected=torch.logsumexp(base,-1)-math.log(8);assert torch.equal(expected,o['score'])
    assert len(set(refs))==1
    modified=r.clone();modified[:,201:]+=19;ff2=features(modified,var);assert torch.equal(ff[:,:201],ff2[:,:201])
    errors=[]
    for w in [1,4,16,64,256]:
        actual=trailing_mean(r,w);expected=torch.stack([r[:,max(0,t-w+1):t+1].mean(1) for t in range(r.shape[1])],1);errors.append(float((actual-expected).abs().max()))
    assert max(errors)<1e-12
    m=TemporalInnovationHead('multiscale',348).cuda();out=m(ff);y=torch.zeros(2,384,dtype=torch.int64,device='cuda');y[1,250:254]=1;roots=torch.tensor([0,3],device='cuda');loss,_,_=objective(out,y,roots);loss.backward();assert any(v.grad is not None and v.grad.abs().sum()>0 for v in m.parameters())
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters()) and ff.requires_grad is False
    perm=torch.tensor([6,1,3,4,7,0,5,2],device='cuda');op=m(ff[:,:,perm]);unperm=torch.argsort(perm);err=float((op['root'][:,:,unperm]-out['root']).abs().max());assert err<5e-6
    result=dict(status='passed',parameters=1217,matched_initialization=True,compressed_point_zero_adapter_exact=True,causal_feature_future_invariance_exact=True,trailing_mean_max_error=max(errors),FP32_permutation_root_error=err,FP32_permutation_tolerance=5e-6,gradients_only_head=True,technical_feature_raw_slots=r.numel(),technical_head_node_time_applications=5*r.numel(),benchmark_arrays_opened=False)
    atomic_json(a.output,result);print(json.dumps(result))
