"""Analytic timing, likelihood and frozen normal-mean interface checks."""
import sys,json,argparse,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from proposals.causal_meta_basis import CausalMetaBasis
from proposals.innovation_readout import readouts
from tools.infer_innovation_probe import extension
from causal_tsad.io import atomic_json,sha256
def run(a):
    torch.set_num_threads(1);ext=extension(str(a.conditional_code/'proposals/conditional_change_evidence.py'));torch.manual_seed(348)
    dtype=torch.float64;device='cuda';r=torch.randn(2,31,device=device,dtype=dtype);var=torch.rand_like(r)+.5;z=torch.randn(2,31,3,device=device,dtype=dtype)
    actual=ext.integrated_bf(r,var,z);rw=ext.rolling((r/var.sqrt())[...,None]).squeeze(-1);u=ext.rolling(z/var.sqrt()[...,None]);terms=[]
    for tau2 in ext.TAU2:
        A=torch.eye(16,device=device,dtype=dtype)+tau2*(u@u.transpose(-1,-2));terms.append(.5*((rw.square()).sum(-1)-(rw[...,None,:]@torch.linalg.solve(A,rw[...,None])).squeeze(-1).squeeze(-1)-torch.linalg.slogdet(A)[1]))
    expected=torch.logsumexp(torch.stack(terms),0)-math.log(3);dense_error=float((actual-expected).abs().max());assert dense_error<1e-10
    null=ext.integrated_bf(torch.zeros_like(r),var,torch.zeros_like(z));assert float(null.abs().max())==0
    spike=np.zeros(512);spike[256]=100;y=np.zeros(512,np.int8);y[256]=1
    smoothed=F.avg_pool1d(F.pad(torch.tensor(spike)[None,None],(15,0)),16,stride=1).numpy().reshape(-1)
    point_AP=average_precision_score(y,spike);window_AP=average_precision_score(y,smoothed);assert point_AP==1 and window_AP==1/16
    x=torch.randn(2,384,8,device=device,dtype=dtype);g=torch.zeros(8,8,device=device,dtype=dtype);g[0,1]=1
    m=CausalMetaBasis(348).to(device).eval().requires_grad_(False);ck=torch.load(a.weights/'normal_only_s348.pth',map_location='cpu',weights_only=False);m.load_state_dict(ck['model'])
    with torch.no_grad():o=m(x,g);_,old_error=readouts(m,o);modified=x.clone();modified[:,350,1]+=100;modified[:,351:]+=17;o2=m(modified,g)
    # Own current and all future points cannot change normal mean at t=350.
    mu_error=float((o['mean'][:,350-320,1]-o2['mean'][:,350-320,1]).abs().max());assert mu_error==0
    with torch.no_grad():heads,audit=ext.evidence_heads(x,g,o['residual'],o['variance'],'preflight')
    assert all(torch.isfinite(v['score']).all() for v in heads.values())
    assert all(len(v['parents'])==len(v['nonparents']) for v in audit['partners'].values())
    result=dict(status='passed',dense_likelihood_max_error=dense_error,null_BF_max_error=0.,own_current_future_normal_mean_error=mu_error,
        old_head_error=old_error,ideal_spike_AP_point=point_AP,ideal_spike_AP_trailing16=window_AP,technical_core_input_slots=2*x.numel(),
        technical_likelihood_factorizations=audit['factorization_count']+6*2*31,checkpoint_SHA256=sha256(a.weights/'normal_only_s348.pth'),target_arrays_opened=False)
    atomic_json(a.output,result);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--conditional-code',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
