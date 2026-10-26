"""G0 checks before any bounded I09 run; outputs audit only."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from causal_tsad.io import atomic_json,sha256
from causal_tsad.frozen_detector import parameter_sha
from proposals.context_mixture import components,ContextSelector,mixture_logpdf,node_evidence
from proposals.context_crossfit import predict,configurations,blocks
from tools.run_i09_mixture import load_model

def main(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    plan=json.loads(a.plan.read_text());model=load_model(plan);initial=parameter_sha(model)
    row=json.loads((Path(plan['source_eval'])/'manifest.json').read_text())['rows'][0]
    with np.load(Path(plan['source_eval'])/row['path']) as d:x=torch.as_tensor(d['values'][2],device='cuda',dtype=torch.float64);g=torch.as_tensor(d['graph'],device='cuda',dtype=torch.float64)
    o=components(model,x,g);o={k:(v.clone() if torch.is_tensor(v) else v) for k,v in o.items()}
    p=predict(model,x,g,configurations()['xfit'])
    e1=float((o['response']-o['mu'][...,0]-p['residual']).abs().max());e2=float((o['variance'][...,0]-p['variance']).abs().max());assert max(e1,e2)<1e-8
    sel=ContextSelector(348).cuda().double();lw=sel(o['features'],o['valid'])
    uniform=o['valid'].double()/o['valid'].sum(-1,keepdim=True);initerr=float((lw.exp()-uniform).abs().max());assert initerr<1e-12
    perm=torch.tensor([3,1,4,0,2],device='cuda');err=float((sel(o['features'][...,perm,:],o['valid'][...,perm])-lw[...,perm]).nan_to_num().abs().max());assert err<1e-12
    full=torch.full_like(lw,-torch.inf);full[...,0]=0
    energyerr=float((node_evidence(o['response'],o['mu'],o['variance'],full)-.5*((2/3)*p['residual'].square()/p['variance']-np.log(3))).abs().max());assert energyerr<1e-8
    # Alter query values: their own block's fitted components and selector
    # context statistics must not change (query predictors can legitimately).
    lo,hi=blocks(1024,8,'crossfit')[4];bad=x.clone();bad[lo:hi]+=17
    b=components(model,bad,g);protected=float((o['features'][lo-320:hi-320]-b['features'][lo-320:hi-320]).abs().max());assert protected==0
    node=0;t=600;bad=x.clone();bad[t,node]+=10;bad[t+1:,node]+=20
    # For feature-causality use the same fixed fold normalization/context.
    # Future values outside a fold can affect offline fit legitimately, so
    # the strict own-current test is checked on design(), not whole fitting.
    from proposals.context_crossfit import design
    ts=torch.arange(32,1024,device='cuda');pa=torch.where(g[:,node]>0)[0].tolist()
    f,ps=design(model,x,node,pa,ts);h,_=design(model,bad,node,pa,ts);ownerr=float((f[t-32]-h[t-32]).abs().max());assert ownerr==0
    loss=-mixture_logpdf(o['response'].detach(),o['mu'].detach(),o['variance'].detach(),lw).mean();loss.backward()
    assert all(p.grad is None for p in model.parameters()) and any(p.grad is not None and p.grad.abs().sum()>0 for p in sel.parameters())
    opt=torch.optim.Adam(sel.parameters(),lr=.001);opt.step();assert parameter_sha(model)==initial
    atomic_json(a.output,dict(status='passed',full_residual_error=e1,full_variance_error=e2,full_energy_error=energyerr,
        zero_selector_uniform_error=initerr,component_permutation_error=err,protected_context_feature_error=protected,
        own_current_future_feature_error=ownerr,frozen_basis_unchanged=True,gradients_selector_only=True,selector_parameters=sum(p.numel() for p in sel.parameters()),
        technical_component_input_scalars=2*x.numel(),technical_reference_input_scalars=x.numel(),technical_selector_updates=1,
        component_budget={k:2*v for k,v in o['audit'].items()},source_row=row['key'],source_SHA256=row['SHA256']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
