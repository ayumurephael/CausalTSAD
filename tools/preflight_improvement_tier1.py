"""Leakage, temporal support, exact identity and empty-graph input controls."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from causal_tsad.io import atomic_json
from proposals.causal_meta_basis import CausalMetaBasis
from proposals.context_crossfit import configurations,context_mask,blocks,predict
from proposals.relative_context_scores import readouts,average
import tools.run_improvement_tier1  # All worker imports must resolve in snapshot.
def run(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    m=CausalMetaBasis(348).cuda().eval().requires_grad_(False)
    rng=np.random.RandomState(7321);raw=rng.normal(size=(640,3));x=torch.tensor(raw,dtype=torch.float64,device='cuda')
    g=torch.zeros((3,3),dtype=x.dtype,device='cuda');g[0,1]=1;g[1,2]=1
    config=configurations()['xfit'];o=predict(m,x,g,config)
    # Held-out own current value cannot enter its own prediction via fitting.
    change=x.clone();change[320,0]+=100
    p=predict(m,change,g,config);selferr=float((p['prediction'][0,0]-o['prediction'][0,0]).abs());assert selferr<1e-9
    for lo,hi in blocks(640,8,'crossfit'):
        t=np.arange(32,640);keep=context_mask(640,lo,hi,0,range(1,33),'crossfit')
        assert not np.any(keep&((t>=lo)&(t<hi)))
        assert all(not np.any(keep&((t-l>=lo)&(t-l<hi))) for l in range(1,33))
    online=configurations()['online'];a0=predict(m,x,g,online);change=x.clone();change[321:]*=3
    a1=predict(m,change,g,online);futureerr=float((a0['prediction'][0]-a1['prediction'][0]).abs().max());assert futureerr<1e-9
    empty=torch.zeros_like(g);oe=predict(m,x,empty,config);assert torch.equal(oe['baseline'],x[320:])
    r=o['residual'].cpu().numpy();v=o['variance'].cpu().numpy();z=readouts(r,v)
    from scipy.special import logsumexp
    expected=logsumexp(.5*((2/3)*r*r/v-np.log(3)),axis=-1)-np.log(3)
    err=float(np.max(np.abs(z['point']['score']-expected)));assert err<1e-9
    sequence=np.arange(15.)[:,None];expected=np.array([np.mean(np.maximum(np.arange(t-3,t+1),0)) for t in range(15)])
    assert np.max(np.abs(average(sequence,4,False)[:,0]-expected))<1e-12
    for name in ['period','parent32','period_parent32']:
        result=predict(m,x,g,configurations()[name]);assert torch.isfinite(result['residual']).all()
    assert all(not p.requires_grad for p in m.parameters())
    from causal_tsad.frozen_detector import FrozenDetector
    detector=FrozenDetector(a.detector_code,a.detector_weights)
    for name,data in [('selftrained_uni',raw[:128,:1]),('selftrained_multi',raw[:128]),('selftrained_multi',np.concatenate([raw[:128],raw[:128]],axis=1))]:
        d=detector.predict(data,name,'native');assert len(d['logit'])==128 and np.isfinite(d['logit']).all()
    detector.verify_frozen()
    atomic_json(a.output,dict(status='passed',own_current_prediction_error=selferr,online_future_prediction_error=futureerr,
        strict_holdout_response_and_lag_support_purge=True,empty_graph_parentremoval_identity=True,raw_energy_error=err,
        trailing_average_exact=True,E2_finite=True,technical_input_scalar_slots=8*raw.size,global_updates=0,
        benchmark_opened=False,worker_imports_passed=True,TimeRCD_technical_scalars=detector.actual_input_scalars,TimeRCD_technical_calls=detector.encoder_calls,
        notes='Full-sequence ACF/statistic controls are explicitly offline, not strict holdout-independent statistics.'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--detector-code',type=Path,required=True);p.add_argument('--detector-weights',type=Path,required=True);run(p.parse_args())
