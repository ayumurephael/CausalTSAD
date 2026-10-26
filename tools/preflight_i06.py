"""Numerical equivalence, input controls, actual 40-channel memory and G0."""
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from causal_tsad.io import atomic_json,sha256
from causal_tsad.frozen_detector import FrozenDetector,parameter_sha
from proposals.i06_input_controls import detector_scores,changed_residual
from proposals.causal_meta_basis import CausalMetaBasis
from proposals.context_crossfit import predict,configurations,context_mask

def main(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    plan=json.loads(a.plan.read_text());torch.cuda.set_per_process_memory_fraction(plan['worker_GiB']*2**30/torch.cuda.get_device_properties(0).total_memory)
    d=FrozenDetector(plan['detector_code'],plan['detector_weights']);rng=np.random.default_rng(930048)
    errors=[]
    for n,c in [(704,8),(992,40)]:
        x=rng.normal(size=(n,c));native=d.predict(x,'selftrained_multi','native')['logit'];new=detector_scores(d,x,c//2)['mean']
        errors.append(float(np.max(np.abs(native-new))))
    assert max(errors)<=1e-6;d.verify_frozen()
    row=next(t['row'] for t in plan['tasks'] if t['row']['C']==20)
    with np.load(Path(plan['source'])/row['path']) as z:v=z['values'][0];g=z['graph']
    m=CausalMetaBasis(348).cuda().eval().requires_grad_(False)
    weight=Path(plan['weights'])/'normal_only_s348.pth';assert sha256(weight)==plan['weight_SHA256']['348']
    m.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False)['model']);h=parameter_sha(m)
    x=torch.as_tensor(v,dtype=torch.float64,device='cuda');gg=torch.as_tensor(g,dtype=torch.float64,device='cuda')
    old=predict(m,x,gg,configurations()['xfit']);new=predict(m,x,gg,dict(configurations()['xfit'],first=32))
    diff={k:float(torch.max(torch.abs(old[k]-new[k][288:]))) for k in ['residual','variance','baseline','prediction']}
    assert max(diff.values())<1e-8 and parameter_sha(m)==h
    r=new['residual'].cpu().numpy()
    assert np.array_equal(np.sort(changed_residual(r,42,'time'),axis=0),np.sort(r,axis=0))
    assert np.array_equal(np.sort(changed_residual(r,42,'channel'),axis=1),np.sort(r,axis=1))
    t=np.arange(32,1024);mask=context_mask(1024,404,528,32,range(1,33),'crossfit')
    assert not ((t[mask,None]-np.arange(33)>=404)&(t[mask,None]-np.arange(33)<528)).any()
    atomic_json(a.output,dict(status='passed',native_mean_max_error=max(errors),first32_tail_equivalence=diff,
        frozen_parameters=True,zero_target_training=True,benchmark_arrays_opened=False,TimeRCD_RCA=False,
        actual_peak_cuda_bytes=torch.cuda.max_memory_allocated(),technical_detector_scalars=d.actual_input_scalars,
        technical_detector_calls=d.encoder_calls,technical_causal_scalars=2*v.size,
        technical_feature_rows=sum(o['audit']['feature_node_rows'] for o in [old,new]),
        technical_factorizations=sum(o['audit']['factorizations'] for o in [old,new])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
