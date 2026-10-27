"""Frozen actual-neural-model context controls on the existing source systems.

Every background is paired on identical base observations. The old 32 source
confirmation systems are reused: this is NOT a fresh blind confirmation.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,argparse,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from causal_tsad.io import atomic_json,sha256,array_sha
from causal_tsad.frozen_detector import parameter_sha
from proposals.causal_meta_basis import CausalMetaBasis
from proposals.improvement3.components import components as original
from proposals.nondescendant_components import components
from proposals.nondescendant_context import preflight,donors
from tools.run_improvement3 import save_npz

MODES=['original32','none65','fourier65','nondesc65','all65','sham65']
BACKGROUNDS=['plain','period075','period150','multifrequency','subset_piecewise']

def backgrounds(base,key):
    T,C=base.shape;rng=np.random.default_rng(348200+int(key.split('_')[-1]))
    t=np.arange(T)[:,None];sd=base.std(0).clip(1e-3)
    phi=rng.uniform(0,2*np.pi,(1,C));out={'plain':np.zeros_like(base)}
    for name,p in [('period075',.75),('period150',1.5)]:out[name]=4*sd*np.sin(2*np.pi*t/(p*T)+phi)
    phases=rng.uniform(0,2*np.pi,(3,1,C))
    out['multifrequency']=sum((4/np.sqrt(3))*sd*np.sin(2*np.pi*t/(p*T)+phases[i]) for i,p in enumerate([.35,.75,1.5]))
    subset=np.zeros((1,C));subset[:,rng.choice(C,max(1,C//2),replace=False)]=1
    ramp=np.maximum(0,(t/T-.2)) -2*np.maximum(0,(t/T-.5))+np.maximum(0,(t/T-.8))
    out['subset_piecewise']=12*sd*subset*ramp
    return out

@torch.no_grad()
def main(a):
    start=time.time();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    p=json.loads(a.plan.read_text());torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(0).total_memory)
    b=CausalMetaBasis(348).cuda().eval().requires_grad_(False)
    assert sha256(p['weight'])==p['weight_SHA256'];b.load_state_dict(torch.load(p['weight'],map_location='cpu',weights_only=False)['model']);before=parameter_sha(b)
    if a.task<0:
        check=preflight();rng=np.random.default_rng(348);v=torch.as_tensor(rng.normal(size=(400,5)),device='cuda');g=torch.zeros((5,5),device='cuda',dtype=torch.float64);g[0,1]=g[1,2]=1
        x=original(b,v,g);y=components(b,v,g,'original32')
        err=max(float((x[k]-y[k]).abs().max()) for k in ['mu','variance','response','contribution'])
        assert err==0;check['original_pipeline_max_error']=err
        for mode in MODES[1:]:
            z=components(b,v,g,mode);assert torch.isfinite(z['mu']).all()
        check['basis_parameter_SHA256']=before;check['all_six_modes_finite']=True
        atomic_json(a.output,check);return
    task=p['tasks'][a.task];row=task['row'];src=Path(p['source'])/row['path'];assert sha256(src)==row['SHA256']
    with np.load(src) as f:values=f['values'][:8].copy();labels=f['labels'][:8].copy();graph=f['graph'].copy()
    g=torch.as_tensor(graph,device='cuda',dtype=torch.float64);bg=backgrounds(values[0],row['key'])
    out=a.output;out.mkdir(parents=True,exist_ok=False);files=[];exposure=factors=0
    views=range(2) if row['split']=='calibration' else range(8)
    for name,delta in bg.items():
        arrays={}
        for vi in views:
            x=values[vi]+delta;z=components(b,torch.as_tensor(x,device='cuda'),g,task['mode'])
            arrays['r'+str(vi)]=(z['response']-z['mu'][...,0]).cpu().numpy()
            arrays['v'+str(vi)]=z['variance'][...,0].cpu().numpy()
            arrays['g'+str(vi)]=z['contribution'][...,0].cpu().numpy()
            arrays['y'+str(vi)]=labels[vi,32:];arrays['input_sha'+str(vi)]=np.array(array_sha(x))
            exposure+=x.size;factors+=z['audit']['factorizations']
        arrays['edges']=z['edges'].cpu().numpy();fn=out/(name+'.npz');save_npz(fn,**arrays)
        files.append(dict(path=str(fn),SHA256=sha256(fn),background=name))
    assert parameter_sha(b)==before
    atomic_json(out/'result.json',dict(status='complete',task=task,source_SHA256=sha256(src),files=files,
       input_scalars=exposure,factorizations=factors,optimizer_steps=0,TimeRCD_forwards=0,
       benchmark_arrays_read=False,basis_parameter_SHA256=before,basis_unchanged=True,
       donor_counts=[len(donors(g,j,task['mode'][:-2])) for j in range(len(g))] if task['mode'] in ['nondesc65','all65','sham65'] else None,
       plan_SHA256=sha256(a.plan),seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--task',type=int,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
