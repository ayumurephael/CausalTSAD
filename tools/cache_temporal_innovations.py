"""Only stage with frozen normal predictor forwards; each seed/source/background once."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from causal_tsad.io import atomic_json,sha256,array_sha
from proposals.causal_meta_basis import CausalMetaBasis,FIRST
from proposals.temporal_innovation_head import features
from tools.train_meta_basis import nuisance,psha
def run(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    plan=json.loads(a.plan.read_text());source=Path(plan['source']);assert sha256(source/'manifest.json')==plan['source_SHA256'];rows=json.loads((source/'manifest.json').read_text())['files'];out=[];cost=0;query=0;models={};hashes={}
    for seed in plan['seeds']:
        p=Path(plan['weights'])/('normal_only_s'+str(seed)+'.pth');assert sha256(p)==plan['weight_SHA256']['normal_only_s'+str(seed)]
        m=CausalMetaBasis(seed).cuda().eval().requires_grad_(False);m.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']);models[seed]=m;hashes[seed]=psha(m)
    tasks=[(seed,row,bg) for seed in plan['seeds'] for row in rows for bg in ['plain','smooth']]
    for seed,row,bg in tasks[a.worker::a.workers]:
        assert time.time()<plan['deadline_unix'];p=source/row['path'];assert sha256(p)==row['SHA256']
        with np.load(p) as f:cases=f['cases'];graph=f['graph'];labels=f['labels'][:,FIRST:]
        assert array_sha(cases)==row['cases_SHA256']
        if row['split']=='calibration':cases=cases[:2];labels=labels[:2]
        x=nuisance(cases,row['key'],bg);values=torch.tensor(x,device='cuda',dtype=torch.float64)
        with torch.no_grad():o=models[seed](values,torch.tensor(graph,device='cuda',dtype=torch.float64));ff=features(o['residual'],o['variance']);energy=.5*((2/3)*o['residual'].square()/o['variance']-np.log(3))
        path=a.root/'features'/('s'+str(seed)+'_'+row['key']+'_'+bg+'.npz');assert not path.exists();np.savez_compressed(path,features=ff.cpu().numpy(),energy=energy.cpu().numpy(),labels=labels,temperature=float(models[seed].log_temperature.exp().clamp(.1,10)),bias=float(models[seed].bias))
        record=dict(seed=seed,key=row['key'],background=bg,split=row['split'],C=row['C'],path=str(path.relative_to(a.root)),SHA256=sha256(path),input_SHA256=array_sha(x),feature_SHA256=array_sha(ff.cpu().numpy()),source_file_SHA256=row['SHA256'])
        out.append(record);cost+=x.size;query+=len(x)*(x.shape[1]-FIRST)*x.shape[-1]
        atomic_json(a.root/('cache'+str(a.worker)+'_progress.json'),dict(status='running',completed=len(out),full_input_slots=cost))
    assert all(psha(m)==hashes[s] for s,m in models.items())
    atomic_json(a.root/('cache'+str(a.worker)+'_manifest.json'),dict(status='complete',records=out,full_input_slots=cost,query_slots=query,normal_model_parameters_unchanged=True,parameter_SHA256=hashes,weight_SHA256=plan['weight_SHA256'],optimizer_updates=0))
    atomic_json(a.root/('cache'+str(a.worker)+'_progress.json'),dict(status='complete',completed=len(out),full_input_slots=cost))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,required=True);a=p.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors'/('cache'+str(a.worker)+'.json'),dict(error=traceback.format_exc()));raise
