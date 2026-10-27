"""Source-only fixed-checkpoint graph/root diagnostics. No optimizer or target."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from causal_tsad.io import atomic_json,sha256
from tools.run_improvement3 import cache_rows,load,evaluate_model,save_npz
from proposals.improvement3.models import CandidateMethod
from proposals.improvement3.metrics import evaluate

def model_from(item):
    p=Path(item['path']);assert sha256(p)==item['SHA256']
    cp=torch.load(p,map_location='cpu',weights_only=False)
    m=CandidateMethod(cp['arm'],cp['seed']).cuda().eval().requires_grad_(False)
    m.load_state_dict(cp['model']);return m,cp

@torch.no_grad()
def run(plan,task,out):
    start=time.time();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(plan['worker_GiB']*2**30/torch.cuda.get_device_properties(0).total_memory)
    root=Path(plan['source_root']);pp=root/'job_plan.json';assert sha256(pp)==plan['source_plan_SHA256']
    records=cache_rows(root,json.loads(pp.read_text()),plan['profile'])
    out.mkdir(parents=True,exist_ok=False);extra={};exposure=0
    if task['kind']=='graph_cross':
        model,cp=model_from(task['checkpoint']);result,arrays=evaluate_model(root,model,records)
        original=[]
        for rec in records:
            row=rec['task']['row']
            if row['split']!='confirmation':continue
            for item in rec['files']:
                vi=item['view'];key=row['key']+'__'+str(vi)
                with np.load(root/item['path']) as f:y=f['labels'].copy()
                original.append(dict(key=row['key'],view=vi,**evaluate(arrays[key],y,cp['threshold'],arrays.get('root__'+key),row['root'])))
        exposure=result['frozen_evaluation_exposure'];extra=dict(checkpoint_original_threshold=cp['threshold'],original_threshold_metrics=original)
    elif task['kind']=='root_readout':
        model,cp=model_from(task['checkpoint']);assert cp['arm']=='stat_typed'
        old=Path(task['original_scores']);assert sha256(old)==task['original_scores_SHA256']
        metrics=[];arrays={};maxerr=0.
        with np.load(old) as before:
            for rec in records:
                row=rec['task']['row']
                if row['split']!='confirmation':continue
                for item in rec['files']:
                    vi=item['view'];z,_,label=load(root,item);p=model(z);exposure+=item['input_scalars']
                    key=row['key']+'__'+str(vi);score=before[key].copy()
                    maxerr=max(maxerr,float(np.max(np.abs(score-p['score'].cpu().numpy()))))
                    # Detection is reused byte-for-byte; only RCA readout changes.
                    types=p['types'];fault=torch.logsumexp(types[...,:3],-1)-math.log(3)
                    odds=torch.logsumexp(types[...,:3],-1)-types[...,3]
                    original=p['root'].cpu().numpy();assert np.allclose(original,before['root__'+key],atol=1e-8,rtol=0)
                    for name,node in [('original',original),('fault_only',fault.cpu().numpy()),('fault_vs_normal',odds.cpu().numpy())]:
                        arrays[name+'__root__'+key]=node
                        metrics.append(dict(method=name,key=row['key'],C=row['C'],family=row['family'],view=vi,
                            background='plain' if vi<8 else 'smooth',fault=row['fault_names'][vi%8],
                            **evaluate(score,label.cpu().numpy(),cp['threshold'],node,row['root'])))
        assert maxerr<1e-8
        result=dict(metrics=metrics,threshold=cp['threshold']);extra=dict(detection_scores_reused_exact=True,recomputed_score_max_error=maxerr)
    elif task['kind']=='bound':
        metrics=[];arrays={}
        for rec in records:
            row=rec['task']['row']
            if row['split']!='confirmation':continue
            fpath=root/rec['frozen_path'];assert sha256(fpath)==rec['frozen_SHA256']
            with np.load(fpath) as f:
                for item in rec['files']:
                    vi=item['view'];p=root/item['path'];assert sha256(p)==item['SHA256']
                    with np.load(p) as c:y=c['labels'].copy()
                    if not y.any():continue
                    raw=f['raw'][vi];oracle=raw+np.where(y>0,2.,-2.)
                    for name,score in [('raw',raw),('bounded_label_oracle',oracle)]:
                        metrics.append(dict(method=name,key=row['key'],view=vi,fault=row['fault_names'][vi%8],background='plain' if vi<8 else 'smooth',**evaluate(score,y,0.)))
        result=dict(metrics=metrics);extra=dict(uses_confirmation_labels=True,deployable=False,only_AP_PR_AUC_is_bound=True)
    else:raise ValueError(task['kind'])
    if arrays:save_npz(out/'scores.npz',**arrays)
    atomic_json(out/'result.json',dict(status='complete',task=task,result=result,**extra,
      cached_head_input_scalars=exposure,optimizer_steps=0,new_basis_forwards=0,new_TimeRCD_forwards=0,
      target_arrays_opened=False,seconds=time.time()-start,plan_SHA256=plan['_SHA256'],
      score_SHA256=sha256(out/'scores.npz') if arrays else None))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--task',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());plan['_SHA256']=sha256(a.plan);run(plan,plan['tasks'][a.task],a.output)
