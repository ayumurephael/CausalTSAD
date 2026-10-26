"""Oracle diagnostic only. True coefficients/lag/noise NEVER a deployable input."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,traceback,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import average_precision_score
from causal_tsad.io import atomic_json,sha256,array_sha
FIRST=320
def lse(e):
    m=e.max(-1);return m+np.log(np.exp(e-m[...,None]).sum(-1))-math.log(e.shape[-1])
def mean16(x):
    cs=np.pad(x.cumsum(0),(1,0));t=np.arange(len(x));lo=(t+1-16).clip(0);return (cs[t+1]-cs[lo])/(t+1-lo)
def dcor(x,y):
    if len(x)<4:return None
    x=np.asarray(x).reshape(len(x),-1);y=np.asarray(y).reshape(len(y),-1)
    a=np.linalg.norm(x[:,None]-x[None],axis=-1);b=np.linalg.norm(y[:,None]-y[None],axis=-1)
    a=a-a.mean(0)[None]-a.mean(1)[:,None]+a.mean();b=b-b.mean(0)[None]-b.mean(1)[:,None]+b.mean()
    denominator=np.sqrt(np.mean(a*a)*np.mean(b*b))
    return float(np.sqrt(max(float(np.mean(a*b)/denominator),0))) if denominator>0 else 0.
def run(a):
    assert not (a.output/'manifest.json').exists();a.output.mkdir(parents=True,exist_ok=True);plan=json.loads((a.probe/'job_plan.json').read_text());rows=[v for v in plan['tasks'] if v['kind']=='source'];out=[];slots=0;cacheapps=0
    for row in rows:
        p=Path(plan['source'])/row['path'];assert sha256(p)==row['SHA256']
        with np.load(p) as f:x=f['cases'];g=f['graph'];labels=f['labels'][:,FIRST:]
        private=json.loads((Path(plan['source'])/(row['key']+'_audit_NOT_MODEL_INPUT.json')).read_text());params=private['parameters'];perm=np.asarray(params['new_to_old']);rho=np.asarray(params['rho'])[perm];b=np.asarray(params['coefficient_before_permutation'])[perm][:,perm];lag=np.asarray(params['lag_before_permutation'])[perm][:,perm];sigma=np.asarray(params['sigma'])[perm];f=(lambda z:z) if row['family']=='linear' else np.tanh if row['family']=='tanh' else np.sin
        rng=np.random.default_rng(row['seed']+91);child=int(rng.choice(np.flatnonzero(g.sum(0)>0)));parent=int(rng.choice(np.flatnonzero(g[:,child])))
        ids=[0,1]+[i for i,m in enumerate(row['case_metadata']) if m['kind'] in ['edge_coefficient','edge_lag']];xx=x[ids];t=np.arange(FIRST,x.shape[1]);mu=xx[:,t-1]*rho[None,None]
        for node in range(row['C']):
            for par in np.flatnonzero(g[:,node]):mu[:,:,node]+=b[par,node]*f(xx[:,t-lag[par,node],par])
        residual=xx[:,FIRST:]-mu;q=np.square(residual/sigma[None,None])
        energy=.5*((2/3)*q-np.log(3));oracle_scores=lse(energy);oracle_normal_q=float(q[:2].mean());stats={}
        for seed in [348,349,350]:
            path=a.probe/'innovation_cache'/('source_'+row['key']+'__s'+str(seed)+'__true.npz');old=json.loads((a.probe/'records'/('source_'+row['key']+'.json')).read_text());meta=next(v for v in old['innovation_cache'] if v['path']==str(path.relative_to(a.probe)).replace('\\','/'));assert sha256(path)==meta['SHA256']
            with np.load(path) as z:stats[seed]=(z['residual'][ids],z['variance'][ids]);cacheapps+=len(ids)*len(t)*row['C']
        for j,i in enumerate(ids[2:],2):
            m=row['case_metadata'][i];assert m['root']==child;y=labels[i]>0;duration=m['duration'];mask=np.flatnonzero(y)
            phi=f(xx[j,t-lag[parent,child],parent]);shift=.25*b[parent,child]*phi if m['kind']=='edge_coefficient' else b[parent,child]*(f(xx[j,t-lag[parent,child]-8,parent])-phi)
            # Known alternative is evaluated at EVERY query time. Event windows
            # enter only metrics and signal-size audit, never the oracle score.
            lr=(residual[j,:,child]*shift-.5*shift**2)/(sigma[child]**2)
            dependency_features=phi if m['kind']=='edge_coefficient' else np.stack([phi,f(xx[j,t-lag[parent,child]-8,parent])],-1)
            estimated=[]
            for seed,(er,ev) in stats.items():
                rr=er[j,:,child]/np.sqrt(ev[j,:,child]);d1=dcor(rr[mask],dependency_features[mask]);r0=er[m['base'],:,child]/np.sqrt(ev[m['base'],:,child]);d0=dcor(r0[mask],dependency_features[mask]);estimated.append(dict(seed=seed,AP=old['metrics']['s'+str(seed)+'__true__point'][i]['AP'],conditional_dependence_change=None if d1 is None else d1-d0))
            oracle_r=residual[j,:,child]/sigma[child];normal_r=residual[m['base'],:,child]/sigma[child]
            dc=dcor(oracle_r[mask],dependency_features[mask]);dc0=dcor(normal_r[mask],dependency_features[mask])
            out.append(dict(key=row['key'],C=row['C'],family=row['family'],kind=m['kind'],duration=duration,base=m['base'],query_prevalence=float(y.mean()),oracle_normal_q=oracle_normal_q,
                oracle_normal_model_energy_AP=float(average_precision_score(y,oracle_scores[j])),oracle_known_root_edge_alternative_LR_AP=float(average_precision_score(y,lr)),oracle_known_root_edge_alternative_LR16_AP=float(average_precision_score(y,mean16(lr))),
                conditional_expected_KL=float(.5*np.sum((shift[mask]/sigma[child])**2)),oracle_residual_parent_dcor_change=None if dc is None else dc-dc0,estimated=estimated,
                oracle_scope='privileged root/changed-edge/normal coefficients/lag/noise for conditional diagnosis only; no deployable RCA claim'))
        slots+=len(ids)*len(t)*row['C']
    atomic_json(a.output/'cases.json',out);atomic_json(a.output/'manifest.json',dict(status='complete',systems=len(rows),edge_cases=len(out),oracle_normal_prediction_query_slots=slots,reused_estimated_cache_slot_applications=cacheapps,new_backbone_forwards=0,optimizer_updates=0,target_arrays_opened=False,
        source_plan_SHA256=sha256(a.probe/'job_plan.json'),cases_SHA256=sha256(a.output/'cases.json'),normal_parameters_only_oracle=True,conditional_actual_fault_parameter_oracle=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except Exception:a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
