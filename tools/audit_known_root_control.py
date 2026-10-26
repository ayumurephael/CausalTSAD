"""Match time window/root information before interpreting the oracle LR gap."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import average_precision_score
from causal_tsad.io import atomic_json,sha256
from tools.audit_source_edge_oracle import FIRST,lse,mean16
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--oracle-results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    oldmanifest=json.loads((a.oracle_results/'manifest.json').read_text());assert oldmanifest['status']=='complete' and sha256(a.oracle_results/'cases.json')==oldmanifest['cases_SHA256'];prior=json.loads((a.oracle_results/'cases.json').read_text());lookup={(v['key'],v['kind'],v['duration'],v['base']):v for v in prior};plan=json.loads((a.probe/'job_plan.json').read_text());out=[];slots=0;maximum=0.
    for row in [v for v in plan['tasks'] if v['kind']=='source']:
        path=Path(plan['source'])/row['path'];assert sha256(path)==row['SHA256']
        with np.load(path) as f:x=f['cases'];graph=f['graph'];labels=f['labels'][:,FIRST:]
        pp=json.loads((Path(plan['source'])/(row['key']+'_audit_NOT_MODEL_INPUT.json')).read_text())['parameters'];perm=np.asarray(pp['new_to_old']);rho=np.asarray(pp['rho'])[perm];b=np.asarray(pp['coefficient_before_permutation'])[perm][:,perm];lag=np.asarray(pp['lag_before_permutation'])[perm][:,perm];sigma=np.asarray(pp['sigma'])[perm]
        transform=(lambda z:z) if row['family']=='linear' else np.tanh if row['family']=='tanh' else np.sin
        ids=[0,1]+[i for i,m in enumerate(row['case_metadata']) if m['kind'] in ['edge_coefficient','edge_lag']];xx=x[ids];t=np.arange(FIRST,x.shape[1]);mu=xx[:,t-1]*rho[None,None]
        for child in range(row['C']):
            for parent in np.flatnonzero(graph[:,child]):mu[:,:,child]+=b[parent,child]*transform(xx[:,t-lag[parent,child],parent])
        q=np.square((xx[:,FIRST:]-mu)/sigma[None,None]);e=.5*((2/3)*q-np.log(3));slots+=xx.shape[0]*len(t)*row['C']
        for j,i in enumerate(ids[2:],2):
            m=row['case_metadata'][i];y=labels[i]>0;root=m['root'];reference=lookup[(row['key'],m['kind'],m['duration'],m['base'])];point=float(average_precision_score(y,lse(e[j])));error=abs(point-reference['oracle_normal_model_energy_AP']);assert error<1e-10;maximum=max(maximum,error)
            pooled=np.stack([mean16(e[j,:,node]) for node in range(row['C'])],-1)
            out.append(dict(key=row['key'],C=row['C'],family=row['family'],kind=m['kind'],duration=m['duration'],base=m['base'],unknown_root_energy_AP=point,known_root_energy_AP=float(average_precision_score(y,e[j,:,root])),unknown_root_energy16_AP=float(average_precision_score(y,lse(pooled))),known_root_energy16_AP=float(average_precision_score(y,pooled[:,root])),known_root_edge_alternative_LR_AP=reference['oracle_known_root_edge_alternative_LR_AP'],known_root_edge_alternative_LR16_AP=reference['oracle_known_root_edge_alternative_LR16_AP']))
    atomic_json(a.output/'cases.json',out);atomic_json(a.output/'manifest.json',dict(status='complete',systems=len({v['key'] for v in out}),cases=len(out),oracle_prediction_query_slots=slots,original_oracle_AP_replay_max_error=maximum,source_plan_SHA256=sha256(a.probe/'job_plan.json'),old_oracle_cases_SHA256=oldmanifest['cases_SHA256'],cases_SHA256=sha256(a.output/'cases.json'),new_backbone_forwards=0,optimizer_updates=0,target_arrays_opened=False,window_policy='all16-readouts partial-valid leading windows, exactly same point set',scope='all privileged oracle; LR still additionally knows actual alternative, never a deployable RCA score'))
