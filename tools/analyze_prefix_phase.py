"""Paired period/phase interactions with seed aggregation and exact input audit."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(xs):return float(np.mean(xs))
def delta(xs):
    x=np.asarray(xs,float);rng=np.random.RandomState(348);b=x[rng.randint(len(x),size=(10000,len(x)))].mean(1);return dict(mean=mean(x),CI95=np.quantile(b,[.025,.975]).tolist(),n=len(x),positive=int((x>0).sum()),negative=int((x<0).sum()))
def run(a):
    root=a.root/'2229';assert (root/'fetch_receipt.json').exists();manifest=read(root/'manifest.json');assert manifest['status']=='complete';rows=[];slots=queries=0;inputs={}
    for job in manifest['jobs']:
        assert job['parameters_unchanged'] and job['optimizer_updates']==0;slots+=job['full_input_scalar_slots'];queries+=job['query_scalar_forwards']
        for item in job['records']:
            p=root/item['path'];assert sha(p)==item['SHA256'];v=read(p);assert sha(p.with_suffix('.npz'))==v['scores_SHA256'];ik=(v['key'],v['period'],v['phase']);inputs.setdefault(ik,set()).add(v['input_SHA256']);metrics=['AP','physical_AP','FPR','normal_NLL','normal_whitened_square','normal_whitened_q99'];record={k:v[k] for k in ['key','C','family','arm','mode','period','phase','normal_outside_prefix_range_fraction','normal_prefix_range_median','normal_query_range_median']};record.update({k:mean([v['methods'][str(s)][k] for s in [348,349,350]]) for k in metrics});record['Top1']=mean([e['Top1'] for s in [348,349,350] for e in v['methods'][str(s)]['RCA'] if e['kind']=='matched_measurement']);rows.append(record)
    assert len(rows)==512 and slots==176160768 and queries==121110528 and all(len(v)==1 for v in inputs.values());lookup={(v['arm'],v['mode'],v['period'],v['phase'],v['key']):v for v in rows};summary={};effects={};interactions={};fields=['AP','physical_AP','FPR','Top1','normal_NLL','normal_whitened_square','normal_whitened_q99','normal_outside_prefix_range_fraction']
    for arm in ['frozen_basis','normal_only']:
        for mode in ['true','empty']:
            for period in [.75,1.5]:
                for phase in ['crossing','extremum']:
                    xs=[v for v in rows if (v['arm'],v['mode'],v['period'],v['phase'])==(arm,mode,period,phase)];assert len(xs)==32;summary[arm+'/'+mode+'/'+str(period)+'/'+phase]={k:mean([v[k] for v in xs]) for k in fields}
                peaks=[v for v in rows if (v['arm'],v['mode'],v['period'],v['phase'])==(arm,mode,period,'extremum')];effects[arm+'/'+mode+'/'+str(period)]={k:delta([v[k]-lookup[(arm,mode,period,'crossing',v['key'])][k] for v in peaks]) for k in fields}
            keys=[v['key'] for v in rows if (v['arm'],v['mode'],v['period'],v['phase'])==(arm,mode,.75,'crossing')]
            interactions[arm+'/'+mode]={k:delta([(lookup[(arm,mode,1.5,'extremum',key)][k]-lookup[(arm,mode,1.5,'crossing',key)][k])-(lookup[(arm,mode,.75,'extremum',key)][k]-lookup[(arm,mode,.75,'crossing',key)][k]) for key in keys]) for k in ['AP','FPR','Top1','normal_whitened_square']}
    result=dict(status='complete',source_systems=32,method_records=512,optimization_seeds=3,full_input_scalar_slots=slots,query_scalar_forwards=queries,inputs_bitwise_equal_for_paired_arms_and_graphs=True,summary=summary,phase_effect_extremum_minus_crossing=effects,period_phase_interaction=interactions,source_only=True,optimizer_updates=0,benchmark_arrays_opened=False,conditional_RCA=True,interpretation_limit='phase changes local slope/curvature and excitation as well as range; not a unique mediation proof')
    for name,value in [('analysis.json',result),('paired_cases.json',rows)]:
        p=a.root/name;assert not p.exists();p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(summary=summary,normal_only_true_phase_effect={str(p):effects['normal_only/true/'+str(p)] for p in [.75,1.5]})))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args())
