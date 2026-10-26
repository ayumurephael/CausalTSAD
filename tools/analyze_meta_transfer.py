"""Full four-metric paired transfer analysis, system-level seed aggregation."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(xs):
    xs=[x for x in xs if x is not None];return float(np.mean(xs)) if xs else None
def delta(xs):
    x=np.asarray(xs,float);rng=np.random.RandomState(348);b=x[rng.randint(len(x),size=(10000,len(x)))].mean(1);return dict(mean=float(x.mean()),CI95=np.quantile(b,[.025,.975]).tolist(),n=len(x),positive=int((x>0).sum()),negative=int((x<0).sum()))
def run(a):
    root=a.project/'results/c04_meta_basis_generated_transfer_v1';rows=[];costs={k:0 for k in ['full_input_scalar_slots','query_scalar_forwards','context_factorizations']};mapping={'Affiliation_F':'Affiliation-F','F1_T':'F1_T','Standard_F1':'Standard-F1','VUS_PR':'VUS-PR'}
    for port in ['2226','2229']:
        local=root/port;assert (local/'fetch_receipt.json').exists();plan=read(local/'job_plan.json');manifest=read(local/'manifest.json');receipt=read(local/'fetch_receipt.json');assert manifest['status']=='complete';receipts={v['id']:v for v in receipt['records']}
        for w in manifest['workers']:
            assert w['parameters_unchanged']
            for k in costs:costs[k]+=w[k]
        for meta in plan['rows']:
            ident=meta['id'];p=local/'metrics'/(ident+'.json');assert sha(p)==receipts[ident]['metric_SHA256'];v=read(p);inf=read(local/'inference'/(ident+'.json'));assert sha(local/'inference'/(ident+'.json'))==receipts[ident]['inference_SHA256'];scorefile=local/'inference'/(ident+'.npz');assert sha(scorefile)==v['scores_SHA256']==inf['scores_SHA256'];assert inf['input_observed_SHA256']==meta['time_series_sha256'] and inf['input_normal_SHA256']==meta['normal_sha256'] and inf['weight_SHA256']==plan['weight_SHA256'];assert v['target_updates']==0
            old=read(a.project/'results/c04_syndromes_generated_pilot_v1'/port/'development/metrics'/(ident+'.json'))
            with np.load(scorefile) as f,np.load(a.project/'results/c04_context_likelihood_controls_v1'/port/'fixed_baseline_extract'/(ident+'.npz')) as ref:
                assert f['B0'].dtype==ref['B0'].dtype and np.array_equal(f['B0'],ref['B0'])
            assert v['methods']['B0']['metrics']==old['methods']['B0']['metrics'] and v['methods']['B0']['AP']==old['methods']['B0']['AP']
            for name in ['B0']+[arm+'__'+mode for arm in ['frozen_basis','normal_only','normal_relation'] for mode in ['true','empty','rewired']]:
                tags=['B0'] if name=='B0' else [name.split('__')[0]+'_s'+str(s)+'__'+name.split('__')[1] for s in [348,349,350]];entries=[v['methods'][tag] for tag in tags];events=[e for tag in tags for e in v['RCA'].get(tag,[])];ns=[len(v['RCA'].get(tag,[])) for tag in tags];assert len(set(ns))==1
                row=dict(id=ident,system_id=v['system_id'],track=v['track'],C=v['channels'],family=v['family'],event_types=v['event_types'],requested_strata=v['requested_strata'],method=name,AP=mean([e['AP'] for e in entries]),ROC_AUC=mean([e['ROC_AUC'] for e in entries]),FPR=mean([e['normal_FPR_source99'] for e in entries]),events=ns[0],**{k:mean([e['metrics'][mk] for e in entries]) for k,mk in mapping.items()},**{k:mean([e[k] for e in events]) for k in ['Top1','Top3','MRR']});rows.append(row)
    assert costs['full_input_scalar_slots']==234516384 and costs['query_scalar_forwards']==215162784 and len({v['id'] for v in rows})==80 and len(rows)==800
    fields=['AP','ROC_AUC','FPR','Affiliation_F','F1_T','Standard_F1','VUS_PR','Top1','Top3','MRR']
    def summarize(xs):return dict(systems=len(xs),AP_eligible=sum(v['AP'] is not None for v in xs),RCA_events=sum(v['events'] for v in xs),**{k:mean([v[k] for v in xs]) for k in fields})
    methods=sorted(set(v['method'] for v in rows));summary={m:summarize([v for v in rows if v['method']==m]) for m in methods};by_track={t:{m:summarize([v for v in rows if v['method']==m and v['track']==t]) for m in methods} for t in ['id','mechanism_holdout']};by_C={str(c):{m:summarize([v for v in rows if v['method']==m and v['C']==c]) for m in methods} for c in [8,12,16,20]};lookup={(v['method'],v['id']):v for v in rows};pairs={};case_deltas=[]
    comparisons=[('normal_only__true','frozen_basis__true'),('normal_relation__true','normal_only__true'),('normal_only__true','B0'),('normal_only__true','normal_only__empty'),('normal_only__true','normal_only__rewired')]
    for aa,bb in comparisons:
        xs=[v for v in rows if v['method']==aa];pairs[aa+'-'+bb]={k:delta([v[k]-lookup[(bb,v['id'])][k] for v in xs if v[k] is not None and lookup[(bb,v['id'])][k] is not None]) for k in fields if any(v[k] is not None and lookup[(bb,v['id'])][k] is not None for v in xs)}
        case_deltas.extend(dict(id=v['id'],system_id=v['system_id'],C=v['C'],track=v['track'],family=v['family'],comparison=aa+'-'+bb,AP_difference=v['AP']-lookup[(bb,v['id'])]['AP']) for v in xs if v['AP'] is not None)
    result=dict(status='complete',independent_target_systems=80,optimization_seeds=3,conditions=27,costs=costs,summary=summary,by_track=by_track,by_C=by_C,paired=pairs,baseline_score_and_metrics_equivalent=True,TimeRCD_RCA_evaluated=False,RCA_scope='conditional true event window',FPR_caveat='B0 old plain calibration pool; new causal methods same pooled source normal set; not strictly equal B0 calibration pool',target_updates=0,validation_or_locked_arrays_opened=False)
    for name,value in [('analysis.json',result),('paired_cases.json',rows),('all_changes.json',case_deltas)]:
        p=root/name;assert not p.exists();p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(summary=summary,paired=pairs)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);run(p.parse_args())
