"""Independent-system analysis, identical source thresholds, fixed12-run audit."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
ARMS=['frozen_basis','normal_only','normal_relation','wrong_graph_relation'];SEEDS=[348,349,350]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(xs):return float(np.mean(xs))
def diff(xs):
    x=np.asarray(xs,float);rng=np.random.RandomState(348);b=x[rng.randint(len(x),size=(10000,len(x)))].mean(1)
    return dict(mean=mean(x),CI95=np.quantile(b,[.025,.975]).tolist(),positive=int((x>0).sum()),negative=int((x<0).sum()),n=len(x))
def run(a):
    r=a.root;assert (r/'fetch_receipts.json').exists();receipts=read(r/'fetch_receipts.json');data={};runs={};input_plans=[];thresholds={};outcal={};cost=0;normalization_repeated_slots=0
    for port in ['2226','2229']:
        root=r/port;plan=read(root/'job_plan.json');input_plans.append(plan['training_keys'])
        for job in plan['jobs']:
            name=job['arm']+'_s'+str(job['seed']);rr=root/'runs'/name;m=read(rr/'manifest.json');assert m['status']=='complete' and m['observed_input_exposure']==7340032 and m['updates']==128 and len(m['records'])==384
            curves=read(rr/'curves.json');assert len(curves)==128 and [v['key'] for v in curves]==plan['training_keys'];assert sha(rr/'final.pth')==m['final_checkpoint_SHA256'] and sha(rr/'initial.pth')==m['initial_checkpoint_SHA256'];cost+=m['observed_input_exposure'];normalization_repeated_slots+=m['source_evaluation']['full_input_scalar_slots'];assert receipts[port]['runs'][name]['frozen_feature_integrity_passed']
            runs[(job['arm'],job['seed'])]=(m,curves)
            for item in m['records']:
                p=rr/item['path'];assert sha(p)==item['SHA256'];v=read(p);assert sha(p.with_suffix('.npz'))==v['score_SHA256'];k=(job['arm'],job['seed'],v['split'],v['mode'],v['background'],v['key']);assert k not in data
                data[k]=(v,p.with_suffix('.npz'))
    assert cost==88080384 and normalization_repeated_slots==429391872 and len(runs)==12 and len(data)==4608 and input_plans[0]==input_plans[1]
    reference=runs[('frozen_basis',348)][1]
    for (arm,seed),(m,curves) in runs.items():
        assert all((x['step'],x['key'],x['input_SHA256'],x['observed_input_exposure'])==(y['step'],y['key'],y['input_SHA256'],y['observed_input_exposure']) for x,y in zip(curves,reference))
        if arm!='wrong_graph_relation':assert all(x['graph_SHA256']==y['graph_SHA256'] for x,y in zip(curves,reference))
        assert m['initial_parameter_SHA256']==runs[('frozen_basis',seed)][0]['initial_parameter_SHA256']
    # Aggregate only the designated calibration split's normal views/backgrounds.
    for arm in ARMS:
        for seed in SEEDS:
            for mode in ['true','empty','rewired']:
                pieces=[];keys=[]
                for (aa,ss,split,mm,bg,key),(v,p) in data.items():
                    if (aa,ss,split,mm)==(arm,seed,'calibration',mode):
                        assert bg in ['plain','smooth']
                        with np.load(p) as f:assert f['scores'].shape==(2,704);pieces.append(f['scores'].reshape(-1))
                        keys.append(key)
                assert len(pieces)==32 and len(set(keys))==16;pool=np.concatenate(pieces);assert len(pool)==45056;theta=float(np.quantile(pool,.99));thresholds[(arm,seed,mode)]=theta;outcal[arm+'/'+str(seed)+'/'+mode]=dict(threshold=theta,systems=16,score_count=len(pool),pool_SHA256=hashlib.sha256(pool.tobytes()).hexdigest())
    cases=[]
    for (arm,seed,split,mode,bg,key),(v,p) in data.items():
        if split!='source_validation':continue
        with np.load(p) as f:scores=f['scores'];assert scores.shape==(8,704);fpr=float((scores[:2]>thresholds[(arm,seed,mode)]).mean())
        cases.append(dict(arm=arm,seed=seed,key=key,C=v['C'],family=v['family'],mode=mode,background=bg,AP=v['metrics']['matched']['AP'],physical_AP=v['metrics']['physical']['AP'],measurement_AP=v['metrics']['paired_measurement']['AP'],FPR=fpr,normal_NLL=v['normal_NLL'],normal_whitened_square=v['normal_whitened_square'],Top1=mean([e['Top1'] for e in v['RCA'] if e['kind']=='matched_measurement']),Top3=mean([e['Top3'] for e in v['RCA'] if e['kind']=='matched_measurement']),MRR=mean([e['MRR'] for e in v['RCA'] if e['kind']=='matched_measurement'])))
    metrics=['AP','physical_AP','measurement_AP','FPR','normal_NLL','normal_whitened_square','Top1','Top3','MRR'];grouped={}
    for row in cases:grouped.setdefault((row['arm'],row['mode'],row['background'],row['key']),[]).append(row)
    aggregates=[]
    for (arm,mode,bg,key),rows in grouped.items():
        assert len(rows)==3 and sorted(v['seed'] for v in rows)==SEEDS
        aggregates.append(dict(arm=arm,mode=mode,background=bg,key=key,C=rows[0]['C'],family=rows[0]['family'],**{m:mean([v[m] for v in rows]) for m in metrics}))
    summary={};seed_summary={};pairs={};lookup={(v['arm'],v['mode'],v['background'],v['key']):v for v in aggregates}
    for bg in ['plain','smooth','held_period']:
        summary[bg]={};seed_summary[bg]={}
        for mode in ['true','empty','rewired']:
            summary[bg][mode]={};seed_summary[bg][mode]={}
            for arm in ARMS:
                xs=[v for v in aggregates if (v['arm'],v['mode'],v['background'])==(arm,mode,bg)];assert len(xs)==32;summary[bg][mode][arm]={m:mean([v[m] for v in xs]) for m in metrics}
                seed_summary[bg][mode][arm]={str(seed):{m:mean([v[m] for v in cases if (v['arm'],v['mode'],v['background'],v['seed'])==(arm,mode,bg,seed)]) for m in metrics} for seed in SEEDS}
        for a1,a0 in [('normal_relation','normal_only'),('normal_only','frozen_basis'),('normal_relation','frozen_basis'),('normal_relation','wrong_graph_relation')]:
            xs=[v for v in aggregates if (v['arm'],v['mode'],v['background'])==(a1,'true',bg)];pairs[bg+'/'+a1+'-'+a0]={m:diff([v[m]-lookup[(a0,'true',bg,v['key'])][m] for v in xs]) for m in ['AP','FPR','Top1','normal_NLL']}
        for arm in ARMS:
            xs=[v for v in aggregates if (v['arm'],v['mode'],v['background'])==(arm,'true',bg)]
            for mm in ['empty','rewired']:pairs[bg+'/'+arm+'/true-'+mm]={m:diff([v[m]-lookup[(arm,mm,bg,v['key'])][m] for v in xs]) for m in ['AP','FPR','Top1']}
    result=dict(status='complete',training_input_exposure=cost,extra_source_evaluation_slots=normalization_repeated_slots,training_runs=12,source_calibration_systems=16,source_validation_systems=32,optimization_seeds=3,all_training_input_plans_equal=True,initialization_matched_within_seed=True,threshold_scope='source calibration only, 2normal x2background x16systems',summary=summary,seed_summary=seed_summary,paired=pairs,source_only=True,benchmark_arrays_opened=False,conditional_RCA=True)
    for name,value in [('analysis.json',result),('source_cases.json',aggregates),('per_seed_cases.json',cases),('calibration.json',outcal)]:
        p=r/name;assert not p.exists();p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(summary={bg:v['true'] for bg,v in summary.items()},paired={k:v for k,v in pairs.items() if 'normal_relation-normal_only' in k})))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args())
