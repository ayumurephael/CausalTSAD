"""System-paired posthoc I19 summary; never chooses a deployment threshold."""
import json,argparse
from pathlib import Path
import numpy as np

def paired(a,b):
    keys=sorted(a.keys()&b.keys());d=np.array([a[k]-b[k] for k in keys]);rng=np.random.default_rng(348)
    bs=np.mean(rng.choice(d,(10000,len(d)),replace=True),axis=1)
    return dict(mean_pp=float(d.mean()*100),CI95_pp=(np.quantile(bs,[.025,.975])*100).tolist(),
                up=int((d>1e-10).sum()),down=int((d< -1e-10).sum()),all_system_differences=dict(zip(keys,d.tolist())))

def main(a):
    out={};budget=0
    for port in ['2226','2229']:
        recs=[json.loads(p.read_text()) for p in (a.root/port/'records').rglob('result.json')];assert len(recs)==4
        assert all(r['bonferroni_shift_AP_error']<1e-12 for r in recs)
        rows=sum([r['metrics'] for r in recs],[]);budget+=sum(r['cached_input_scalars'] for r in recs)
        methods=sorted({r['method'] for r in rows});summary={};by={}
        for method in methods:
            subset=[r for r in rows if r['method']==method];summary[method]={};by[method]={}
            for metric in ['AP','PR_AUC','conditional_Top1','joint_detected_root_recall','FPR','false_alarm_events','hyst_FPR','hyst_alarm_segments','hyst_joint_detected_root_recall']:
                vals={}
                for r in subset:
                    if metric=='hyst_alarm_segments' and r['positive_points']!=0:continue
                    v=r.get('hysteresis',{}).get(metric[5:]) if metric.startswith('hyst_') else r.get(metric)
                    if v is not None:vals.setdefault(r['system'],[]).append(v)
                if vals:
                    by[method][metric]={k:float(np.mean(v)) for k,v in vals.items()}
                    summary[method][metric]=float(np.mean(list(by[method][metric].values())))
        pairs={}
        for x,y in [('node_edge','node'),('node_edge_raw','raw'),('node_edge_bonf_clip','node_edge'),
                    ('root_node_centered','root_node'),('root_node_centered','root_node_sham'),('root_edge_source_center','root_edge_raw'),('root_edge_proxy_median','root_edge_raw')]:
            for metric in ['AP','conditional_Top1','joint_detected_root_recall','FPR']:
                if metric in by[x] and metric in by[y]:pairs[x+' minus '+y+' '+metric]=paired(by[x][metric],by[y][metric])
        strata={}
        for field in ['background','fault','C']:
            strata[field]={}
            for value in sorted({str(r[field]) for r in rows}):
                strata[field][value]={}
                for method in ['node','node_edge','node_edge_raw','raw','root_node','root_node_centered','root_node_sham']:
                    r0=[r for r in rows if str(r[field])==value and r['method']==method]
                    metrics={}
                    for metric in ['AP','conditional_Top1','joint_detected_root_recall','FPR']:
                        d={}
                        for r in r0:
                            if r.get(metric) is not None:d.setdefault(r['system'],[]).append(r[metric])
                        if d:metrics[metric]=float(np.mean([np.mean(v) for v in d.values()]))
                    strata[field][value][method]=metrics
        out[port]=dict(profile=recs[0]['profile'],systems=32,summary=summary,paired=pairs,strata=strata)
    result=dict(status='complete',nodes=out,cached_input_scalars=budget,training_exposure=0,new_basis_forwards=0,new_TimeRCD_forwards=0)
    (a.root/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({p:v['summary'] for p,v in out.items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args())
