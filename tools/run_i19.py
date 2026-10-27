"""Source cache-only I19; never loads training_target or a target benchmark."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from sklearn.metrics import average_precision_score
from causal_tsad.io import atomic_json,sha256
from tools.run_improvement3 import cache_rows,WHITELIST,save_npz
from proposals.calibrated_evidence import statistics,calibrate,scores,roots,alarm_mask,tail
from proposals.improvement3.metrics import evaluate,spans

def load_statistics(root,item):
    p=root/item['path'];assert sha256(p)==item['SHA256']
    with np.load(p) as f:z={k:torch.as_tensor(f[k],device='cuda') for k in WHITELIST}
    return statistics(z)

def main(a):
    start=time.time();plan=json.loads(a.plan.read_text());root=Path(plan['source_root'])
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(0).total_memory)
    if a.task<0:
        torch.manual_seed(348);r=torch.randn(256,4,device='cuda',dtype=torch.float64)
        z=dict(response=r,mu=torch.zeros(256,4,1,device='cuda',dtype=torch.float64),variance=torch.ones(256,4,1,device='cuda',dtype=torch.float64),
               edges=torch.tensor([[0,1],[0,2],[2,3]],device='cuda'),contribution=torch.randn(256,3,1,device='cuda',dtype=torch.float64),raw_logit=r[:,0])
        s=statistics(z);cal=calibrate([s]);d=scores(s,cal);rr=roots(s,cal)
        assert all(np.isfinite(x).all() for x in [*d.values(),*rr.values()])
        assert abs((s['edge'][32]-cal['moments']['32']['edge_mean']).mean())<1e-12
        y=np.arange(256)%3==0
        assert abs(average_precision_score(y,d['node_edge'])-average_precision_score(y,d['node_edge_bonf_shift']))<1e-12
        assert np.array_equal(alarm_mask(np.array([0.,3.,1.5,0.,0.,3.,0.]),2.,1.),np.array([0,1,1,1,1,1,0],bool))
        atomic_json(a.output,dict(status='passed',calibration_center_exact=True,constant_bonferroni_AP_invariant=True,
                                  all_outputs_finite=True,hysteresis_gap_check=True,technical_input_scalars=1024));return
    assert sha256(root/'job_plan.json')==plan['source_plan_SHA256']
    records=cache_rows(root,json.loads((root/'job_plan.json').read_text()),plan['profile'])
    task=plan['tasks'][a.task];C=task['C'];normal=[];read=[];exposure=0
    for rec in records:
        row=rec['task']['row']
        if row['split']!='calibration' or row['C']!=C:continue
        for item in rec['files']:
            if item['view']%8>=2:continue
            normal.append(load_statistics(root,item));read.append(item);exposure+=item['input_scalars']
    assert len(normal)==16
    cal=calibrate(normal);ns=[scores(s,cal) for s in normal]
    thresholds={k:[float(np.quantile(np.concatenate([x[k] for x in ns]),q)) for q in [.99,.95]] for k in ns[0]}
    # Calibration sanity does not assert a nominal dependent-sequence error rate.
    assert tail(np.array([0.,1.,2.]),np.array([2.]))[0]==.5
    out=a.output;out.mkdir(parents=True,exist_ok=False);metrics=[];arrays={};shift_error=0.
    chosen=[r for r in records if r['task']['row']['split']=='confirmation' and r['task']['row']['C']==C]
    assert len(chosen)==8
    for rec in chosen:
        row=rec['task']['row']
        for item in rec['files']:
            s=load_statistics(root,item);exposure+=item['input_scalars'];read.append(item)
            ds=scores(s,cal);rr=roots(s,cal);vi=item['view'];key=row['key']+'__'+str(vi)
            # Labels/root enter metrics only after all predictions are fixed.
            with np.load(root/item['path']) as f:y=f['labels'].copy()
            meta=dict(system=row['key'],C=C,family=row['family'],view=vi,fault=row['fault_names'][vi%8],background='plain' if vi<8 else 'smooth')
            for name,score in ds.items():
                hi,lo=thresholds[name];arrays[key+'__'+name]=score
                base=evaluate(score,y,hi)
                mask=alarm_mask(score,hi,lo)
                alarm=evaluate(mask.astype(float),y,.5)
                event_fields={k:v for k,v in alarm.items() if k in ['event_recall','unmatched_alarm_events','false_alarm_events','events','FPR']}
                event_fields['alarm_segments']=len(spans(mask))
                metrics.append(dict(**meta,method=name,**base,hysteresis=event_fields,
                    score_unique_fraction=float(len(np.unique(score))/len(score))))
                if name.endswith('_bonf_shift') and y.any():
                    shift_error=max(shift_error,abs(average_precision_score(y,score)-average_precision_score(y,ds[name.removesuffix('_bonf_shift')])))
            # The detection score is identical across RCA readout alternatives.
            score=ds['node_edge'];hi,lo=thresholds['node_edge'];hm=alarm_mask(score,hi,lo)
            for name,node in rr.items():
                arrays[key+'__root_'+name]=node
                raw=evaluate(score,y,hi,node,row['root']);h=evaluate(hm.astype(float),y,.5,node,row['root'])
                metrics.append(dict(**meta,method='root_'+name,**raw,hysteresis={k:v for k,v in h.items() if k in ['events','joint_detected_root_recall','event_recall','unmatched_alarm_events','FPR']}))
    assert shift_error<1e-12
    save_npz(out/'scores.npz',**arrays)
    save_npz(out/'calibration.npz',**cal['refs'])
    atomic_json(out/'result.json',dict(status='complete',task=task,profile=plan['profile'],metrics=metrics,
      thresholds=thresholds,null_moments=cal['moments'],calibration_systems=4,confirmation_systems=8,
      calibration_points_dependent=True,nominal_FPR_guarantee=False,cached_input_scalars=exposure,
      source_files=[dict(path=x['path'],SHA256=x['SHA256']) for x in read],scores_SHA256=sha256(out/'scores.npz'),
      calibration_SHA256=sha256(out/'calibration.npz'),bonferroni_shift_AP_error=shift_error,
      optimizer_steps=0,new_basis_forwards=0,new_TimeRCD_forwards=0,benchmark_arrays_read=False,
      plan_SHA256=sha256(a.plan),seconds=time.time()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--task',type=int,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
