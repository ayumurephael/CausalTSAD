"""Independent source toy proof: no C04/TimeRCD model or benchmark arrays."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from proposals.bayes_propagation import BayesianPropagation,generate,fit_source,T
from causal_tsad.io import atomic_json,sha256
VARIANTS=['independent','instant','physical_only','dag_delay_mixture','reverse','wrong','oracle_lag']
def run(a):
 assert not a.output.exists();a.output.mkdir(parents=True);torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(1.5*2**30/torch.cuda.get_device_properties(0).total_memory);p=json.loads(a.parameters.read_text());assert p['fit_seed']==a.seed;atomic_json(a.output/'source_fitted_parameters.json',p);m=BayesianPropagation(p);records=[];actual=0;start=time.monotonic()
 for i in range(a.worker,128,a.workers):
  assert time.monotonic()-start<2*3600;r=generate(i,'evaluation');c=r['graph'].shape[0];methods={}
  for variant in VARIANTS:
   scores=[];roots=[];mode=[];normal=[];forecast=[]
   for k,x in enumerate(r['cases']):
    out=m.infer(x,r['graph'],variant,oracle_lag=r['lag'] if variant=='oracle_lag' else None,forecast=(variant in ['independent','dag_delay_mixture'] and k in [4,6]));s=out['score'];assert np.isfinite(s).all();scores.append(s);actual+=x.size
    if k<4:normal.extend(s.tolist())
    else:
     root=out['root'][r['onset']:].mean(0);rank=np.argsort(-root,kind='stable');rr=int(np.flatnonzero(rank==r['root'])[0])+1;roots.append(dict(mode=r['modes'][k],Top1=int(rr==1),Top3=int(rr<=3),MRR=1/rr));mode.append(float(out['physical_given_fault'][r['onset']:].mean()))
    if out['forecast8'] is not None:
     truth=r['oracle_activation_only'][0 if k==4 else 2][8:];prediction=out['forecast8'][:-8];forecast.append(dict(mode=r['modes'][k],Brier_oracle_latent_future8=float(np.mean((prediction-truth)**2)),scope='oracle latent simulation audit only, never model input or new threshold labels'))
   score=np.stack(scores);labels=r['labels'];methods[variant]=dict(group_AP=float(average_precision_score(labels.ravel(),score.ravel())),group_ROC_AUC=float(roc_auc_score(labels.ravel(),score.ravel())),normal_FPR_posterior_0_5=float(np.mean(np.asarray(normal)>.5)),RCA=roots,mean_physical_given_fault_by_case=mode,forecast=forecast,
    measurement_AP=float(average_precision_score(labels[:6].ravel(),score[:6].ravel())),physical_AP=float(average_precision_score(labels[[0,1,2,3,6,7]].ravel(),score[[0,1,2,3,6,7]].ravel())))
   path=a.output/'scores'/(str(i)+'_'+variant+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,score=score);methods[variant]['scores_SHA256']=sha256(path)
  records.append(dict(index=i,system_seed=r['seed'],C=c,lag_oracle_audit=r['lag'],strength_audit=r['scale'],root_has_children=bool(r['graph'][r['root']].any()),methods=methods,matched_root_observations=True));atomic_json(a.output/'progress.json',dict(status='running',systems=len(records),hypothesis_time_cells=m.hypothesis_time_cells,repeated_input_scalar_forwards=actual,optimizer_updates=0,benchmark_arrays_opened=False))
 atomic_json(a.output/'results.json',dict(status='complete',seed=a.seed,records=records,source_parameters_SHA256=sha256(a.output/'source_fitted_parameters.json'),hypothesis_time_cells=m.hypothesis_time_cells,repeated_input_scalar_forwards=actual,benchmark_arrays_opened=False,optimizer_updates=0,scope='evidence-level controlled feasibility; not rawTSAD foundation or retained250Mbenchmark'))
 atomic_json(a.output/'progress.json',dict(status='complete',systems=len(records),hypothesis_time_cells=m.hypothesis_time_cells,repeated_input_scalar_forwards=actual,optimizer_updates=0,benchmark_arrays_opened=False,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--parameters',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,required=True);a=p.parse_args()
 try:run(a)
 except Exception:
  a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
