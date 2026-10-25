"""Fixed, root-blind missingness counterexample; no fitting or benchmark arrays."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from proposals.bayes_propagation import BayesianPropagation,generate,availability_mask,distances
from causal_tsad.io import atomic_json,sha256,array_sha
VARIANTS=['independent','instant','physical_only','dag_delay_mixture','reverse','wrong','oracle_lag']
def run(a):
 assert not a.output.exists();a.output.mkdir(parents=True);torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(1.5*2**30/torch.cuda.get_device_properties(0).total_memory);params=json.loads(a.parameters.read_text());assert params['fit_seed']==a.seed;atomic_json(a.output/'source_fitted_parameters.json',params);m=BayesianPropagation(params);records=[];slots=observed=0;start=time.monotonic()
 for i in range(a.worker,128,a.workers):
  assert time.monotonic()-start<2*3600;r=generate(i,'evaluation');c=len(r['graph']);mask=availability_mask(i,c);methods={}
  for variant in VARIANTS:
   scores=[];root_arrays=[];rca=[];forecasts=[];mode=[]
   for k,x in enumerate(r['cases']):
    out=m.infer(x,r['graph'],variant,oracle_lag=r['lag'] if variant=='oracle_lag' else None,forecast=variant in ['independent','dag_delay_mixture'],observed_mask=mask);scores.append(out['score']);root_arrays.append(out['root']);slots+=x.size;observed+=int(mask.sum());assert np.isfinite(out['score']).all()
    if k>=4:
     rank=np.argsort(-out['root'][r['onset']:].mean(0),kind='stable');rr=int(np.flatnonzero(rank==r['root'])[0])+1;rca.append(dict(mode=r['modes'][k],Top1=int(rr==1),Top3=int(rr<=3),MRR=1/rr));mode.append(float(out['physical_given_fault'][r['onset']:].mean()))
    if out['forecast8'] is not None:
     truth=(np.zeros_like(x,dtype=bool) if k<4 else r['oracle_activation_only'][k-4])[8:];forecasts.append(dict(mode=r['modes'][k],Brier_oracle_latent_future8=float(np.mean((out['forecast8'][:-8]-truth)**2))))
   s=np.stack(scores);y=r['labels'];p=a.output/'scores'/(str(i)+'_'+variant+'.npz');p.parent.mkdir(exist_ok=True);np.savez_compressed(p,score=s,root=np.stack(root_arrays));methods[variant]=dict(group_AP=float(average_precision_score(y.ravel(),s.ravel())),group_ROC_AUC=float(roc_auc_score(y.ravel(),s.ravel())),normal_FPR_posterior_0_5=float(np.mean(s[:4]>.5)),measurement_AP=float(average_precision_score(y[:6].ravel(),s[:6].ravel())),physical_AP=float(average_precision_score(y[[0,1,2,3,6,7]].ravel(),s[[0,1,2,3,6,7]].ravel())),RCA=rca,mean_physical_given_fault_by_case=mode,forecast=forecasts,scores_SHA256=sha256(p))
  dist=distances(r['graph'])[r['root']];records.append(dict(index=i,seed=r['seed'],C=c,root_observed=bool(mask[0,r['root']]),observed_descendants=int(((dist>0)&(dist<=c)&mask[0]).sum()),inputs_SHA256=dict(cases=array_sha(r['cases']),graph=array_sha(r['graph']),availability=array_sha(mask),labels_audit=array_sha(r['labels'])),methods=methods,mask_scope='halfchannels per system using independent RNG namespace; not chosen from root/graph/labels'))
  atomic_json(a.output/'progress.json',dict(status='running',systems=len(records),full_input_scalar_slots=slots,observed_evidence_scalar_forwards=observed,hypothesis_time_cells=m.hypothesis_time_cells,optimizer_updates=0,benchmark_arrays_opened=False))
 atomic_json(a.output/'results.json',dict(status='complete',seed=a.seed,source_parameters_SHA256=sha256(a.parameters),records=records,full_input_scalar_slots=slots,observed_evidence_scalar_forwards=observed,hypothesis_time_cells=m.hypothesis_time_cells,optimizer_updates=0,benchmark_arrays_opened=False));atomic_json(a.output/'progress.json',dict(status='complete',systems=len(records),full_input_scalar_slots=slots,observed_evidence_scalar_forwards=observed,hypothesis_time_cells=m.hypothesis_time_cells,optimizer_updates=0,benchmark_arrays_opened=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parameters',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,required=True);a=p.parse_args()
 try:run(a)
 except Exception:
  a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
