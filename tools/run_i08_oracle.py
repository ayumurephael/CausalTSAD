"""Fixed I06 queries; source-only oracle context/feature factorial diagnosis."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scipy.special import logsumexp,softmax
from causal_tsad.io import atomic_json,sha256,array_sha
from causal_tsad.frozen_detector import parameter_sha
from proposals.causal_meta_basis import CausalMetaBasis
from proposals.context_crossfit import configurations
from proposals.context_feature_oracle import diagnose,VARIANTS
from tools.train_meta_basis import graph_for
from tools.run_i06_controls import metrics

def score(r,v,temp,bias):
 e=.5*((2/3)*r*r/v-np.log(3))
 return temp*(logsumexp(e/temp,axis=-1)-np.log(r.shape[-1]))+bias,e

def run(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 plan=json.loads(a.plan.read_text());task=plan['tasks'][a.task];row=task['row'];seed=task['seed'];ident=task['id'];start=time.time()
 torch.cuda.set_per_process_memory_fraction(plan['worker_GiB']*2**30/torch.cuda.get_device_properties(0).total_memory)
 dest=a.root/'records'/ident;assert not dest.with_suffix('.json').exists()
 source=Path(plan['source'])/row['path'];assert sha256(source)==row['SHA256']
 with np.load(source) as z:values=z['values'];labels=z['labels'];g=z['graph']
 assert array_sha(values)==row['input_SHA256'] and array_sha(labels)==row['labels_SHA256']
 prior=Path(plan['prior_root']);oldpath=prior/'records'/(ident+'.json');old=json.loads(oldpath.read_text())
 assert sha256(oldpath)==task['prior_record_SHA256'] and old['task']==task['prior_task']
 oldcache=prior/'context_cache'/(ident+'.npz');assert sha256(oldcache)==old['context_cache_SHA256']
 with np.load(oldcache) as z:cache={k:z[k] for k in z.files}
 weight=Path(plan['weights'])/f'normal_only_s{seed}.pth';assert sha256(weight)==plan['weight_SHA256'][str(seed)]
 model=CausalMetaBasis(seed).cuda().eval().requires_grad_(False)
 model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False)['model']);initial=parameter_sha(model)
 temp=float(model.log_temperature.exp().clamp(.1,10));bias=float(model.bias)
 scores={};nodes={};rca={};audit=[];budget={};g0=[]
 for mode in ['true','empty','wrong']:
  graph=torch.as_tensor(graph_for(g,row['key'],mode),dtype=torch.float64,device='cuda')
  arrays={v:{k:np.array(cache[mode+'__'+k][:,288:],copy=True) for k in ['residual','variance']} for v in VARIANTS}
  for view in [2,3,4,5,6,7,10,11,12,13,14,15]:
   atomic_json(a.root/'task_progress'/(ident+'.json'),dict(stage=mode+'_view'+str(view),seconds=time.time()-start))
   b=8 if view>=8 else 0;paired=b+(view%2)
   outs,bud=diagnose(model,torch.as_tensor(values[view],dtype=torch.float64,device='cuda'),torch.as_tensor(values[paired],dtype=torch.float64,device='cuda'),graph,configurations()['xfit'])
   for k,v in bud.items():
    if isinstance(v,(int,float)) and not isinstance(v,bool):budget[k]=budget.get(k,0)+v
   errors={}
   for name,o in outs.items():
    for k,v in o.items():arrays[name][k][view]=v.cpu().numpy()
   for k in ['residual','variance']:
    errors[k]=float(np.max(np.abs(arrays['OO'][k][view]-cache[mode+'__'+k][view,288:])))
   assert max(errors.values())<1e-8,errors
   g0.append(dict(mode=mode,view=view,**errors))
  for name,arr in arrays.items():
   for scaling in ['nativevar','OOvar']:
    tag=mode+'__'+name+'__'+scaling
    variance=arr['variance'] if scaling=='nativevar' else arrays['OO']['variance']
    s,e=score(arr['residual'],variance,temp,bias);scores[tag]=s
    rr=[]
    for b in [0,8]:
     for vi,lo,hi in [(2,320,1024),(3,320,1024),(4,512,513),(5,512,513),(6,512,640),(7,512,640)]:
      p=softmax(e[b+vi,lo-320:hi-320],axis=-1).mean(0)
      rank=int(np.flatnonzero(np.argsort(-p,kind='stable')==row['root'])[0])+1
      rr.append(dict(view=b+vi,Top1=int(rank==1),Top3=int(rank<=3),MRR=1/rank))
    rca[tag]=rr
   for k,v in arr.items():nodes[mode+'__'+name+'__'+k]=v
  # Dataset-level G0: all oracle operations are null on unchanged normal views.
  for name in VARIANTS:
   assert np.array_equal(scores[mode+'__'+name+'__nativevar'][[0,1,8,9]],scores[mode+'__OO__nativevar'][[0,1,8,9]])
 assert parameter_sha(model)==initial
 np.savez_compressed(dest.with_suffix('.npz'),**scores)
 np.savez_compressed(a.root/'node_cache'/(ident+'.npz'),**nodes)
 atomic_json(dest.with_suffix('.json'),dict(task=task,input_SHA256=array_sha(values),labels_SHA256=array_sha(labels),
  graph_SHA256=array_sha(g),basis_weight_SHA256=sha256(weight),basis_parameter_SHA256=initial,parameters_unchanged=True,
  prior_record_SHA256=sha256(oldpath),prior_context_SHA256=sha256(oldcache),G0=g0,
  scores_SHA256=sha256(dest.with_suffix('.npz')),node_cache_SHA256=sha256(a.root/'node_cache'/(ident+'.npz')),
  metrics={k:metrics(v,labels,row) for k,v in scores.items()},conditional_causal_RCA=rca,budget=budget,
  technical_note='OO refit is necessary: I06 did not save regression coefficients; baseline r/v equivalence audited for every faulty view. Normal-view predictions are reused.',
  seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),code=plan['code'],
  oracle_only=True,normal_twin_as_target_input=False,benchmark_arrays_opened=False,TimeRCD_RCA=False))
 atomic_json(a.root/'task_progress'/(ident+'.json'),dict(stage='complete',seconds=time.time()-start))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--task',type=int,required=True);a=p.parse_args()
 try:run(a)
 except Exception:atomic_json(a.root/'errors'/f'task{a.task}.json',dict(error=traceback.format_exc()));raise
