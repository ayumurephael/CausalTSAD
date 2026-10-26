"""Read existing I09 caches; zero basis/fit/optimizer work. All arms predeclared."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,argparse,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from causal_tsad.io import sha256,atomic_json
from causal_tsad.frozen_detector import parameter_sha
from proposals.context_mixture import ContextSelector,aggregate
from proposals.mixture_evidence_audit import readouts,normal_target_nll,METHODS
from tools.run_i06_controls import metrics

def setup(plan,task):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(plan['worker_GiB']*2**30/torch.cuda.get_device_properties(0).total_memory)
 base=Path(plan['prior']);ident=task['run'];rp=base/'records'/(ident+'.json')
 assert sha256(rp)==task['run_record_SHA256'];rec=json.loads(rp.read_text())
 cp=base/'records'/(ident+'_checkpoint.pth');assert sha256(cp)==rec['checkpoint_SHA256']
 selector=ContextSelector(task['seed']).cuda().double().eval().requires_grad_(False)
 selector.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['selector']);before=parameter_sha(selector)
 assert before==rec['selector_final_SHA256']
 cm=base/'cache'/ident/'manifest.json';assert sha256(cm)==rec['cache_manifest_SHA256']
 cache=json.loads(cm.read_text());assert len(cache['files'])==1088
 weight=Path(plan['weight']);assert sha256(weight)==plan['weight_SHA256']
 state=torch.load(weight,map_location='cpu',weights_only=False)['model']
 temp=float(state['log_temperature'].exp().clamp(.1,10));bias=float(state['bias'])
 return selector,before,cache,rec,temp,bias

def load(item):
 p=Path(item['path']);assert sha256(p)==item['SHA256']
 with np.load(p) as z:return {k:torch.as_tensor(z[k],device='cuda') for k in z.files}

def preflight(plan,output):
 task=plan['tasks'][0];sel,_,cache,rec,temp,bias=setup(plan,task)
 item=next(v for v in cache['files'] if v['row'].get('split')=='source_validation')
 z=load(item);e,d=readouts(z,sel);errors={};base=Path(plan['prior'])/'records'
 for method,old,step in [('component0','full',0),('density_uniform','uniform',0),('density_learned','learned',512)]:
  with np.load(base/(task['run']+'_'+str(step)+'.npz')) as f:target=f[old+'__'+item['key']][item['view']]
  errors[method]=float(np.max(np.abs(aggregate(e[method],temp,bias).cpu().numpy()-target)))
 assert max(errors.values())<1e-8
 perm=torch.tensor([4,2,0,3,1],device='cuda');zp=dict(z)
 for k in ['mu','variance','valid']:zp[k]=z[k][...,perm]
 zp['features']=z['features'][...,perm,:];ep,_=readouts(zp,sel)
 pe=max(float((e[k]-ep[k]).abs().max()) for k in ['density_uniform','density_learned','mean_uniform','mean_learned'])
 assert pe<1e-8
 atomic_json(output,dict(status='passed',baseline_error=errors,mixture_identity_error=d['identity_error'],permutation_error=pe,
  cache_reads=2,optimizer_updates=0,new_basis_forwards=0,new_context_factorizations=0))

@torch.inference_mode()
def run(a):
 start=time.time();plan=json.loads(a.plan.read_text());task=plan['tasks'][a.task]
 sel,before,cache,rec,temp,bias=setup(plan,task)
 items=[v for v in cache['files'] if v['C']==task['C']];groups={}
 for v in items:groups.setdefault(v['key'],[]).append(v)
 results=[];saved={};budget=dict(cache_cases=0,cached_input_scalars_read=0,node_points=0,readout_node_outputs=0,normal_target_scalar_diagnostics=0,optimizer_updates=0,new_basis_forwards=0,new_context_factorizations=0)
 maxerr=0
 for key,group in sorted(groups.items()):
  if time.time()>plan['deadline_unix']:raise TimeoutError('I10 one-hour finite audit limit')
  group=sorted(group,key=lambda v:v['view']);row=group[0]['row'];istrain='split' not in row
  scores={k:[] for k in METHODS};channels={k:[] for k in METHODS};nll=[];diags=[]
  for item in group:
   z=load(item);e,d=readouts(z,sel);maxerr=max(maxerr,d['identity_error'])
   budget['cache_cases']+=1;budget['cached_input_scalars_read']+=1024*row['C'];budget['node_points']+=704*row['C'];budget['readout_node_outputs']+=11*704*row['C']
   diags.append(dict(view=item['view'],mean_weights=d['weights'].mean((0,1)).cpu().tolist(),
    normal_responsibility=d['normal_responsibility'].mean((0,1)).cpu().tolist(),variance_ratio_to_full=(z['variance']/z['variance'][...,0:1]).mean((0,1)).cpu().tolist(),
    standardized_squared_error=((z['response'][...,None]-z['mu']).square()/z['variance']).mean((0,1)).cpu().tolist()))
   if istrain:
    nll.append(dict(view=item['view'],**normal_target_nll(z,sel)));budget['normal_target_scalar_diagnostics']+=704*row['C']
   else:
    for method,values in e.items():
     scores[method].append(aggregate(values,temp,bias).cpu().numpy());channels[method].append(values.cpu().numpy())
  if istrain:results.append(dict(row=row,normal_target_nll=nll,diagnostics=diags));continue
  p=Path(plan['source_eval'])/row['path'];assert sha256(p)==row['SHA256']
  with np.load(p) as f:labels=f['labels'];graph=f['graph']
  met={};rca={};node=[]
  for method in METHODS:
   sc=np.stack(scores[method]);en=np.stack(channels[method]);saved[key+'__'+method]=sc
   if row['split']=='source_validation':
    met[method]=metrics(sc,labels,row);events=[];root=row['root']
    roles={'root':[root],'children':np.where(graph[root]>0)[0].tolist(),'others':[i for i in range(row['C']) if i!=root and graph[root,i]==0]}
    for view in [2,3,4,5,6,7,10,11,12,13,14,15]:
     v=view%8;lo,hi=(0,704) if v in [2,3] else (192,193) if v in [4,5] else (192,320)
     q=torch.as_tensor(en[view,lo:hi]);rank=int(torch.where(torch.argsort(torch.softmax(q,-1).mean(0),descending=True,stable=True)==root)[0])+1
     events.append(dict(view=view,Top1=int(rank==1),Top3=int(rank<=min(3,row['C'])),MRR=1/rank))
     normal=(view//8)*8+(view%2)
     for role,inds in roles.items():
      if inds:node.append(dict(method=method,view=view,role=role,nodes=len(inds),mean_e=float(en[view,lo:hi][:,inds].mean()),
       paired_normal_excess_e=float((en[view,lo:hi][:,inds]-en[normal,lo:hi][:,inds]).mean())))
    rca[method]=events
  results.append(dict(row=row,metrics=met,conditional_RCA=rca,node_roles=node,diagnostics=diags))
  atomic_json(a.root/'task_progress'/(task['id']+'.json'),dict(stage='cached_readout',systems=len(results),seconds=time.time()-start))
 assert parameter_sha(sel)==before
 prefix=a.root/'records'/task['id'];np.savez_compressed(str(prefix)+'.npz',**saved)
 atomic_json(str(prefix)+'.json',dict(status='complete',task=task,rows=results,budget=budget,selector_unchanged=True,selector_SHA256=before,
  source_record_SHA256=task['run_record_SHA256'],scores_SHA256=sha256(str(prefix)+'.npz'),G0_identity_max=maxerr,seconds=time.time()-start,
  code=plan['code'],benchmark_arrays_opened=False,normal_twin_input=False,TimeRCD_RCA=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--plan',type=Path,required=True);p.add_argument('--task',type=int);p.add_argument('--preflight',type=Path);a=p.parse_args()
 if a.preflight:preflight(json.loads(a.plan.read_text()),a.preflight)
 else:
  try:run(a)
  except Exception:atomic_json(a.root/'errors'/f'task{a.task}.json',dict(error=traceback.format_exc()));raise
