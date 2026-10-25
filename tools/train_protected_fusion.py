"""One fixed source pass; only434 fusion parameters update, all teachers stay frozen."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from torch.nn import functional as F
from proposals.protected_fusion import ProtectedFusion,parameter_sha
from causal_tsad.io import atomic_json,sha256
from fusion_source import load_arrays

def evaluate(model,cache,rows,arm):
 from sklearn.metrics import average_precision_score,roc_auc_score
 model.eval();norm=[];base_norm=[];records={};count=0
 def infer(row):
  a=load_arrays(cache,row);base=torch.from_numpy(a['base']).cuda();features=torch.from_numpy(a['features']).cuda()
  with torch.no_grad():s,delta=model(base,features,arm)
  return a,s.cpu().numpy(),delta.cpu().numpy()
 for row in rows:
  if row['split'] not in ['val_old','cal_new']:continue
  a,s,_=infer(row);indices=[1] if row['kind']=='old' else [0,1];norm.extend(s[indices].ravel());base_norm.extend(a['base'][indices].ravel());count+=len(indices)*704*row['channels']
 threshold=float(np.quantile(norm,.99));base_threshold=float(np.quantile(base_norm,.99))
 for row in rows:
  if row['split'] not in ['val_old','val_new']:continue
  a,s,delta=infer(row);y=a['labels'];root=[]
  if arm!='basecal':
   for view,r in enumerate(a['root']):
    mask=a['root_window'][view].astype(bool)
    if r>=0 and mask.any():root.append(int(a['root_logits'][view,mask].mean(0).argmax())==int(r))
  if row['kind']=='old':
   valid=len(np.unique(y[0]))>1;ap=float(average_precision_score(y[0],s[0])) if valid else None;bap=float(average_precision_score(y[0],a['base'][0])) if valid else None;ns=s[1];nb=a['base'][1]
  else:
   ap=float(average_precision_score(y.ravel(),s.ravel()));bap=float(average_precision_score(y.ravel(),a['base'].ravel()));ns=s[:2].ravel();nb=a['base'][:2].ravel()
  records[row['key']]=dict(kind=row['kind'],channels=row['channels'],AP_ready=ap,base_AP_ready=bap,normal_FPR=float(np.mean(ns>threshold)),base_normal_FPR=float(np.mean(nb>base_threshold)),
   delta_abs_mean=float(np.abs(delta).mean()),conditional_ready_root_Top1=float(np.mean(root)) if root else None,
   normal_FPR_scope='calibration sample for old source, independent for new validation')
  count+=s.size*row['channels']
 return dict(records=records,source_threshold=threshold,base_threshold=base_threshold,extra_head_query_scalar_exposures=count,target_arrays_opened=False,
  scope='Source only; old sparse ready AP and new50%positive paired AP remain separate; no original TimeRCD RCA')

def run(a):
 cfg=json.loads(a.config.read_text());assert all(k in cfg for k in ['data_seed','learning_rate','weight_decay','old_source_trust_weight']);m=json.loads((a.cache/'manifest.json').read_text());assert m['status']=='complete';assert not a.output.exists();a.output.mkdir(parents=True)
 rows=m['rows'];train=[r for r in rows if r['split']=='train'];assert len(train)==568
 torch.set_num_threads(1);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.75*2**30/torch.cuda.get_device_properties(0).total_memory)
 arrays=[load_arrays(a.cache,r) for r in train];allf=np.concatenate([v['features'].reshape(-1,7) for v in arrays]);mean=allf.mean(0,dtype=np.float64);std=allf.std(0,dtype=np.float64)
 oldnormal=np.concatenate([v['base'][1].ravel() for r,v in zip(train,arrays) if r['kind']=='old']);base_scale=max(1.,float((np.quantile(oldnormal,.75)-np.quantile(oldnormal,.25))/1.349))
 model=ProtectedFusion(mean,std,base_scale).cuda();initial=parameter_sha(model);assert sum(p.numel() for p in model.parameters())==434
 status=dict(status='initial_source_validation',pid=os.getpid(),arm=a.arm,seed=a.seed,local_commit=a.local_commit,github_commit=a.github_commit,
  cache_manifest_SHA256=sha256(a.cache/'manifest.json'),frozen_teacher_checkpoint_SHA256=m['teacher_checkpoint_SHA256'],source_plan_SHA256=m['plan_SHA256'],
  base_checkpoint_SHA256=cfg['base_checkpoint_SHA256'],initial_parameter_SHA256=initial,parameters=434,base_scale=base_scale,
  updates=0,optimized_ready_query_scalar_exposures=0,full_source_input_scalar_exposures=0,extra_head_query_scalar_exposures=0,
  TimeRCD_or_C04_parameters_updated=False,target_arrays_opened=False,normalization_from_source_train_only=True,nominal_capacity_matched=True)
 atomic_json(a.output/'progress.json',status);start=time.monotonic()
 def ev(name):
  result=evaluate(model,a.cache,rows,a.arm);atomic_json(a.output/(name+'.json'),result);status['extra_head_query_scalar_exposures']+=result['extra_head_query_scalar_exposures'];return result
 sample=arrays[0];b=torch.from_numpy(sample['base']).cuda();f=torch.from_numpy(sample['features']).cuda()
 with torch.no_grad():assert torch.equal(model(b,f,a.arm)[0],b)
 ev('initial_source_validation');opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
 order=np.random.RandomState(cfg['data_seed']).permutation(568)
 for j,index in enumerate(order):
  assert time.monotonic()-start<3600 and time.time()<a.deadline
  row=train[int(index)];v=arrays[int(index)];assert np.array_equal(v['base'],v['features'][...,0])
  b=torch.from_numpy(v['base']).cuda();f=torch.from_numpy(v['features']).cuda();y=torch.from_numpy(v['labels']).float().cuda();model.train();opt.zero_grad(set_to_none=True);s,delta=model(b,f,a.arm)
  loss=F.binary_cross_entropy_with_logits(s,y)
  if row['kind']=='old':loss=loss+cfg['old_source_trust_weight']*(delta/base_scale).square().mean()
  # Equal total old/new group weight per pass, not equal independent case count.
  loss=loss*(568/(2*(312 if row['kind']=='old' else 256)));assert torch.isfinite(loss)
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad);opt.step()
  status['updates']+=1;status['optimized_ready_query_scalar_exposures']+=v['base'].size*row['channels'];status['full_source_input_scalar_exposures']+=v['base'].shape[0]*1024*row['channels'];status.update(status='training',last_loss=float(loss),seconds=time.monotonic()-start)
  if status['updates']==1 or status['updates']%32==0:
   atomic_json(a.output/'progress.json',status)
   with (a.output/'learning_curve.jsonl').open('a') as out:out.write(json.dumps({k:status[k] for k in ['updates','optimized_ready_query_scalar_exposures','last_loss','seconds']})+'\n')
  if j==283:ev('mid_source_validation')
 assert status['updates']==568 and status['optimized_ready_query_scalar_exposures']==15561216 and status['full_source_input_scalar_exposures']==22634496
 final=ev('final_source_validation');torch.save(dict(model=model.state_dict(),config=cfg,arm=a.arm,seed=a.seed,source_calibration=final,cache_manifest_SHA256=sha256(a.cache/'manifest.json'),teacher_checkpoint_SHA256=m['teacher_checkpoint_SHA256']),a.output/'final.pth')
 status.update(status='complete',seconds=time.monotonic()-start,final_parameter_SHA256=parameter_sha(model),checkpoint_SHA256=sha256(a.output/'final.pth'),C1_and_warmup_policy='exact baseline identity by full_scores',peak_cuda_bytes=torch.cuda.max_memory_allocated());atomic_json(a.output/'progress.json',status)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['cache','config','output']:p.add_argument('--'+n,type=Path,required=True)
 for n in ['local-commit','github-commit']:p.add_argument('--'+n,required=True)
 p.add_argument('--arm',choices=['basecal','degree','true','wrong'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--deadline',type=float,default=float('inf'));a=p.parse_args()
 try:run(a)
 except Exception:
  a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
