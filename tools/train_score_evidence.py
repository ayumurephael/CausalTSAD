"""Finite 2x2 directional constraint/ranking test; all teachers cached and frozen."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,argparse,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from torch.nn import functional as F
from proposals.score_evidence import EvidenceHead
from proposals.protected_fusion import parameter_sha
from fusion_source import load_arrays
from train_protected_fusion import evaluate
from causal_tsad.io import atomic_json,sha256
def run(a):
 cfg=json.loads(a.config.read_text());assert all(k in cfg for k in ['data_seed','learning_rate','pair_loss_weight']);m=json.loads((a.cache/'manifest.json').read_text());assert m['status']=='complete' and not m['target_arrays_opened'];assert not a.output.exists();a.output.mkdir(parents=True)
 torch.set_num_threads(1);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.75*2**30/torch.cuda.get_device_properties(0).total_memory)
 rows=m['rows'];train=[r for r in rows if r['split']=='train'];assert len(train)==568;arrays=[load_arrays(a.cache,r) for r in train]
 allf=np.concatenate([v['features'].reshape(-1,7) for v in arrays]);normal=np.concatenate([v['base'][1].ravel() for r,v in zip(train,arrays) if r['kind']=='old']);scale=max(1.,float((np.quantile(normal,.75)-np.quantile(normal,.25))/1.349));model=EvidenceHead(allf.mean(0,dtype=np.float64),allf.std(0,dtype=np.float64),scale).cuda();assert sum(p.numel() for p in model.parameters())==8
 status=dict(status='initial_source_validation',pid=os.getpid(),arm=a.arm,variant=a.variant,seed=a.seed,updates=0,optimized_ready_query_scalar_exposures=0,full_source_input_scalar_exposures=0,extra_head_query_scalar_exposures=0,pairwise_comparisons=0,
  local_commit=a.local_commit,github_commit=a.github_commit,config_SHA256=sha256(a.config),cache_manifest_SHA256=sha256(a.cache/'manifest.json'),source_plan_SHA256=m['plan_SHA256'],frozen_teacher_checkpoint_SHA256=m['teacher_checkpoint_SHA256'],initial_parameter_SHA256=parameter_sha(model),parameters=8,
  target_arrays_opened=False,TimeRCD_or_C04_parameters_updated=False,optimization_linear_forward_FLOPs=0,point_and_pair_loss_evaluated_all_arms=True)
 atomic_json(a.output/'progress.json',status);start=time.monotonic()
 def ev(name):
  result=evaluate(model,a.cache,rows,a.arm);atomic_json(a.output/(name+'.json'),result);status['extra_head_query_scalar_exposures']+=result['extra_head_query_scalar_exposures'];return result
 sample=arrays[0]
 with torch.no_grad():b=torch.from_numpy(sample['base']).cuda();assert torch.equal(model(b,torch.from_numpy(sample['features']).cuda(),a.arm)[0],b)
 ev('initial_source_validation');opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);positive=a.variant.startswith('positive');pair_weight=cfg['pair_loss_weight'] if a.variant.endswith('pair') else 0
 for j,index in enumerate(np.random.RandomState(cfg['data_seed']+a.seed-348).permutation(568)):
  assert time.monotonic()-start<3600 and time.time()<a.deadline;row=train[int(index)];v=arrays[int(index)];assert np.array_equal(v['base'],v['features'][...,0]);b=torch.from_numpy(v['base']).cuda();features=torch.from_numpy(v['features']).cuda();y=torch.from_numpy(v['labels']).float().cuda();opt.zero_grad(set_to_none=True);score,delta=model(b,features,a.arm);loss=F.binary_cross_entropy_with_logits(score,y)
  if row['kind']=='old':loss=loss+cfg['old_source_trust_weight']*(delta/scale).square().mean()
  else:
   assert score.shape==(4,704) and y[:2].count_nonzero()==0 and y[2:].sum()==1408
   pair=F.softplus(-(score[2:,None,:]-score[None,:2,:])).mean();loss=loss+pair_weight*pair;status['pairwise_comparisons']+=2816
  loss=loss*(568/(2*(312 if row['kind']=='old' else 256)));assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad);opt.step();model.project(positive)
  status['updates']+=1;status['optimized_ready_query_scalar_exposures']+=v['base'].size*row['channels'];status['full_source_input_scalar_exposures']+=v['base'].shape[0]*1024*row['channels'];status['optimization_linear_forward_FLOPs']+=14*v['base'].size;status.update(status='training',last_loss=float(loss),seconds=time.monotonic()-start)
  if j==0 or j%32==31:
   atomic_json(a.output/'progress.json',status)
   with (a.output/'learning_curve.jsonl').open('a') as f:f.write(json.dumps({k:status[k] for k in ['updates','last_loss','optimized_ready_query_scalar_exposures','seconds']})+'\n')
  if j==283:ev('mid_source_validation')
 assert status['updates']==568 and status['optimized_ready_query_scalar_exposures']==15561216 and status['full_source_input_scalar_exposures']==22634496 and status['pairwise_comparisons']==720896
 final=ev('final_source_validation');torch.save(dict(model=model.state_dict(),config=cfg,arm=a.arm,variant=a.variant,seed=a.seed,source_calibration=final,teacher_checkpoint_SHA256=m['teacher_checkpoint_SHA256']),a.output/'final.pth');status.update(status='complete',checkpoint_SHA256=sha256(a.output/'final.pth'),final_parameter_SHA256=parameter_sha(model),weights=model.linear.weight.detach().cpu().tolist(),bias=model.linear.bias.detach().cpu().tolist(),seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated());atomic_json(a.output/'progress.json',status)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['cache','config','output']:p.add_argument('--'+n,type=Path,required=True)
 for n in ['arm','variant','local-commit','github-commit']:p.add_argument('--'+n,required=True)
 p.add_argument('--seed',type=int,required=True);p.add_argument('--deadline',type=float,default=float('inf'));a=p.parse_args()
 try:run(a)
 except Exception:
  a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
