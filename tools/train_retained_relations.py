"""Fixed source-only clean/augmented continuation, unprotected vs actual-step protection."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from torch.nn import functional as F
from proposals.causal_contrast import CausalContrast
from proposals.benign_nuisance import transform
from proposals.retention_step import vector,protect
from proposals.raw_context import parameter_sha
from tools.train_causal_contrast import load_pool,graph_for,evaluate
from causal_tsad.io import atomic_json,sha256,array_sha
def run(a):
 assert not a.output.exists();a.output.mkdir(parents=True);torch.set_num_threads(1);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory);start=time.monotonic();saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert saved['seed']==a.seed and saved['mode']==a.mode;cfg=saved['config'];m=CausalContrast().cuda();m.load_state_dict(saved['model'],strict=True);data=load_pool(a.source,saved['source_manifest_SHA256']);params=list(m.parameters());blocks={'shared':[p for n,p in m.named_parameters() if not n.startswith('root.')],'root':[p for n,p in m.named_parameters() if n.startswith('root.')]};opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=cfg['weight_decay']);rng=np.random.RandomState(a.seed);counts={'linear':0}
 def hook(layer,inp,out):counts['linear']+=2*layer.in_features*out.numel()
 for layer in m.modules():
  if isinstance(layer,torch.nn.Linear):layer.register_forward_hook(hook)
 status=dict(status='training',arm=a.arm,mode=a.mode,seed=a.seed,pid=os.getpid(),local_commit=a.local_commit,github_commit=a.github_commit,initial_checkpoint_SHA256=sha256(a.checkpoint),initial_parameter_SHA256=parameter_sha(m),source_manifest_SHA256=saved['source_manifest_SHA256'],updates=0,optimized_observed_query_scalars=0,full_observed_input_scalar_exposures=0,extra_frozen_query_predictions=0,optimization_Linear_forward_FLOPs=0,optimization_attention_forward_FLOPs=0,optimization_graph_aggregate_forward_FLOPs=0,projection_vector_operations=0,target_arrays_opened=False,original_TimeRCD_updated=False,extra_teacher_forwards=0)
 atomic_json(a.output/'progress.json',status)
 for index in rng.permutation(256):
  assert time.monotonic()-start<3*3600 and time.time()<a.deadline;row=data['train'][int(index)];strength=0. if rng.rand()<.25 else float(rng.uniform(0,8));aug=transform(row,strength,seed=a.seed);g=graph_for(row,a.mode);before={k:vector(v) for k,v in blocks.items()};gradients=[];losses=[];lin=counts['linear'];att=m.predictor.attention_forward_FLOPs;agg=m.graph_aggregate_forward_FLOPs;m.train()
  for branch in [row,aug]:
   opt.zero_grad(set_to_none=True);value=0.
   for view in range(4):
    x=torch.tensor(branch['cases'][view],dtype=torch.float32,device='cuda');o=m(x,g,a.mode);loss=F.binary_cross_entropy_with_logits(o['logit'],torch.full_like(o['logit'],float(view>=2)))
    if view<2:loss+=cfg['normal_NLL_weight']*o['normal_NLL'].mean()
    else:loss+=cfg['root_loss_weight']*F.cross_entropy(o['root_logits'],torch.full((704,),row['root'],dtype=torch.long,device='cuda'))/np.log(x.shape[1])
    assert torch.isfinite(loss);(loss/4).backward();value+=float(loss)/4;status['optimized_observed_query_scalars']+=704*x.shape[1];status['full_observed_input_scalar_exposures']+=x.numel()
   gradients.append({k:vector(v,gradient=True) for k,v in blocks.items()});losses.append(value)
  for k,v in blocks.items():
   combined=(gradients[0][k]+gradients[1][k])/2;offset=0
   for p in v:p.grad=combined[offset:offset+p.numel()].reshape_as(p).to(p.dtype);offset+=p.numel()
  assert torch.isfinite(torch.nn.utils.clip_grad_norm_(params,1.));opt.step();step={}
  for k,v in blocks.items():
   step[k]=protect(v,before[k],gradients[0][k],a.arm=='protected_step');r=gradients[0][k];s=gradients[1][k];step[k]['clean_aug_gradient_dot']=float(r.dot(s));step[k]['clean_aug_cosine']=float(r.dot(s)/(r.norm()*s.norm()).clamp_min(1e-30));status['projection_vector_operations']+=v.__len__()+r.numel()*10
  status['updates']+=1;status['optimization_Linear_forward_FLOPs']+=counts['linear']-lin;status['optimization_attention_forward_FLOPs']+=m.predictor.attention_forward_FLOPs-att;status['optimization_graph_aggregate_forward_FLOPs']+=m.graph_aggregate_forward_FLOPs-agg;status.update(last_clean_loss=losses[0],last_aug_loss=losses[1],seconds=time.monotonic()-start)
  with (a.output/'learning_curve.jsonl').open('a') as f:f.write(json.dumps(dict(updates=status['updates'],system_key=row['key'],clean_input_SHA256=array_sha(row['cases']),augmented_input_SHA256=array_sha(aug['cases']),network_input_dtype='float32',clean_loss=losses[0],aug_loss=losses[1],strength=strength,steps=step))+'\n')
  if status['updates']==1 or status['updates']%16==0:atomic_json(a.output/'progress.json',status)
 assert status['updates']==256 and status['optimized_observed_query_scalars']==20185088 and status['full_observed_input_scalar_exposures']==29360128
 results={}
 for name,strength,kind in [('clean',0.,'linear'),('linear2',2.,'linear'),('linear8',8.,'linear'),('quadratic4',4.,'quadratic')]:
  ev=dict(data);ev['source_validation']=[transform(r,strength,kind) for r in data['source_validation']] if strength else data['source_validation'];result=evaluate(m,ev,a.mode,min(a.deadline,start+3*3600+time.time()-time.monotonic()));results[name]=result;status['extra_frozen_query_predictions']+=result['extra_frozen_query_predictions'];atomic_json(a.output/(name+'_source_validation.json'),result)
 torch.save(dict(model=m.state_dict(),config=dict(cfg,retention_arm=a.arm,continuation_learning_rate=1e-4),mode=a.mode,seed=a.seed,source_calibration=results['clean'],source_manifest_SHA256=saved['source_manifest_SHA256']),a.output/'final.pth');status.update(status='complete',checkpoint_SHA256=sha256(a.output/'final.pth'),final_parameter_SHA256=parameter_sha(m),peak_cuda_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-start);atomic_json(a.output/'progress.json',status)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['checkpoint','source','output']:p.add_argument('--'+n,type=Path,required=True)
 for n in ['mode','arm','local-commit','github-commit']:p.add_argument('--'+n,required=True)
 p.add_argument('--seed',type=int,required=True);p.add_argument('--deadline',type=float,default=float('inf'));a=p.parse_args();assert a.arm in ['unprotected_split','protected_step']
 try:run(a)
 except Exception:
  a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
