"""Same original C04 task, clean branch retained, matched graph and compute controls."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,argparse,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from torch.nn import functional as F
from proposals.causal_contrast import CausalContrast
from proposals.benign_nuisance import transform
from tools.train_causal_contrast import load_pool,graph_for,evaluate
from proposals.raw_context import parameter_sha
from causal_tsad.io import atomic_json,sha256
def run(a):
 assert not a.output.exists();a.output.mkdir(parents=True);torch.set_num_threads(1);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(0).total_memory)
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert saved['mode']==a.mode and saved['seed']==a.seed;model=CausalContrast().cuda();model.load_state_dict(saved['model'],strict=True);cfg=saved['config'];data=load_pool(a.source,saved['source_manifest_SHA256']);count={'linear':0}
 def hook(layer,inp,out):count['linear']+=2*layer.in_features*out.numel()
 for layer in model.modules():
  if isinstance(layer,torch.nn.Linear):layer.register_forward_hook(hook)
 status=dict(status='training',pid=os.getpid(),mode=a.mode,arm=a.arm,seed=a.seed,initial_parameter_SHA256=parameter_sha(model),initial_checkpoint_SHA256=sha256(a.checkpoint),source_manifest_SHA256=saved['source_manifest_SHA256'],local_commit=a.local_commit,github_commit=a.github_commit,
  updates=0,optimized_observed_query_scalars=0,full_observed_input_scalar_exposures=0,extra_frozen_query_predictions=0,optimization_Linear_forward_FLOPs=0,optimization_attention_forward_FLOPs=0,optimization_graph_aggregate_forward_FLOPs=0,original_TimeRCD_updated=False,target_arrays_opened=False,
  clean_branch_and_second_branch_always_executed=True,RCA_gradient_scope='root head only; shared hidden detached',nuisance_episode_count=0)
 atomic_json(a.output/'progress.json',status);start=time.monotonic();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=cfg['weight_decay']);rng=np.random.RandomState(348+a.seed-348)
 for index in rng.permutation(256):
  assert time.monotonic()-start<3*3600 and time.time()<a.deadline;row=data['train'][int(index)];strength=0. if rng.rand()<.25 else float(rng.uniform(0,8));aug=transform(row,strength,seed=a.seed);second=aug if a.arm=='augmented' else row;status['nuisance_episode_count']+=int(a.arm=='augmented' and strength>0);g=graph_for(row,a.mode);model.train();opt.zero_grad(set_to_none=True);loss_sum=0.;lin=count['linear'];att=model.predictor.attention_forward_FLOPs;agg=model.graph_aggregate_forward_FLOPs
  for branch in [row,second]:
   for view in range(4):
    x=torch.tensor(branch['cases'][view],dtype=torch.float32,device='cuda');out=model(x,g,a.mode);loss=F.binary_cross_entropy_with_logits(out['logit'],torch.full_like(out['logit'],float(view>=2)))
    if view<2:loss+=cfg['normal_NLL_weight']*out['normal_NLL'].mean()
    else:loss+=cfg['root_loss_weight']*F.cross_entropy(out['root_logits'],torch.full((704,),row['root'],dtype=torch.long,device='cuda'))/np.log(x.shape[1])
    assert torch.isfinite(loss);(loss/8).backward();loss_sum+=float(loss)/8;status['optimized_observed_query_scalars']+=704*x.shape[1];status['full_observed_input_scalar_exposures']+=x.numel()
  grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad);opt.step();status['updates']+=1;status['optimization_Linear_forward_FLOPs']+=count['linear']-lin;status['optimization_attention_forward_FLOPs']+=model.predictor.attention_forward_FLOPs-att;status['optimization_graph_aggregate_forward_FLOPs']+=model.graph_aggregate_forward_FLOPs-agg
  status.update(last_loss=loss_sum,seconds=time.monotonic()-start)
  if status['updates']==1 or status['updates']%16==0:
   atomic_json(a.output/'progress.json',status)
   with (a.output/'learning_curve.jsonl').open('a') as f:f.write(json.dumps({k:status[k] for k in ['updates','optimized_observed_query_scalars','last_loss','seconds']})+'\n')
 assert status['updates']==256 and status['optimized_observed_query_scalars']==20185088 and status['full_observed_input_scalar_exposures']==29360128
 results={}
 for condition,strength,kind in [('clean',0.,'linear'),('linear2',2.,'linear'),('linear8',8.,'linear'),('quadratic4',4.,'quadratic')]:
  # The calibration pool stays clean for all nuisance stress conditions, so a
  # raised stress threshold cannot conceal false positives.
  evaluation=dict(data);evaluation['source_validation']=[transform(row,strength,kind) for row in data['source_validation']] if strength else data['source_validation'];remaining=3*3600-(time.monotonic()-start);assert remaining>0;result=evaluate(model,evaluation,a.mode,min(a.deadline,time.time()+remaining));results[condition]=result;status['extra_frozen_query_predictions']+=result['extra_frozen_query_predictions'];atomic_json(a.output/(condition+'_source_validation.json'),result)
 final_cfg=dict(cfg,continuation_learning_rate=1e-4,continuation_passes=1,augmentation_arm=a.arm);torch.save(dict(model=model.state_dict(),config=final_cfg,mode=a.mode,seed=a.seed,source_calibration=results['clean'],source_manifest_SHA256=saved['source_manifest_SHA256']),a.output/'final.pth');status.update(status='complete',checkpoint_SHA256=sha256(a.output/'final.pth'),final_parameter_SHA256=parameter_sha(model),seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated());atomic_json(a.output/'progress.json',status)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['checkpoint','source','output']:p.add_argument('--'+n,type=Path,required=True)
 for n in ['mode','arm','local-commit','github-commit']:p.add_argument('--'+n,required=True)
 p.add_argument('--seed',type=int,required=True);p.add_argument('--deadline',type=float,default=float('inf'));a=p.parse_args()
 try:run(a)
 except Exception:
  a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
