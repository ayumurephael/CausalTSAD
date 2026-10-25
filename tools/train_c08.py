"""One fixed pass through multivariate independent source systems, frozen detector/teacher."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,time,traceback,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import sha256,atomic_json
from proposals.c08.data import load_manifest,order,batch,infer,digest
from proposals.c08.model import JointRootReadout,ARMS,linear_flops

def evaluate(model,cache,rows,arm,cfg):
 import torch
 from sklearn.metrics import average_precision_score
 results={};normal_all=[];base_normal=[];slots=0;points=0;root_evidence_points=0
 model.eval()
 with torch.no_grad():
  for group in order(rows,cfg['batch_input_scalars'],9002):
   assert all(r['kind']=='source_validation' for r in group)
   b=batch(cache,group,arm);out={v:infer(model,b,v) for v in ['x','normal']}
   normal_all.extend(out['normal']['logit'].cpu().numpy().ravel());base_normal.extend(b['normal_base'].cpu().numpy().ravel())
   slots+=2*sum(r['scalars'] for r in group);points+=2*int(b['valid'].sum())
   for j,r in enumerate(group):
    y=b['labels'][j].cpu().numpy();s=out['x']['logit'][j].cpu().numpy();root=None
    mask=b['root_window'][j].bool()&b['labels'][j].bool();root_evidence_points+=int(mask.sum())
    if r['channels']>1 and mask.any():
     scores=out['x']['root_log_prob'][j,mask].exp().mean(0).cpu().numpy();ranking=np.argsort(-scores,kind='stable');rank=int(np.flatnonzero(ranking==int(b['root'][j]))[0])+1
     root=dict(Top1=int(rank==1),Top3=int(rank<=min(3,r['channels'])),MRR=1/rank,rank=rank)
    results[r['key']]=dict(channels=r['channels'],AP=float(average_precision_score(y,s)) if len(np.unique(y))>1 else None,
     base_AP=float(average_precision_score(y,b['x_base'][j].cpu().numpy())) if len(np.unique(y))>1 else None,
     conditional_RCA=root,normal_logit_mean=float(out['normal']['logit'][j].mean()))
 threshold=float(np.quantile(normal_all,1-cfg['normal_FPR_calibration']));base_threshold=float(np.quantile(base_normal,1-cfg['normal_FPR_calibration']))
 return dict(records=results,AP_macro=float(np.mean([r['AP'] for r in results.values() if r['AP'] is not None])),
  threshold=threshold,base_threshold=base_threshold,head_query_scalars=slots,head_query_timepoints=points,root_evaluation_points=root_evidence_points,
  source_only=True,target_arrays_opened=False)

def run(a):
 import torch
 from torch.nn import functional as F
 cfg=json.loads(a.config.read_text());torch.set_num_threads(1);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(cfg['memory_gib_per_worker']*2**30/torch.cuda.get_device_properties(0).total_memory)
 assert not a.output.exists();a.output.mkdir(parents=True)
 m=load_manifest(a.cache);assert m['source_manifest_sha256']==cfg['source_manifest_sha256']
 training=[r for r in m['files'] if r['kind']=='train' and r['channels']>1]
 validation=[r for r in m['files'] if r['kind']=='source_validation' and r['channels']>1]
 assert len(training)==312 and len(validation)==123
 for r in training+validation:assert sha256(a.cache/r['path'])==r['SHA256']
 model=JointRootReadout(cfg['feature_dim'],cfg['width'],cfg['layers']).cuda();initial=digest(model)
 one_pass=sum(r['scalars'] for r in training);assert one_pass==cfg['original_multivariate_scalars_per_pass']
 epoch_plans=[order(training,cfg['batch_input_scalars'],cfg['data_seed']+epoch) for epoch in range(cfg['passes'])]
 plan=[rows for epoch in epoch_plans for rows in epoch];total=one_pass*cfg['passes']
 positive=0;timepoints=0;root_points=0
 for r in training:
  with np.load(a.cache/r['path']) as z:positive+=int(z['labels'].sum());timepoints+=len(z['labels']);root_points+=int((z['root_window']&z['labels'].astype(bool)).sum())
 weight=min(100.,max(1.,(2*timepoints-positive)/max(1,positive)))
 status=dict(status='running',pid=os.getpid(),arm=a.arm,seed=a.seed,updates=0,total_updates=len(plan),
  parameters=sum(p.numel() for p in model.parameters()),initial_parameter_sha256=initial,
  cache_manifest_sha256=sha256(a.cache/'manifest.json'),source_manifest_sha256=cfg['source_manifest_sha256'],config_sha256=sha256(a.config),
  teacher_checkpoint_sha256=cfg['teacher_checkpoint_sha256'],original_checkpoint_sha256=cfg['base_checkpoint_sha256'],
  local_commit=a.local_commit,github_commit=a.github_commit,optimized_original_observed_scalars=0,additional_normal_training_scalars=0,
  supervised_root_timepoints=0,available_training_root_timepoints=root_points,root_loss_enabled=ARMS[a.arm][1],
  passes=cfg['passes'],extra_source_evaluation_query_scalars=0,two_view_Linear_forward_FLOPs=2*cfg['passes']*sum(linear_flops(model,r['length'],r['channels']) for r in training),
  FLOPs_note='Dense Linear forward only; normalization/softmax/root-loss/backward/eval/cache distinct; active effective capacity differs under masking',
  original_or_teacher_parameters_updated=False,target_arrays_in_gradient=False,positive_weight=weight)
 atomic_json(a.output/'progress.json',status);start=time.monotonic()
 first=evaluate(model,a.cache,validation,a.arm,cfg);atomic_json(a.output/'initial_source_validation.json',first);status['extra_source_evaluation_query_scalars']+=first['head_query_scalars']
 optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
 for step,group in enumerate(plan,1):
  if time.monotonic()-start>cfg['max_train_hours_per_run']*3600:raise TimeoutError('Fixed pilot deadline')
  assert all(r['kind']=='train' and r['channels']>1 for r in group)
  model.train();b=batch(a.cache,group,a.arm);optimizer.zero_grad(set_to_none=True);x=infer(model,b,'x');n=infer(model,b,'normal');valid=b['valid'].bool()
  ce=.5*(F.binary_cross_entropy_with_logits(x['logit'][valid],b['labels'][valid].float(),pos_weight=torch.tensor(weight,device='cuda'))+
          F.binary_cross_entropy_with_logits(n['logit'][valid],torch.zeros_like(n['logit'][valid])))
  roots=b['root'][:,None].expand_as(b['labels']).long();mask=b['root_window'].bool()&b['labels'].bool()&valid
  root_loss=(-x['root_log_prob'].gather(-1,roots[...,None]).squeeze(-1)[mask].mean()/math.log(group[0]['channels'])) if mask.any() else x['node_logits'].sum()*0
  enabled=ARMS[a.arm][1];loss=ce+cfg['root_loss_weight']*int(enabled)*root_loss
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);assert torch.isfinite(grad)
  optimizer.step();count=sum(r['scalars'] for r in group)
  status['optimized_original_observed_scalars']+=count;status['additional_normal_training_scalars']+=count
  status['supervised_root_timepoints']+=int(mask.sum())*int(enabled)
  status.update(updates=step,last_loss=float(loss),last_BCE=float(ce),last_root_NLL=float(root_loss),seconds=time.monotonic()-start)
  with (a.output/'learning_curve.jsonl').open('a') as f:f.write(json.dumps({k:status[k] for k in ['updates','last_loss','last_BCE','last_root_NLL','optimized_original_observed_scalars','supervised_root_timepoints','seconds']})+'\n')
  if step==1 or step%20==0 or step==len(plan):atomic_json(a.output/'progress.json',status)
  if step==len(epoch_plans[0]):
   probe=evaluate(model,a.cache,validation,a.arm,cfg);atomic_json(a.output/'pass1_source_validation.json',probe)
   status['extra_source_evaluation_query_scalars']+=probe['head_query_scalars']
 assert status['optimized_original_observed_scalars']==total
 final=evaluate(model,a.cache,validation,a.arm,cfg);atomic_json(a.output/'final_source_validation.json',final);status['extra_source_evaluation_query_scalars']+=final['head_query_scalars']
 p=a.output/'final.partial';torch.save(dict(model=model.state_dict(),config=cfg,arm=a.arm,seed=a.seed,source_calibration=final,cache_manifest_sha256=sha256(a.cache/'manifest.json')),p);os.replace(p,a.output/'final.pth')
 status.update(status='training_complete',final_parameter_sha256=digest(model),checkpoint_sha256=sha256(a.output/'final.pth'),final_source_AP=final['AP_macro'],peak_cuda_bytes=torch.cuda.max_memory_allocated())
 atomic_json(a.output/'progress.json',status)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--arm',required=True,choices=list(ARMS));p.add_argument('--seed',type=int,required=True);p.add_argument('--local-commit',required=True);p.add_argument('--github-commit',required=True);a=p.parse_args()
 try:run(a)
 except Exception:atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
