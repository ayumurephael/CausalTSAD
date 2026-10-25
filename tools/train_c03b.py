"""C03v2: learned guarded historical context, fixed3 density +3 fusion passes."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,time,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import sha256,atomic_json
from proposals.c09.context_model import ConditionalTail,TailFusion,ARMS,linear_flops
from proposals.c09.context_data import manifest,order,batch,digest,infer

def evaluate(tail,fusion,cache,rows,arm,cfg):
 import torch
 from sklearn.metrics import average_precision_score
 tail.eval();fusion.eval();records={};normal_all=[];base_all=[];slots=0
 alternate={}
 for c in [8,12,16,20]:
  group=sorted([r for r in rows if r['channels']==c],key=lambda r:r['key']);assert len(group)>1
  for i,r in enumerate(group):alternate[r['key']]=group[(i+1)%len(group)]
 with torch.no_grad():
  for group in order(rows,cfg['batch_input_scalars'],9002):
   assert all(r['kind']=='source_validation' for r in group);b=batch(cache,group);ready=b['ready'].bool();normal=infer(tail,fusion,b,'normal',arm);x=infer(tail,fusion,b,'x',arm)
   swapped=batch(cache,[alternate[r['key']] for r in group]);swap=tail(b['normal_context'].float(),arm,b['normal_residual'].float(),context_source=(swapped['normal_context'].float(),swapped['normal_residual'].float()))['quantiles']
   polluted=tail(b['normal_context'].float(),arm,b['normal_residual'].float(),context_source=(b['x_context'].float(),b['x_residual'].float()))['quantiles']
   error=b['normal_residual'][...,None]-normal['quantiles'];swap_error=b['normal_residual'][...,None]-swap
   pinball=torch.maximum(tail.taus*error,(tail.taus-1)*error).mean(-1);swap_pinball=torch.maximum(tail.taus*swap_error,(tail.taus-1)*swap_error).mean(-1)
   polluted_error=b['normal_residual'][...,None]-polluted
   polluted_pinball=torch.maximum(tail.taus*polluted_error,(tail.taus-1)*polluted_error).mean(-1)
   coverage=(b['normal_residual'][...,None]<=normal['quantiles']).float()
   normal_all.extend(normal['logit'].cpu().numpy().ravel());base_all.extend(b['normal_base'].cpu().numpy().ravel());slots+=4*sum(r['scalars'] for r in group)
   for j,r in enumerate(group):
    mask=ready[j];y=b['labels'][j].cpu().numpy();s=x['logit'][j].cpu().numpy();node=x['node'][j].cpu().numpy();root=None
    event=b['root_window'][j].bool()&ready[j]
    if event.any():
     score=node[event.cpu().numpy()].mean(0);rank=int(np.flatnonzero(np.argsort(-score,kind='stable')==int(b['root'][j]))[0])+1
     root=dict(Top1=int(rank==1),Top3=int(rank<=min(3,r['channels'])),MRR=1/rank)
    records[r['key']]=dict(channels=r['channels'],AP=float(average_precision_score(y,s)) if len(np.unique(y))>1 else None,
     base_AP=float(average_precision_score(y,b['x_base'][j].cpu().numpy())) if len(np.unique(y))>1 else None,
     density_ready_AP=float(average_precision_score(y[mask.cpu().numpy()],node[mask.cpu().numpy()].max(-1))) if len(np.unique(y[mask.cpu().numpy()]))>1 else None,
     normal_pinball=float(pinball[j,mask].mean()),swapped_context_pinball=float(swap_pinball[j,mask].mean()),
     observed_history_normal_query_pinball=float(polluted_pinball[j,mask].mean()),
     normal_coverage=coverage[j,mask].mean((0,1)).cpu().tolist(),conditional_ready_RCA=root)
 return dict(records=records,AP_macro=float(np.mean([r['AP'] for r in records.values() if r['AP'] is not None])),
  common_normal_pinball=float(np.mean([r['normal_pinball'] for r in records.values()])),
  threshold=float(np.quantile(normal_all,1-cfg['normal_FPR_calibration'])),base_threshold=float(np.quantile(base_all,1-cfg['normal_FPR_calibration'])),
  extra_frozen_head_query_scalars=slots,source_only=True,target_arrays_opened=False,
  context_swap_scope='normal source diagnostic only; main observed branch always its own observed context')

def run(a):
 import torch
 from torch.nn import functional as F
 cfg=json.loads(a.config.read_text());torch.set_num_threads(1);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(cfg['memory_gib_per_worker']*2**30/torch.cuda.get_device_properties(0).total_memory)
 assert not a.output.exists();a.output.mkdir(parents=True);m=manifest(a.cache)
 assert m['source_manifest_sha256']==cfg['source_manifest_sha256'] and m['teacher_checkpoint_sha256']==cfg['teacher_checkpoint_sha256']
 train=[r for r in m['files'] if r['kind']=='train'];val=[r for r in m['files'] if r['kind']=='source_validation'];assert len(train)==312 and len(val)==123
 for r in train+val:assert r['channels']>1 and sha256(a.cache/r['path'])==r['SHA256']
 tail=ConditionalTail(cfg['feature_dim'],cfg['width']).cuda();fusion=TailFusion().cuda();td=digest(tail);fd=digest(fusion)
 assert sum(r['scalars'] for r in train)==cfg['source_M_input_scalars_per_pass'];assert sum(r['ready_scalars'] for r in train)==cfg['ready_query_scalars_per_pass']
 status=dict(status='running',stage='density',pid=os.getpid(),arm=a.arm,seed=a.seed,local_commit=a.local_commit,github_commit=a.github_commit,
  source_manifest_sha256=cfg['source_manifest_sha256'],cache_manifest_sha256=sha256(a.cache/'manifest.json'),config_sha256=sha256(a.config),
  teacher_checkpoint_sha256=cfg['teacher_checkpoint_sha256'],base_checkpoint_sha256=cfg['base_checkpoint_sha256'],
  initial_tail_parameter_sha256=td,initial_fusion_parameter_sha256=fd,tail_parameters=sum(p.numel() for p in tail.parameters()),fusion_parameters=sum(p.numel() for p in fusion.parameters()),
  density_updates=0,fusion_updates=0,optimized_normal_density_query_scalars=0,optimized_observed_fusion_query_scalars=0,additional_normal_fusion_query_scalars=0,
  density_full_input_scalar_exposures=0,fusion_full_input_scalar_exposures=0,extra_source_evaluation_query_scalars=0,
  density_Linear_forward_FLOPs=cfg['density_passes']*sum(linear_flops(r['length'],r['channels'])[0] for r in train),
  fusion_stage_Linear_forward_FLOPs=2*cfg['fusion_passes']*sum(sum(linear_flops(r['length'],r['channels'])) for r in train),
  original_or_teacher_parameters_updated=False,target_arrays_in_gradient=False,
  FLOPs_note='Dense Linear forward only. Prefix/median/nonlinear/backward excluded; all slots computed, no_context effective capacity lower')
 atomic_json(a.output/'progress.json',status);start=time.monotonic();first=evaluate(tail,fusion,a.cache,val,a.arm,cfg);atomic_json(a.output/'initial_source_validation.json',first);status['extra_source_evaluation_query_scalars']+=first['extra_frozen_head_query_scalars']
 opt=torch.optim.AdamW(tail.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
 def check():
  if time.monotonic()-start>cfg['max_stage_hours']*3600:raise TimeoutError('Fixed C09 training deadline')
 def record():
  status['seconds']=time.monotonic()-start;atomic_json(a.output/'progress.json',status)
  with (a.output/'learning_curve.jsonl').open('a') as f:f.write(json.dumps({k:status[k] for k in ['stage','density_updates','fusion_updates','optimized_normal_density_query_scalars','optimized_observed_fusion_query_scalars','seconds','last_loss'] if k in status})+'\n')
 for epoch in range(cfg['density_passes']):
  for group in order(train,cfg['batch_input_scalars'],cfg['data_seed']+epoch):
   check();b=batch(a.cache,group);tail.train();opt.zero_grad(set_to_none=True);loss,_=tail.objective(b['normal_context'].float(),b['normal_residual'].float(),a.arm,b['ready'].bool());assert torch.isfinite(loss)
   loss.backward();grad=torch.nn.utils.clip_grad_norm_(tail.parameters(),cfg['gradient_clip']);assert torch.isfinite(grad);opt.step()
   status['density_updates']+=1;status['optimized_normal_density_query_scalars']+=sum(r['ready_scalars'] for r in group);status['density_full_input_scalar_exposures']+=sum(r['scalars'] for r in group);status['last_loss']=float(loss)
   if status['density_updates']==1 or status['density_updates']%20==0:record()
  if epoch==0:
   result=evaluate(tail,fusion,a.cache,val,a.arm,cfg);atomic_json(a.output/'density_pass1_source_validation.json',result);status['extra_source_evaluation_query_scalars']+=result['extra_frozen_head_query_scalars']
 density=evaluate(tail,fusion,a.cache,val,a.arm,cfg);atomic_json(a.output/'density_final_source_validation.json',density);status['extra_source_evaluation_query_scalars']+=density['extra_frozen_head_query_scalars']
 fixed_tail=digest(tail);tail.eval().requires_grad_(False);assert digest(fusion)==fd
 positives=0;points=0
 for r in train:
  with np.load(a.cache/r['path']) as z:positives+=int(z['labels'][z['ready']].sum());points+=int(z['ready'].sum())
 weight=min(100.,max(1.,(2*points-positives)/max(1,positives)));status.update(stage='fusion',fixed_tail_parameter_sha256=fixed_tail,positive_weight=weight)
 opt=torch.optim.AdamW(fusion.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
 for epoch in range(cfg['fusion_passes']):
  for group in order(train,cfg['batch_input_scalars'],cfg['data_seed']+epoch):
   check();b=batch(a.cache,group);ready=b['ready'].bool();fusion.train();opt.zero_grad(set_to_none=True)
   with torch.no_grad():nx,_=tail.evidence(b['x_context'].float(),b['x_residual'].float(),a.arm);nn,_=tail.evidence(b['normal_context'].float(),b['normal_residual'].float(),a.arm)
   x=fusion(nx,b['x_base'].float(),ready,a.arm);normal=fusion(nn,b['normal_base'].float(),ready,a.arm)
   loss=.5*(F.binary_cross_entropy_with_logits(x[ready],b['labels'][ready].float(),pos_weight=torch.tensor(weight,device='cuda'))+
    F.binary_cross_entropy_with_logits(normal[ready],torch.zeros_like(normal[ready])));assert torch.isfinite(loss)
   loss.backward();grad=torch.nn.utils.clip_grad_norm_(fusion.parameters(),cfg['gradient_clip']);assert torch.isfinite(grad);opt.step()
   count=sum(r['ready_scalars'] for r in group);status['fusion_updates']+=1;status['optimized_observed_fusion_query_scalars']+=count;status['additional_normal_fusion_query_scalars']+=count
   status['fusion_full_input_scalar_exposures']+=2*sum(r['scalars'] for r in group);status['last_loss']=float(loss)
   if status['fusion_updates']==1 or status['fusion_updates']%20==0:record()
 assert digest(tail)==fixed_tail and status['density_updates']==status['fusion_updates']==936
 assert status['optimized_normal_density_query_scalars']==status['optimized_observed_fusion_query_scalars']==status['additional_normal_fusion_query_scalars']==8948736
 final=evaluate(tail,fusion,a.cache,val,a.arm,cfg);atomic_json(a.output/'final_source_validation.json',final);status['extra_source_evaluation_query_scalars']+=final['extra_frozen_head_query_scalars']
 p=a.output/'final.partial';torch.save(dict(tail=tail.state_dict(),fusion=fusion.state_dict(),config=cfg,arm=a.arm,seed=a.seed,source_calibration=final,cache_manifest_sha256=sha256(a.cache/'manifest.json')),p);os.replace(p,a.output/'final.pth')
 status.update(status='training_complete',stage='final',final_tail_parameter_sha256=fixed_tail,final_fusion_parameter_sha256=digest(fusion),checkpoint_sha256=sha256(a.output/'final.pth'),final_source_AP=final['AP_macro'],peak_cuda_bytes=torch.cuda.max_memory_allocated());record()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--local-commit',required=True);p.add_argument('--github-commit',required=True);a=p.parse_args()
 try:run(a)
 except Exception:atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
