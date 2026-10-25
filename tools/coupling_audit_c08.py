"""Zero-update causal joint-head diagnosis: training-only gradients and frozen readout controls."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,time,traceback,math,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import sha256,atomic_json
from proposals.c08.model import JointRootReadout
from proposals.c08.data import load_manifest,batch,infer,digest

def flat_grad(loss,model,retain=False):
 import torch
 gradients=torch.autograd.grad(loss,list(model.parameters()),retain_graph=retain,allow_unused=True)
 return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(model.parameters(),gradients)])

def variants(out,seed,key,normal_constant=0.):
 s=out['logit'][0].detach().cpu().double().numpy();lp=out['root_log_prob'][0].detach().cpu().double().numpy();c=lp.shape[-1]
 kl=-lp.mean(-1)-math.log(c)
 assert kl.min()>-1e-5
 shift=np.random.RandomState(seed^int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little')).permutation(len(kl))
 return dict(original=s,remove_uniform_to_root_KL=s-kl,subtract_source_normal_mean_KL=s-normal_constant,subtract_time_shuffled_KL=s-kl[shift]),kl

def main(a):
 import torch
 from torch.nn import functional as F
 from sklearn.metrics import average_precision_score
 cfg=json.loads(a.config.read_text());torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(cfg['memory_gib_per_worker']*2**30/torch.cuda.get_device_properties(0).total_memory)
 assert not a.output.exists();a.output.mkdir(parents=True);began=time.monotonic();m=load_manifest(a.cache);base=a.parent/'runs'/f'{a.arm}_s{a.seed}'
 saved=torch.load(base/'final.pth',map_location='cpu',weights_only=False);progress=json.loads((base/'progress.json').read_text());assert sha256(base/'final.pth')==progress['checkpoint_sha256']
 final=JointRootReadout().cuda().eval();final.load_state_dict(saved['model'],strict=True);before=digest(final);assert before==progress['final_parameter_sha256']
 torch.manual_seed(a.seed);initial=JointRootReadout().cuda().eval();assert digest(initial)==progress['initial_parameter_sha256']
 selected=[]
 for c in cfg['channels']:
  group=[r for r in m['files'] if r['kind']=='train' and r['channels']==c];group.sort(key=lambda r:hashlib.sha256(r['key'].encode()).hexdigest());assert len(group)>=16;selected+=group[:16]
 gradients=[];grad_slots=0
 for stage,model in [('initial',initial),('final',final)]:
  for row in selected:
   assert row['kind']=='train' and sha256(a.cache/row['path'])==row['SHA256'];b=batch(a.cache,[row],a.arm);x=infer(model,b,'x');normal=infer(model,b,'normal');valid=b['valid'].bool();mask=b['root_window'].bool()&b['labels'].bool()&valid
   det=.5*(F.binary_cross_entropy_with_logits(x['logit'][valid],b['labels'][valid].float(),pos_weight=torch.tensor(progress['positive_weight'],device='cuda'))+F.binary_cross_entropy_with_logits(normal['logit'][valid],torch.zeros_like(normal['logit'][valid])))
   roots=b['true_root'][:,None,None].expand(*b['labels'].shape,1).long();root=-x['root_log_prob'].gather(-1,roots).squeeze(-1)[mask].mean()/math.log(row['channels']) if mask.any() else x['node_logits'].sum()*0
   gd=flat_grad(det,model,True);gr=flat_grad(root,model);dot=float(torch.dot(gd,gr));nd=float(gd.norm());nr=float(gr.norm());cos=dot/(nd*nr) if nd*nr>0 else None
   gradients.append(dict(system_id=row['key'],channels=row['channels'],stage=stage,detector_loss=float(det),root_loss=float(root),dot=dot,detector_norm=nd,root_norm=nr,cosine=cos,root_points=int(mask.sum())))
   grad_slots+=2*row['scalars'];model.zero_grad(set_to_none=True)
  atomic_json(a.output/'progress.json',dict(status='running',stage='source_train_gradients',completed=len(gradients),gradient_query_forward_scalars=grad_slots,optimizer_updates=0))
 assert digest(initial)==progress['initial_parameter_sha256'] and digest(final)==before
 source=[r for r in m['files'] if r['kind']=='source_validation' and r['channels']>1];development=[r for r in m['files'] if r['kind']=='development'];assert len(source)==123 and len(development)==100
 predictions={};slots=0;null=0.;normal_kl=[]
 with torch.no_grad():
  for row in source+development:
   if time.monotonic()-began>cfg['max_hours_per_job']*3600:raise TimeoutError('Fixed audit deadline')
   b=batch(a.cache,[row],a.arm);stored={}
   for view in ['x','normal']:
    out=infer(final,b,view);vs,kl=variants(out,a.seed,row['key']+'/'+view);stored[view]=dict(variants=vs,kl=kl,root=out['root_log_prob'][0].exp().cpu().numpy());slots+=row['scalars']
   stored['labels']=b['labels'][0].cpu().numpy();predictions[row['key']]=stored
   if row['kind']=='source_validation':normal_kl.extend(stored['normal']['kl'].tolist())
   else:
    old=np.load(base/'development_scores'/(row['name']+'.npz'))
    for v in ['x','normal']:null=max(null,float(np.max(abs(stored[v]['variants']['original']-old[v+'_fused']))))
 assert null<1e-6,'Frozen original readout must match prior scores'
 constant=float(np.mean(normal_kl));thresholds={};methods=cfg['readout_variants']
 for method in methods:
  values=np.concatenate([predictions[r['key']]['normal']['variants'][method]-(constant if method=='subtract_source_normal_mean_KL' else 0) for r in source])
  thresholds[method]=float(np.quantile(values,.99))
 records=[]
 for row in source+development:
  p=predictions[row['key']];y=p['labels'];result={}
  for method in methods:
   xs=p['x']['variants'][method]-(constant if method=='subtract_source_normal_mean_KL' else 0);ns=p['normal']['variants'][method]-(constant if method=='subtract_source_normal_mean_KL' else 0)
   result[method]=dict(AP=float(average_precision_score(y,xs)) if len(np.unique(y))>1 else None,normal_FPR=float(np.mean(ns>thresholds[method])))
  if len(np.unique(y))>1:assert abs(result['original']['AP']-result['subtract_source_normal_mean_KL']['AP'])<1e-7,'Constant shift should be a rank null'
  records.append(dict(system_id=row['key'],kind=row['kind'],channels=row['channels'],track=row.get('row',{}).get('track'),methods=result,
   normal_KL_mean=float(p['normal']['kl'].mean()),observed_normal_label_KL_mean=float(p['x']['kl'][y==0].mean()) if (y==0).any() else None,
   anomaly_KL_mean=float(p['x']['kl'][y==1].mean()) if (y==1).any() else None))
 assert digest(final)==before and sha256(base/'final.pth')==progress['checkpoint_sha256']
 atomic_json(a.output/'result.json',dict(status='complete',arm=a.arm,seed=a.seed,gradients=gradients,records=records,source_normal_KL_mean=constant,source_thresholds=thresholds,
  original_null_max_error=null,parameter_SHA_before=before,parameter_SHA_after=digest(final),checkpoint_sha256=sha256(base/'final.pth'),cache_manifest_sha256=sha256(a.cache/'manifest.json'),
  gradient_query_forward_scalars=grad_slots,gradient_backward_tasks=2*len(gradients),extra_frozen_query_scalars=slots,optimizer_updates=0,new_encoder_forwards=0,
  target_gradient=False,validation_locked_opened=False,code_local_commit=a.local_commit,code_github_commit=a.github_commit,
  identity='S=B+a+mean(r)+KL(U||p(root)); confidence-coupling hypothesis, not a physical do-intervention'))
 atomic_json(a.output/'progress.json',dict(status='complete',extra_frozen_query_scalars=slots,gradient_query_forward_scalars=grad_slots,optimizer_updates=0))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--local-commit',required=True);p.add_argument('--github-commit',required=True);a=p.parse_args()
 try:main(a)
 except Exception:atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
