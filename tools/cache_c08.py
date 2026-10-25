"""Freeze one own-history teacher, reuse old encoder/base cache without new encoder calls."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,time,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import sha256,atomic_json
from proposals.c08.data import digest

def moving(a,w):
 p=np.concatenate([np.zeros((1,a.shape[1])),np.cumsum(a,axis=0,dtype=np.float64)])
 ends=np.arange(1,len(a)+1);starts=np.maximum(0,ends-w)
 return ((p[ends]-p[starts])/(ends-starts)[:,None]).astype(np.float32)

def run(a):
 import torch
 from dependencies.c01_model import CausalResidualDetector
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 cfg=json.loads(a.config.read_text());assert sha256(a.teacher)==cfg['teacher_checkpoint_sha256']
 assert sha256(a.upstream/'manifest.json')==cfg['upstream_cache_manifest_sha256']
 assert sha256(a.source/'manifest.json')==cfg['source_manifest_sha256']
 upstream=json.loads((a.upstream/'manifest.json').read_text());source=json.loads((a.source/'manifest.json').read_text())
 for r in source['files']:assert sha256(a.source/r['path'])==r['SHA256']
 saved=torch.load(a.teacher,map_location='cpu',weights_only=False);tc=saved['config']
 teacher=CausalResidualDetector(saved['feature_dim'],tc['width'],tc['message_width'],tc['horizon']).cuda().eval().requires_grad_(False)
 teacher.load_state_dict(saved['model'],strict=True);before=digest(teacher)
 tasks=upstream['files'][a.worker::a.workers];rows=[];count=0;flops=0;began=time.monotonic();current=None;pack=None
 for row in tasks:
  assert time.monotonic()-began<cfg['max_campaign_hours']*3600
  assert row['kind'] in ['train','source_validation','development'];assert sha256(a.upstream/row['path'])==row['SHA256']
  with np.load(a.upstream/row['path']) as f:z={k:np.array(f[k]) for k in f.files}
  t=row['length'];c=row['channels'];stored=dict(graph=z['graph'],valid=np.ones(t,bool),labels=z['labels'].ravel()[:t]);n=0
  with torch.no_grad():
   for view in ['x','normal']:
    if c==1:
     stored[view+'_features']=np.zeros((t,1,cfg['feature_dim']),np.float32);stored[view+'_base']=z[view+'_base'].ravel()[:t]
     stored[view+'_feature_clipped']=np.asarray(0);continue
    features=torch.from_numpy(z[view+'_features'])[None].cuda();frac=torch.from_numpy(z[view+'_fraction'])[None].cuda()
    graph=torch.zeros(1,c,c,device='cuda');mu,lv=teacher.density(features,frac,graph)
    mu=mu[0].cpu().numpy().reshape(-1,c)[:t].astype(np.float64);lv=lv[0].cpu().numpy().reshape(-1,c)[:t].astype(np.float64)
    query=z[view+'_query'].reshape(-1,c)[:t].astype(np.float64);residual=(query-mu)*np.exp(-lv/2)
    signed=np.sign(residual)*np.log1p(np.abs(residual));energy=np.log1p(residual**2)
    q=np.sign(query)*np.log1p(np.abs(query));m=np.sign(mu)*np.log1p(np.abs(mu))
    fraction=np.repeat(z[view+'_fraction'],tc['horizon'],axis=0)[:t]
    feats=[signed,energy,lv,m,q,fraction]+[moving(energy,w) for w in [8,32,64]]+[moving(signed,w) for w in [8,32,64]]
    result=np.stack(feats,-1);assert np.isfinite(result).all();clipped=int((np.abs(result)>50).sum())
    stored[view+'_features']=np.clip(result,-50,50).astype(np.float32)
    stored[view+'_base']=z[view+'_base'].ravel()[:t];stored[view+'_feature_clipped']=np.asarray(clipped)
    n+=t*c
    d,w,m,h=saved['feature_dim'],tc['width'],tc['message_width'],tc['horizon']
    flops+=2*((t+h-1)//h)*(c*((d+1)*w+(w+m+1)*w+w*2*h)+c*c*(2*d*m+m*m))
  if row['kind']!='development':
   if current!=row['pack']:
    with np.load(a.source/row['pack']) as f:pack={k:np.array(f[k]) for k in ['root','root_window','labels']}
    current=row['pack']
   i=row['offset'];stored.update(root=np.asarray(pack['root'][i]),root_window=pack['root_window'][i].astype(bool))
   assert np.array_equal(stored['labels'],pack['labels'][i])
  p=a.output/'arrays'/(row['name']+'.npz');assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:np.savez_compressed(f,**stored)
  meta=dict(row,path=str(p.relative_to(a.output)),SHA256=sha256(p),upstream_array_sha256=row['SHA256'],teacher_query_scalars=n)
  atomic_json(a.output/'records'/(row['name']+'.json'),meta);rows.append(meta);count+=n
  atomic_json(a.output/f'worker_{a.worker}.json',dict(status='running',completed=len(rows),total=len(tasks),teacher_query_scalars=count))
 assert digest(teacher)==before
 atomic_json(a.output/f'worker_{a.worker}.json',dict(status='complete',completed=len(rows),total=len(tasks),teacher_query_scalars=count,
  teacher_parameter_sha_before=before,teacher_parameter_sha_after=digest(teacher),encoder_forward_scalars=0,optimizer_updates=0,teacher_Linear_forward_FLOPs=flops,
  normal_twin_never_used_for_observed_features=True,teacher_checkpoint_sha256=sha256(a.teacher),upstream_manifest_sha256=sha256(a.upstream/'manifest.json')))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
 p.add_argument('--teacher',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,required=True);a=p.parse_args()
 try:run(a)
 except Exception:atomic_json(a.output/f'worker_{a.worker}_error.json',dict(error=traceback.format_exc()));raise
