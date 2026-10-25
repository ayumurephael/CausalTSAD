"""Frozen normal-parent teacher; exact raw TimeRCD base reused, no new encoder work."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,pickle,sys,time,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import sha256,atomic_json,model_inputs
from proposals.c09.data import digest

def legal_views(model,values,graph,chunk=128):
 import torch,math
 from dependencies.c02_model import inputs
 own,parents,fraction,query,phase=inputs(values,model.history,model.block);mus=[];lvs=[]
 for start in range(0,values.shape[1],chunk):
  end=start+chunk;mu,lv=model(own[:,start:end],parents['current'][:,start:end],fraction[:,start:end],graph,phase[start:end]);mus.append(mu);lvs.append(lv)
 mu=torch.cat(mus,1);lv=torch.cat(lvs,1);c=values.shape[-1];degree=graph.sum(1)[:,None,:].expand_as(mu)
 # Normalization depends only on this channel's block-start past; legal parents enter through the frozen teacher.
 context=torch.stack([torch.sign(mu)*torch.log1p(mu.abs()),lv,fraction,phase[None,:,None].expand_as(mu)/15,
  own.mean(-1),own.std(-1,unbiased=False),own[...,-1],own[...,-16:].mean(-1),own[...,-16:].std(-1,unbiased=False),
  own[...,-1]-own[...,-2],torch.log1p(degree),torch.full_like(mu,math.log(c))],-1).clamp(-50,50)
 residual=(query-mu)*torch.exp(-lv/2)
 return context,residual,mu,lv

def run(a):
 import torch
 from proposals.c09.graph_views import ValueGraphTeacher,views,value_graph
 cfg=json.loads(a.config.read_text());torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(cfg['memory_gib_per_worker']*2**30/torch.cuda.get_device_properties(0).total_memory)
 assert sha256(a.teacher)==cfg['teacher_checkpoint_sha256'] and sha256(a.base/'manifest.json')==cfg['true_input_cache_manifest_sha256']
 assert sha256(a.source/'manifest.json')==cfg['source_manifest_sha256']
 source=json.loads((a.source/'manifest.json').read_text());references=json.loads((a.base/'manifest.json').read_text())
 for row in source['files']:assert sha256(a.source/row['path'])==row['SHA256']
 saved=torch.load(a.teacher,map_location='cpu',weights_only=False);teacher=ValueGraphTeacher().cuda().eval().requires_grad_(False);teacher.load_state_dict(saved['model'],strict=True);before=digest(teacher)
 count=0;flops=0;rows=[];current=None;pack=None;start_time=time.monotonic()
 for row in references['files'][a.worker::a.workers]:
  assert time.monotonic()-start_time<3*3600
  assert sha256(a.base/row['path'])==row['SHA256']
  with np.load(a.base/row['path']) as f:base={k:np.array(f[k]) for k in f.files}
  if row['kind']=='development':
   info=row['row'];assert info['split']=='development'
   filename=str(a.benchmark/'data'/info['shard'])
   if current!=filename:
    with open(filename,'rb') as f:pack=pickle.load(f)
    current=filename
   sample=pack[info['offset']];assert sample['system_id']==row['key']
   x=model_inputs(sample,'anomalous');normal=model_inputs(sample,'normal');values=x['values'];normal_values=normal['values'];graph=x['adjacency'];labels=np.asarray(sample['labels'],np.int8)
  else:
   filename=str(a.source/row['pack'])
   if current!=filename:
    with np.load(filename) as f:pack={k:np.array(f[k]) for k in f.files}
    current=filename
   i=row['offset'];values=pack['x'][i];normal_values=pack['normal'][i];graph=pack['adjacency'][i];labels=pack['labels'][i]
  assert np.array_equal(labels,base['labels']);t,c=values.shape;stored=dict(x_base=base['x_base'],normal_base=base['normal_base'],labels=labels,ready=np.arange(t)>=cfg['history'])
  with torch.no_grad():
   for view,v in [('x',values),('normal',normal_values)]:
    if c==1:context=np.zeros((t,c,12),np.float32);residual=np.zeros((t,c),np.float32)
    else:
     context,residual,_,_=views(teacher,torch.from_numpy(v)[None].cuda(),torch.from_numpy(graph)[None].cuda(),a.mode,row['key'],cfg['predict_chunk'])
     context=context[0].cpu().numpy();residual=residual[0].cpu().numpy();count+=t*c
     h,w,m,p=64,64,32,16;flops+=2*t*(c*((h+1)*w+(w+m+1)*w+w*2*p)+c*c*(2*h*m+m*m))
    assert np.isfinite(context).all() and np.isfinite(residual).all();stored[view+'_context']=context;stored[view+'_residual']=residual
  if row['kind']!='development':stored.update(root=pack['root'][i],root_window=pack['root_window'][i].astype(bool))
  path=a.output/'arrays'/(row['name']+'.npz');assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**stored)
  control=value_graph(graph,a.mode,row['key']);graph_audit=dict(mode=a.mode,true_edges=int(graph.sum()),value_edges=int(control.sum()),shared_edges=int((graph*control).sum()),self_edges=int(np.trace(control)),degree_feature_preserved=True)
  record=dict(row,path=str(path.relative_to(a.output)),SHA256=sha256(path),true_cache_row_sha256=row['SHA256'],ready_scalars=int(stored['ready'].sum())*c,graph_audit=graph_audit)
  atomic_json(a.output/'records'/(row['name']+'.json'),record);rows.append(record)
  atomic_json(a.output/f'worker_{a.worker}.json',dict(status='running',completed=len(rows),total=len(references['files'][a.worker::a.workers]),teacher_query_scalars=count))
 assert digest(teacher)==before
 atomic_json(a.output/f'worker_{a.worker}.json',dict(status='complete',completed=len(rows),teacher_query_scalars=count,teacher_Linear_forward_FLOPs=flops,
  teacher_parameter_sha_before=before,teacher_parameter_sha_after=digest(teacher),teacher_checkpoint_sha256=sha256(a.teacher),encoder_forward_scalars=0,optimizer_updates=0,
  target_validation_or_locked_arrays_opened=False,observed_branch_never_replaced_with_normal=True))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['degree_only','wrong_degree'],required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True)
 p.add_argument('--teacher',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,required=True);a=p.parse_args()
 try:run(a)
 except Exception:atomic_json(a.output/f'worker_{a.worker}_error.json',dict(error=traceback.format_exc()));raise
