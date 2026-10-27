"""Finite source-only causal-context probe; no benchmark arguments or TimeRCD changes."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,traceback,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from proposals.raw_context import RawContextPredictor,ARMS,prepare,wrong_graph,losses,parameter_sha,FIRST_QUERY
from causal_tsad.io import atomic_json,sha256

def load(source,cfg):
 assert sha256(source/'manifest.json')==cfg['source_manifest_sha256']
 manifest=json.loads((source/'manifest.json').read_text());groups={'train':[],'source_validation':[]}
 for row in manifest['files']:
  if row['channels']==1:continue
  assert sha256(source/row['path'])==row['SHA256']
  with np.load(source/row['path']) as f:
   for i in range(row['systems']):
    key=row['split']+'/'+str(row['channels'])+'/'+str(int(f['seed'][i]))
    groups[row['split']].append(dict(key=key,x=f['x'][i].copy(),normal=f['normal'][i].copy(),graph=f['adjacency'][i].copy(),labels=f['labels'][i].copy(),root=int(f['root'][i]),root_window=f['root_window'][i].copy()))
 assert len(groups['train'])==312 and len(groups['source_validation'])==123
 return groups

def features(row,view,arm):
 values=torch.from_numpy(row[view]).cuda();graph=torch.from_numpy(row['graph']).cuda()
 if arm.startswith('wrong'):graph=wrong_graph(graph,row['key'])
 inputs,y,times,scale=prepare(values)
 return inputs,y,times,scale,graph

def evaluate(model,rows,arm,deadline=float('inf')):
 from sklearn.metrics import average_precision_score
 model.eval();records={};normal_scores=[];query_count=0
 with torch.no_grad():
  for idx,row in enumerate(rows):
   assert time.time()<deadline,'Fixed campaign deadline during source validation'
   inp,y,times,scale,graph=features(row,'normal',arm);mu,lv=model.predict(inp,graph,arm);nll,mse,score=losses(mu,lv,y)
   # Same-C different-system context, preserving this query and its normalization.
   alternate=next(rows[(idx+j)%len(rows)] for j in range(1,len(rows)) if rows[(idx+j)%len(rows)]['normal'].shape[1]==row['normal'].shape[1])
   other,_,_,_,_=features(alternate,'normal',arm)
   swapped=dict(inp)
   for name in ['context_own','context_parent','context_y','context_valid']:swapped[name]=other[name]
   sm,sl=model.predict(swapped,graph,arm);sn,ss,_=losses(sm,sl,y)
   normal_score=score.max(-1).values.cpu().numpy();normal_scores.extend(normal_score.tolist())
   obs,oy,ot,_,og=features(row,'x',arm);om,ol=model.predict(obs,og,arm);on,_,oscore=losses(om,ol,oy)
   labels=row['labels'][FIRST_QUERY:];observed_score=oscore.max(-1).values.cpu().numpy()
   event=row['root_window'][FIRST_QUERY:].astype(bool);rca=None
   if event.any():
    ranking=np.argsort(-oscore.cpu().numpy()[event].mean(0),kind='stable');rank=int(np.flatnonzero(ranking==row['root'])[0])+1
    rca=dict(Top1=int(rank==1),Top3=int(rank<=min(3,len(ranking))),MRR=1/rank)
   c=row['normal'].shape[1]
   records[row['key']]=dict(channels=c,normal_NLL=float(nll.mean()),normal_MSE=float(mse.mean()),normal_raw_MSE=float((mse*scale.square()).mean()),
    swapped_NLL=float(sn.mean()),swapped_MSE=float(ss.mean()),AP_ready=float(average_precision_score(labels,observed_score)) if len(np.unique(labels))>1 else None,
    conditional_ready_RCA=rca,prefix_anomaly_fraction=float(row['labels'][:256].mean()),normal_score_q99=float(np.quantile(normal_score,.99)))
   query_count+=3*y.numel()
 return dict(records=records,normal_NLL_macro=float(np.mean([r['normal_NLL'] for r in records.values()])),normal_MSE_macro=float(np.mean([r['normal_MSE'] for r in records.values()])),
  source_threshold99=float(np.quantile(normal_scores,.99)),extra_frozen_query_predictions=query_count,target_arrays_opened=False,
  scope='source validation only; ready t>=320; no TimeRCD fusion or full detector claims; context is observed prefix, not guaranteed clean')

def run(a):
 cfg=json.loads(a.config.read_text());torch.set_num_threads(1);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(cfg['memory_gib']*2**30/torch.cuda.get_device_properties(0).total_memory)
 assert torch.cuda.is_available() and not a.output.exists();a.output.mkdir(parents=True)
 data=load(a.source,cfg);model=RawContextPredictor(cfg['width']).cuda();initial=parameter_sha(model)
 params=sum(p.numel() for p in model.parameters());start=time.monotonic()
 counts={'linear':0}
 def hook(layer,inputs,output):counts['linear']+=2*layer.in_features*output.numel()
 for layer in model.modules():
  if isinstance(layer,torch.nn.Linear):layer.register_forward_hook(hook)
 status=dict(status='training',pid=os.getpid(),arm=a.arm,seed=a.seed,local_commit=a.local_commit,github_commit=a.github_commit,
  source_manifest_sha256=cfg['source_manifest_sha256'],config_sha256=sha256(a.config),initial_parameter_sha256=initial,parameters=params,
  updates=0,optimized_normal_query_scalars=0,full_normal_input_scalar_exposures=0,extra_frozen_query_predictions=0,target_arrays_opened=False,
  optimization_Linear_forward_FLOPs=0,optimization_attention_matmul_forward_FLOPs=0,extra_evaluation_Linear_forward_FLOPs=0,extra_evaluation_attention_matmul_forward_FLOPs=0,
  original_TimeRCD_or_teacher_loaded=False,training_labels_or_roots_used=False,context_point_pairs_per_channel_edge=48,
  FLOPs_scope='All pair slots, key/query/value matmuls run in all arms. Effective contextual capacity differs for mean/none; backward and nonlinear FLOPs not equated.')
 atomic_json(a.output/'progress.json',status)
 def evaluation(name):
  lin=counts['linear'];att=model.attention_forward_FLOPs
  result=evaluate(model,data['source_validation'],a.arm,a.campaign_deadline);atomic_json(a.output/(name+'.json'),result);status['extra_frozen_query_predictions']+=result['extra_frozen_query_predictions']
  status['extra_evaluation_Linear_forward_FLOPs']+=counts['linear']-lin;status['extra_evaluation_attention_matmul_forward_FLOPs']+=model.attention_forward_FLOPs-att
  return result
 evaluation('initial_source_validation');opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
 expected_per_pass=sum((len(r['normal'])-FIRST_QUERY)*r['normal'].shape[1] for r in data['train'])
 assert expected_per_pass==cfg['normal_query_scalars_per_pass']
 for epoch in range(cfg['passes']):
  order=np.random.RandomState(cfg['data_seed']+epoch).permutation(len(data['train']))
  for index in order:
   assert time.monotonic()-start<cfg['max_hours_per_run']*3600 and time.time()<a.campaign_deadline
   row=data['train'][int(index)];model.train();opt.zero_grad(set_to_none=True)
   lin=counts['linear'];att=model.attention_forward_FLOPs
   inp,y,_,_,graph=features(row,'normal',a.arm);mu,lv=model.predict(inp,graph,a.arm);nll,_,_=losses(mu,lv,y);loss=nll.mean();assert torch.isfinite(loss)
   loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);assert torch.isfinite(grad);opt.step()
   status['updates']+=1;status['optimized_normal_query_scalars']+=y.numel();status['full_normal_input_scalar_exposures']+=row['normal'].size
   status['optimization_Linear_forward_FLOPs']+=counts['linear']-lin;status['optimization_attention_matmul_forward_FLOPs']+=model.attention_forward_FLOPs-att
   status.update(seconds=time.monotonic()-start,last_loss=float(loss))
   if status['updates']==1 or status['updates']%20==0:
    atomic_json(a.output/'progress.json',status)
    with (a.output/'learning_curve.jsonl').open('a') as f:f.write(json.dumps({k:status[k] for k in ['updates','optimized_normal_query_scalars','last_loss','seconds']})+'\n')
  if epoch==0:evaluation('pass1_source_validation')
 assert status['updates']==936 and status['optimized_normal_query_scalars']==cfg['normal_query_scalars_per_pass']*cfg['passes']
 final=evaluation('final_source_validation');torch.save(dict(model=model.state_dict(),config=cfg,arm=a.arm,seed=a.seed,source_calibration=final),a.output/'final.pth')
 status.update(status='complete',seconds=time.monotonic()-start,final_parameter_sha256=parameter_sha(model),checkpoint_sha256=sha256(a.output/'final.pth'),peak_cuda_bytes=torch.cuda.max_memory_allocated())
 atomic_json(a.output/'progress.json',status)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['source','config','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--local-commit',required=True);p.add_argument('--github-commit',required=True);p.add_argument('--campaign-deadline',type=float,default=float('inf'));a=p.parse_args()
 try:run(a)
 except Exception:
  a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
