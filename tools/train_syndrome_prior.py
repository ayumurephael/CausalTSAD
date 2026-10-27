"""Source-only7 parameter hypothesis prior. Raw kernel and context rules remain fixed."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,argparse,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,roc_auc_score
from proposals.causal_syndromes import SyndromePrior
from proposals.raw_context import parameter_sha
from causal_tsad.io import atomic_json,sha256
@torch.no_grad()
def evaluate(model,groups):
 normal=[];queries=0
 for row,factors in groups['calibration']:
  output=model(factors);normal.extend(output['logit'].numpy().reshape(-1).tolist());queries+=704*row['C']*2
 threshold=float(np.quantile(normal,.99));records={}
 for row,factors in groups['source_validation']:
  out=model(factors);scores=out['logit'].numpy();roots=out['root'].numpy();component=out['component'];physical=(component[...,:3].logsumexp((-2,-1))-component.logsumexp((-2,-1))).exp().numpy();metrics={}
  for scope,views in [('matched_measurement',[0,1,2,3]),('paired_measurement',[0,1,4,5]),('physical',[0,1,6,7]),('all8',list(range(8)))]:
   y=np.repeat([float(i>=2) for i in views],704);s=scores[views].reshape(-1);metrics[scope]=dict(AP=float(average_precision_score(y,s)),ROC_AUC=float(roc_auc_score(y,s)))
  rca=[]
  for view in range(2,8):
   ranks=np.argsort(-roots[view].mean(0),kind='stable');rank=int(np.flatnonzero(ranks==row['root'])[0])+1;rca.append(dict(view=view,kind=row['case_kind'][view],Top1=int(rank==1),Top3=int(rank<=3),MRR=1/rank,mean_physical_given_fault=float(physical[view].mean())))
  records[row['key']]=dict(C=row['C'],family=row['family'],prefix_contaminated=row['prefix_contaminated'],metrics=metrics,normal_FPR=float(np.mean(scores[:2]>threshold)),RCA=rca);queries+=704*row['C']*8
 return dict(records=records,source_threshold=threshold,cached_query_equivalent_evaluations=queries,scope='32 independent sourceval groups, conditional true suffix RCA; not raw250M/TSAD SOTA')
def run(a):
 assert not a.output.exists();a.output.mkdir(parents=True);torch.set_num_threads(1);source=json.loads((a.source/'manifest.json').read_text());cache=json.loads((a.cache/'manifest.json').read_text());records={r['key']:r for r in cache['records'] if r['mode']==a.mode};groups={s:[] for s in ['train','calibration','source_validation']}
 for row in source['files']:
  info=records[row['key']];p=a.cache/info['path'];assert sha256(p)==info['SHA256']
  with np.load(p) as f:factors=torch.tensor(f['log_factors'],dtype=torch.float64)
  groups[row['split']].append((row,factors))
 assert [len(groups[s]) for s in groups]==[32,16,32];model=SyndromePrior(a.seed);initial=parameter_sha(model);opt=torch.optim.Adam(model.parameters(),lr=.01);status=dict(status='training',mode=a.mode,seed=a.seed,updates=0,trainable_parameters=7,initial_parameter_SHA256=initial,source_manifest_SHA256=sha256(a.source/'manifest.json'),evidence_manifest_SHA256=sha256(a.cache/'manifest.json'),cached_training_query_equivalent_exposures=0,new_raw_series_forwards=0,target_arrays_opened=False,model_input_scope='only source-derived frozen log-factors, source labels/root only loss');atomic_json(a.output/'progress.json',status);initial_eval=evaluate(model,groups);atomic_json(a.output/'initial_source_validation.json',initial_eval)
 for index in np.random.RandomState(a.seed).permutation(32):
  row,factors=groups['train'][int(index)];opt.zero_grad(set_to_none=True);out=model(factors);y=torch.tensor([0.,0.,1.,1.,1.,1.,1.,1.],dtype=torch.float64)[:,None].expand_as(out['logit']);weights=torch.tensor([.25,.25,.125,.125,.0625,.0625,.0625,.0625],dtype=torch.float64);loss=(F.binary_cross_entropy_with_logits(out['logit'],y,reduction='none').mean(-1)*weights).sum();root_loss=-(out['root'][2:,:,row['root']].clamp_min(1e-300).log().mean(-1)*weights[2:]).sum()/.5/np.log(row['C']);loss=loss+.1*root_loss;assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters());opt.step();status['updates']+=1;status['cached_training_query_equivalent_exposures']+=704*row['C']*8
  with (a.output/'learning_curve.jsonl').open('a') as f:f.write(json.dumps(dict(update=status['updates'],system_key=row['key'],evidence_SHA256=records[row['key']]['SHA256'],loss=float(loss),root_loss=float(root_loss)))+'\n')
 assert status['updates']==32 and status['cached_training_query_equivalent_exposures']==2523136;final=evaluate(model,groups);atomic_json(a.output/'final_source_validation.json',final);torch.save(dict(model=model.state_dict(),mode=a.mode,seed=a.seed,source_threshold=final['source_threshold'],source_manifest_SHA256=status['source_manifest_SHA256']),a.output/'final.pth');status.update(status='complete',cached_evaluation_query_equivalents=initial_eval['cached_query_equivalent_evaluations']+final['cached_query_equivalent_evaluations'],final_parameter_SHA256=parameter_sha(model),checkpoint_SHA256=sha256(a.output/'final.pth'));atomic_json(a.output/'progress.json',status)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['source','cache','output']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--mode',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
 try:run(a)
 except Exception:
  a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
