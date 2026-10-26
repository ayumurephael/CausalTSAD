"""Paired C04i source and generated-pilot evidence; seeds clustered within systems."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
SEEDS=[348,349,350]
def load(p):return json.loads(p.read_text(encoding='utf8'))
def mean(x):
 x=[float(v) for v in x if v is not None];return float(np.mean(x)) if x else None
def paired(d):
 d=np.asarray(d,dtype=float);rng=np.random.RandomState(348);ci=np.quantile(d[rng.randint(len(d),size=(10000,len(d)))].mean(1),[.025,.975]).tolist() if len(d) else None
 return dict(mean=float(d.mean()) if len(d) else None,CI95=ci,systems=len(d),positive=int((d>0).sum()),negative=int((d<0).sum()))
def source(root):
 old=root/'c04_bayesian_syndromes_v1';new=root/'c04_syndrome_covariance_controls_v1';runs={}
 for folder in [old,new]:
  assert (folder/'fetch_manifest.json').exists()
  for p in folder.glob('*/runs/*/final_source_validation.json'):
   status=load(p.parent/'progress.json');runs[(status['mode'],status['seed'])]=load(p)['records']
 modes=['true_signature','true_channelblock','true_diagonal','true_self','true_unit','wrong_signature','empty_signature'];keys=sorted(runs[('true_signature',348)]);summary={};pairs={}
 for mode in modes:
  assert all(sorted(runs[(mode,s)])==keys for s in SEEDS);rs=[runs[(mode,s)][k] for k in keys for s in SEEDS]
  summary[mode]=dict(AP={scope:mean([r['metrics'][scope]['AP'] for r in rs]) for scope in ['matched_measurement','paired_measurement','physical','all8']},normal_FPR=mean([r['normal_FPR'] for r in rs]),RCA={kind:{m:mean([e[m] for r in rs for e in r['RCA'] if e['kind']==kind]) for m in ['Top1','Top3','MRR']} for kind in ['matched_measurement','paired_measurement','physical']})
 for left,right in [('true_signature','true_channelblock'),('true_signature','true_diagonal'),('true_channelblock','true_diagonal')]:
  pairs[left+'_minus_'+right]={scope:paired([mean([runs[(left,s)][k]['metrics'][scope]['AP']-runs[(right,s)][k]['metrics'][scope]['AP'] for s in SEEDS]) for k in keys]) for scope in ['matched_measurement','physical']}
 out=dict(status='complete',independent_systems=32,optimization_seeds=3,summary=summary,paired=pairs,scope='source; fixed equal point variance, channel-block retains temporal marginals; not target performance')
 (new/'analysis.json').write_text(json.dumps(out,indent=2),encoding='utf8');return out
def target(root):
 folder=root/'c04_syndromes_generated_pilot_v1';assert (folder/'fetch_manifest.json').exists();rows=[];modes=['B0','true_signature','true_self','wrong_signature','empty_signature'];names=lambda mode:['B0'] if mode=='B0' else [mode+'_s'+str(s) for s in SEEDS];counts={k:0 for k in ['query_scalar_forwards','full_input_scalar_slots','Gaussian_covariance_factorizations','context_ridge_factorizations']}
 for node in [2226,2229]:
  plan=load(folder/str(node)/'job_plan.json');byid={r['id']:r for r in plan['rows']};manifest=load(folder/str(node)/'inference_manifest.json')
  for k in counts:counts[k]+=manifest[k]
  for p in (folder/str(node)/'development/metrics').glob('*.json'):
   rec=load(p);meta=byid[p.stem];inf=load(folder/str(node)/'development/inference'/p.name);assert rec['scores_SHA256']==inf['scores_SHA256'];assert inf['input_observed_SHA256']==meta['time_series_sha256'] and inf['input_normal_SHA256']==meta['normal_sha256'];methods={}
   for mode in modes:
    entries=[rec['methods'][n] for n in names(mode)];events=[q for n in names(mode) for q in rec['RCA'].get(n,[])];methods[mode]=dict(AP=mean([e['AP'] for e in entries]),VUS_PR=mean([e['metrics']['VUS-PR'] for e in entries]),ROC_AUC=mean([e['ROC_AUC'] for e in entries]),FPR=mean([e['normal_FPR_source99'] for e in entries]),Affiliation_F=mean([e['metrics']['Affiliation-F'] for e in entries]),F1_T=mean([e['metrics']['F1_T'] for e in entries]),Standard_F1=mean([e['metrics']['Standard-F1'] for e in entries]),conditional_RCA={k:mean([e[k] for e in events]) for k in ['Top1','Top3','MRR']},events=len(events)//len(names(mode)))
   if rec['C1_TimeRCD_passthrough']:
    assert all(rec['methods'][n]['score_array_SHA256']==rec['methods']['B0']['score_array_SHA256'] for mode in modes for n in names(mode))
   rows.append(dict(system_id=rec['system_id'],id=p.stem,node=node,track=rec['track'],C=rec['channels'],family=rec['family'],length=rec['length'],ready_points=rec['ready_points'],requested_strata=rec['requested_strata'],event_types=rec['event_types'],anomaly_fraction_full_sequence=meta['anomaly_fraction'],methods=methods,RCA_excluded=rec['RCA_excluded']))
 assert len(rows)==100 and len({r['system_id'] for r in rows})==100;assert counts['query_scalar_forwards']==31875968 and counts['full_input_scalar_slots']==34743168
 def group(rs):
  return dict(systems=len(rs),detection_eligible_systems=sum(r['methods']['B0']['AP'] is not None for r in rs),conditional_RCA_systems=sum(r['methods']['true_signature']['events']>0 for r in rs),conditional_RCA_events=sum(r['methods']['true_signature']['events'] for r in rs),methods={mode:dict(**{m:mean([r['methods'][mode][m] for r in rs]) for m in ['AP','VUS_PR','ROC_AUC','FPR','Affiliation_F','F1_T','Standard_F1']},conditional_RCA={m:mean([r['methods'][mode]['conditional_RCA'][m] for r in rs]) for m in ['Top1','Top3','MRR']}) for mode in modes})
 multi=[r for r in rows if r['C']>1];groups={'multivariate':group(multi),'univariate_passthrough':group([r for r in rows if r['C']==1])}
 for field in ['track','C','family']:
  groups[field]={str(v):group([r for r in multi if r[field]==v]) for v in sorted({r[field] for r in multi},key=str)}
 pairs={}
 for left,right in [('true_signature','B0'),('true_signature','true_self'),('true_signature','wrong_signature'),('true_signature','empty_signature')]:
  pairs[left+'_minus_'+right]={metric:paired([r['methods'][left][metric]-r['methods'][right][metric] for r in multi if r['methods'][left][metric] is not None and r['methods'][right][metric] is not None]) for metric in ['AP','VUS_PR','FPR']}
 bad=sorted([r for r in multi if r['methods']['B0']['AP'] is not None],key=lambda r:r['methods']['true_signature']['AP']-r['methods']['B0']['AP']);good=list(reversed(bad))
 out=dict(status='complete',independent_systems=100,multivariate_systems=80,optimization_seeds=3,first_query=320,groups=groups,paired=pairs,costs=counts,C1_score_exact_passthrough=True,scope='historical development100; no target updates, no TimeRCD RCA; event-window conditional RCA only',largest_detection_losses=[r['system_id'] for r in bad[:10]],largest_detection_gains=[r['system_id'] for r in good[:10]],positive_vs_B0=sum(r['methods']['true_signature']['AP']>r['methods']['B0']['AP'] for r in bad),case_file='paired_cases.json')
 (folder/'analysis.json').write_text(json.dumps(out,indent=2),encoding='utf8');(folder/'paired_cases.json').write_text(json.dumps(rows,indent=2),encoding='utf8');return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results-root',type=Path,required=True);a=p.parse_args();s=source(a.results_root);t=target(a.results_root);print(json.dumps(dict(source=s['summary'],source_paired=s['paired'],target=t['groups'],target_paired=t['paired'],target_positive_vs_B0=t['positive_vs_B0']),ensure_ascii=False))
