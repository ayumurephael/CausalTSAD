"""Same-source calibration, clustered paired analysis, and fault-absorption audit."""
from pathlib import Path
import argparse,json,hashlib,ast
import numpy as np
from sklearn.metrics import average_precision_score
SEEDS=['348','349','350'];VARIANTS=['fixed','rolling','student'];MODES=['true_signature','empty_signature']
def read(p):return json.loads(p.read_text(encoding='utf8'))
def mean(xs):
 xs=[float(x) for x in xs if x is not None];return float(np.mean(xs)) if xs else None
def delta(xs):
 x=np.asarray(xs,float)
 if not len(x):return dict(mean=None,CI95=None,n=0)
 rng=np.random.RandomState(348);boot=x[rng.randint(len(x),size=(10000,len(x)))].mean(1);return dict(mean=float(x.mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),n=len(x),positive=int((x>0).sum()),negative=int((x<0).sum()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(a):
 r=a.results_root/'c04_context_likelihood_controls_v1';assert (r/'fetch_manifest.json').exists() and (r/'baseline_extract_receipts.json').exists();source={};targets={};costs={};source_paths={}
 for port in ['2226','2229']:
  root=r/port;m=read(root/'manifest.json');assert m['status']=='complete'
  for field in ['query_scalar_forwards','full_input_scalar_slots','Gaussian_covariance_factorizations','context_ridge_factorizations','cached_prior_query_applications','extra_failed_prior_query_applications']:costs[field]=costs.get(field,0)+m[field]
  for item in m['files']:
   p=root/item['path'];assert sha(p)==item['SHA256'];v=read(p);assert sha(p.with_suffix('.npz'))==v['scores_SHA256'];key=(v['variant'],v['mode'],v.get('background'),v.get('key',v.get('id')))
   with np.load(p.with_suffix('.npz')) as f:scores=f['scores'].copy()
   if item['path'].startswith('source/'):
    assert v['matched_quartet_exact'];source[key]=(v,scores);source_paths[key]=str(p)
   else:targets[(v['variant'],v['mode'],v['id'])]=(v,scores)
 assert len(source)==576 and len(targets)==320;assert costs['query_scalar_forwards']==69723008 and costs['cached_prior_query_applications']==226200192;assert costs['extra_failed_prior_query_applications']==236544
 # Every method uses precisely the same16 source-calibration systems and two
 # backgrounds. Only two normal views are allowed into threshold computation.
 thresholds={};calibration_audit={}
 for variant in VARIANTS:
  for mode in MODES:
   pairs=[(k,v,s) for k,(v,s) in source.items() if v['variant']==variant and v['mode']==mode and v['split']=='calibration'];assert len(pairs)==32 and len({v['key'] for _,v,_ in pairs})==16
   arrays=[s[:,:2].reshape(3,-1) for _,_,s in pairs];pool=np.concatenate(arrays,axis=1);assert pool.shape==(3,45056);thresholds[(variant,mode)]=np.quantile(pool,.99,axis=1);calibration_audit[variant+'/'+mode]=dict(independent_systems=16,normal_views=2,backgrounds=2,per_seed_score_count=45056,source_files=[source_paths[k] for k,_,_ in pairs],normal_pool_SHA256=hashlib.sha256(pool.tobytes()).hexdigest(),thresholds={seed:float(thresholds[(variant,mode)][si]) for si,seed in enumerate(SEEDS)})
 source_rows=[]
 for key,(v,scores) in source.items():
  if v['split']!='source_validation':continue
  assert scores.shape==(3,8,704);stats={}
  for scope,views in [('matched_measurement',[0,1,2,3]),('paired_measurement',[0,1,4,5]),('physical',[0,1,6,7])]:stats[scope]=mean([v['methods'][seed]['metrics'][scope]['AP'] for seed in SEEDS])
  def ap(window):return mean([average_precision_score(np.repeat([0,0,1,1],scores[si,[0,1,2,3],window].shape[-1]),scores[si,[0,1,2,3],window].reshape(-1)) for si in range(3)])
  source_rows.append(dict(key=v['key'],variant=v['variant'],mode=v['mode'],background=v['background'],C=v['C'],family=v['family'],AP=stats,FPR=mean([(scores[si,:2]>thresholds[(v['variant'],v['mode'])][si]).mean() for si in range(3)]),early_AP=ap(slice(0,128)),late_AP=ap(slice(-128,None)),RCA={kind:{metric:mean([e[metric] for seed in SEEDS for e in v['methods'][seed]['RCA'] if e['kind']==kind]) for metric in ['Top1','Top3','MRR']} for kind in ['matched_measurement','paired_measurement','physical']},normal_prediction_stats=v['normal_prediction_stats']))
 def source_group(rows):return dict(independent_systems=len(rows),AP={scope:mean([v['AP'][scope] for v in rows]) for scope in ['matched_measurement','paired_measurement','physical']},normal_FPR=mean([v['FPR'] for v in rows]),early_AP=mean([v['early_AP'] for v in rows]),late_AP=mean([v['late_AP'] for v in rows]),conditional_RCA={k:mean([v['RCA']['matched_measurement'][k] for v in rows]) for k in ['Top1','Top3','MRR']})
 source_summary={bg:{mode:{variant:source_group([v for v in source_rows if v['background']==bg and v['mode']==mode and v['variant']==variant]) for variant in VARIANTS} for mode in MODES} for bg in ['plain','smooth']};lookup={(v['variant'],v['mode'],v['background'],v['key']):v for v in source_rows};source_pairs={}
 for bg in ['plain','smooth']:
  for mode in MODES:
   for variant in ['rolling','student']:
    rows=[v for v in source_rows if v['background']==bg and v['mode']==mode and v['variant']==variant];d=[(v,lookup[('fixed',mode,bg,v['key'])]) for v in rows];source_pairs[bg+'/'+mode+'/'+variant]=dict(AP=delta([x['AP']['matched_measurement']-y['AP']['matched_measurement'] for x,y in d]),FPR=delta([x['FPR']-y['FPR'] for x,y in d]),late_minus_early_difference=delta([(x['late_AP']-x['early_AP'])-(y['late_AP']-y['early_AP']) for x,y in d]))
 oldroot=a.results_root/'c04_syndromes_generated_pilot_v1';oldrows={};metadata={};fixed={}
 for port in ['2226','2229']:
  plan=read(r/port/'job_plan.json');metadata.update({x['id']:x for x in plan['target_rows']})
  for row in plan['target_rows']:
   ident=row['id'];old=read(oldroot/port/'development/metrics'/(ident+'.json'));assert old['channels']>1;oldrows[ident]=old
   with np.load(r/port/'fixed_baseline_extract'/(ident+'.npz')) as f:fixed[ident]={k:f[k].copy() for k in f.files}
 assert len(metadata)==80
 target_rows=[]
 for ident,meta in metadata.items():
  for mode in MODES:
   for variant in VARIANTS:
    if variant=='fixed':
     ss=fixed[ident][mode];old=oldrows[ident];entries=[old['methods'][mode+'_s'+seed] for seed in SEEDS];events=[e for seed in SEEDS for e in old['RCA'].get(mode+'_s'+seed,[])];ap=mean([e['AP'] for e in entries]);auc=mean([e['ROC_AUC'] for e in entries]);prediction=targets[('student',mode,ident)][0]['normal_prediction_stats'];provenance='identical normal-predictor AST with Student branch; only likelihood differs'
    else:
     v,ss=targets[(variant,mode,ident)];assert v['input_observed_SHA256']==meta['time_series_sha256'] and v['input_normal_SHA256']==meta['normal_sha256'];entries=[v['methods'][seed] for seed in SEEDS];ap=mean([e['AP'] for e in entries]);auc=mean([e['ROC_AUC'] for e in entries]);events=[q for e in entries for q in e['RCA']];prediction=v['normal_prediction_stats'];provenance='own normal forward'
    assert ss.shape==(3,2,meta['length']-320);fpr=mean([(ss[si,1]>thresholds[(variant,mode)][si]).mean() for si in range(3)]);target_rows.append(dict(system_id=meta['system_id'],id=ident,C=meta['channels'],track=meta['track'],family=meta['family'],length=meta['length'],requested_strata=meta['requested_strata'],variant=variant,mode=mode,AP=ap,ROC_AUC=auc,FPR=fpr,RCA={m:mean([e[m] for e in events]) for m in ['Top1','Top3','MRR']},events=len(events)//3,normal_prediction_stats=prediction,prediction_provenance=provenance))
 def target_group(rows):return dict(systems=len(rows),AP_eligible=sum(v['AP'] is not None for v in rows),AP=mean([v['AP'] for v in rows]),ROC_AUC=mean([v['ROC_AUC'] for v in rows]),FPR=mean([v['FPR'] for v in rows]),conditional_RCA={m:mean([v['RCA'][m] for v in rows]) for m in ['Top1','Top3','MRR']},median_normal_whitened_square=float(np.median([v['normal_prediction_stats']['mean_whitened_square'] for v in rows])),median_normal_q99=float(np.median([v['normal_prediction_stats']['whitened_squared_quantiles'][2] for v in rows])))
 target_summary={mode:{variant:target_group([v for v in target_rows if v['mode']==mode and v['variant']==variant]) for variant in VARIANTS} for mode in MODES};target_by_track={track:{variant:target_group([v for v in target_rows if v['track']==track and v['mode']=='true_signature' and v['variant']==variant]) for variant in VARIANTS} for track in ['id','mechanism_holdout']};byid={(v['variant'],v['mode'],v['id']):v for v in target_rows};target_pairs={}
 for mode in MODES:
  for variant in ['rolling','student']:
   pairs=[(byid[(variant,mode,i)],byid[('fixed',mode,i)]) for i in metadata];target_pairs[mode+'/'+variant]={metric:delta([x[metric]-y[metric] for x,y in pairs if x[metric] is not None and y[metric] is not None]) for metric in ['AP','FPR']}
 # Verify the global normal predictor is actually unchanged before sharing audit stats.
 context=a.repository_root/'worktrees/c04contextdiag/proposals/causal_syndromes.py';student=a.repository_root/'worktrees/c04robustlikelihood/proposals/causal_syndromes.py'
 for name in ['_features','condition','innovations']:
  functions=[next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name) for p in [context,student]];assert ast.dump(functions[0],include_attributes=False)==ast.dump(functions[1],include_attributes=False)
 result=dict(status='complete',source_validation_independent_systems=32,source_calibration_independent_systems=16,target_independent_systems=80,optimization_seeds=3,source=source_summary,source_paired=source_pairs,target=target_summary,target_by_track=target_by_track,target_paired=target_pairs,costs=costs,normal_predictor_identical_AST_verified=True,calibration_scope='only same16sourcecal x2background x2normal views, no target/validation calibration',target_scope='previously explored development80; conditional RCA; no new four-metric performance evaluation',target_updates=0,optimizer_updates=0)
 for name,value in [('analysis.json',result),('source_cases.json',source_rows),('target_cases.json',target_rows),('calibration.json',calibration_audit)]:
  p=r/name;assert not p.exists();p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(dict(source=source_summary,source_paired=source_pairs,target=target_summary,target_by_track=target_by_track,target_paired=target_pairs,costs=costs),ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results-root',type=Path,required=True);p.add_argument('--repository-root',type=Path,required=True);run(p.parse_args())
