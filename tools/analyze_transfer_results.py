"""SHA-checked, system-paired analysis of all frozen I17 development outputs."""
from pathlib import Path
from collections import defaultdict
import argparse,json,hashlib,re
import numpy as np

METRICS=['AP','PR_AUC','ROC_AUC','Affiliation-F','F1_T','Standard-F1','VUS-PR',
 'normal_FPR','normal_alarm_events','conditional_Top1','conditional_MRR',
 'joint_detected_root_recall','event_recall','unmatched_alarm_events']
def mean(rows,metric):
 d=defaultdict(list)
 for r in rows:
  if r.get(metric) is not None:d[r['system_id']].append(r[metric])
 return {k:float(np.mean(v)) for k,v in d.items()}
def summary(rows):
 return {k:dict(mean=float(np.mean(list(x.values()))),systems=len(x),p95=float(np.quantile(list(x.values()),.95)))
         for k in METRICS if (x:=mean(rows,k))}
def paired(a,b,k):
 aa,bb=mean(a,k),mean(b,k);keys=sorted(aa.keys()&bb.keys());v=np.array([aa[x]-bb[x] for x in keys]);rng=np.random.default_rng(348)
 if not len(v):return None
 ci=np.quantile(v[rng.integers(0,len(v),(10000,len(v)))].mean(1),[.025,.975]);p=(rng.choice([-1,1],(20000,len(v)))*v).mean(1)
 return dict(mean=float(v.mean()),CI95=ci.tolist(),systems=len(v),p_signflip=float((1+(abs(p)>=abs(v.mean())).sum())/20001),
   up=int((v>0).sum()),down=int((v<0).sum()),all_system_differences=dict(zip(keys,v.tolist())))
def main(a):
 output=a.root/'analysis_v2.json';assert not output.exists()
 rows=defaultdict(list);inputs={};budget=defaultdict(int);ids=set();parameter_hashes=set();records=0
 for port in ['2226','2229']:
  folder=a.root/port;col=json.loads((folder/'collection.json').read_text());assert col['remote_point_arrays_SHA_verified']
  for f in col['files']:
   path=folder/f['relative'];assert hashlib.sha256(path.read_bytes()).hexdigest()==f['SHA256'];inputs[str(path.relative_to(a.root))]=f['SHA256']
  freeze_path=folder/'freeze.json';fr=json.loads(freeze_path.read_text());fh=hashlib.sha256(freeze_path.read_bytes()).hexdigest()
  for idx in fr['task_indices']:
   row=fr['development_rows'][idx];key=row['system_id'];assert key not in ids;ids.add(key)
   path=folder/'output'/(key+'.json');d=json.loads(path.read_text());m=json.loads((folder/'output/metrics'/(key+'.json')).read_text())
   assert d['source_freeze_SHA256']==m['freeze_SHA256']==fh and d['scores_SHA256']==m['scores_SHA256']
   assert d['global_updates']==0 and not d['locked_opened'] and not d['TimeRCD_RCA'] and d['first']==32
   parameter_hashes.add(d['basis_parameter_SHA256']);records+=1
   for bb in d['budgets']:
    for k,v in bb.items():budget[k]+=v
   for k in ['detector_input_scalars','detector_padded_scalars','detector_calls']:budget[k]+=d[k]
   budget['unique_observed_scalars']+=row['length']*row['channels']
   normal={r['method']:r for r in d['metrics'] if r['view']==1};rca={r['method']:r for r in d['conditional_and_detected_RCA']}
   for r in d['metrics']:
    if r['view']!=0:continue
    method=r['method'];arm=re.sub(r'_s[0-9]+$','',method)
    x={k:v for k,v in r.items() if k not in ['events']};x.update(m['methods'][method])
    x.update(system_id=key,track=row['track'],C=row['channels'],family=row['family'],event_types=row['event_types'],
      normal_FPR=normal[method]['FPR'],normal_alarm_events=normal[method]['false_alarm_events'])
    if method in rca:x.update({k:v for k,v in rca[method].items() if k not in ['events','excluded','method']})
    rows[arm].append(x)
 assert records==432 and len(parameter_hashes)==1 and budget['unique_observed_scalars']==24576624
 summaries={k:summary(v) for k,v in rows.items()};strata={}
 for arm,rr in rows.items():
  gg=defaultdict(list)
  for r in rr:
   for k in ['track','C','family']:gg[k+'='+str(r[k])].append(r)
   for k in r['event_types']:gg['event='+k].append(r)
  strata[arm]={k:summary(v) for k,v in gg.items()}
 tests={}
 for left,right in [('stat_temporal','stat_pair'),('stat_temporal','node_scan'),('stat_temporal','raw'),('protected','raw_adapt'),('protected','raw'),('concat','raw')]:
  for metric in ['AP','VUS-PR','normal_FPR','conditional_Top1','joint_detected_root_recall']:
   t=paired(rows[left],rows[right],metric)
   if t:tests[left+'-'+right+'::'+metric]=t
 report=dict(status='complete',systems=records,files_SHA_verified=len(inputs),summary=summaries,strata=strata,paired_tests=tests,
   actual_budget=dict(budget),global_updates=0,locked_opened=False,TimeRCD_RCA=False,all_rows=rows,input_SHA256=inputs,
   limitations=['Development has been explored; not a fresh blind test','Three optimization seeds averaged within each system',
    'Event-type groups may overlap for multi-event systems','RCA uses generated event annotations; AP time labels follow original policy',
    'Failed partial first-run forward compute separately recorded in recovery; not zero'])
 report['analysis_revision']='v2: only strip terminal numeric optimization seed; retain edge_scan/node_scan names'
 output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
 md='# I17：432多变量开发系统完整结果\n\n每系统先平均三个优化种子；t≥32，正常case独立拟合，阈值来源冻结。\n\n| 方法 | AP | VUS-PR | 正常FPR | 条件Top1 | 联合根因召回 | 正常报警段/例 |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for k,s in summaries.items():md+='| '+k+' | '+' | '.join(f"{100*s[m]['mean']:.3f}%" if m in s else '—' for m in ['AP','VUS-PR','normal_FPR','conditional_Top1','joint_detected_root_recall'])+f" | {s['normal_alarm_events']['mean']:.2f} |\n"
 md+='\n原四指标、两轨、C、故障、全部系统和负例见analysis_v2.json。不能与不同后缀或来源表直接相减。初版分析误将_scan当seed后缀，保留初版，不重跑推断。\n'
 (a.root/'完整结果_v2.md').write_text(md,encoding='utf8')
 print(json.dumps(dict(systems=records,files=len(inputs),summary=summaries,
  tests={k:{i:t[i] for i in ['mean','CI95','up','down']} for k,t in tests.items()}),ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args())
