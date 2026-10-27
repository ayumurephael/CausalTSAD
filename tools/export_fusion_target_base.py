"""Export an immutable exact subset of existing B0 arrays; no target model forwards."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from causal_tsad.io import atomic_json,sha256
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 plan=json.loads((a.baseline/'job_plan.json').read_text());rows=[]
 for row in plan['rows']:
  assert row['split']=='development';ident=row['id'];meta=json.loads((a.baseline/'inference'/(ident+'.json')).read_text());src=a.baseline/'scores'/(ident+'.npz');assert sha256(src)==meta['scores_SHA256']
  dst=a.output/'arrays'/(ident+'.npz');dst.parent.mkdir(exist_ok=True)
  with np.load(src) as f:np.savez_compressed(dst,base=np.array(f['selftrained__native__observed__logit']),normal=np.array(f['selftrained__native__normal__logit']))
  metrics=json.loads((a.baseline/'metrics'/(ident+'.json')).read_text());assert metrics['system_id']==row['system_id']
  rows.append(dict(row,path=str(dst.relative_to(a.output)),SHA256=sha256(dst),original_full_scores_SHA256=meta['scores_SHA256'],original_metrics=metrics['methods']['selftrained/native/logit'],period=metrics['periods']['native']))
 atomic_json(a.output/'manifest.json',dict(status='complete',rows=rows,source_baseline_plan_SHA256=sha256(a.baseline/'job_plan.json'),original_source_thresholds=json.loads((a.baseline/'calibration.json').read_text())['thresholds'],encoder_forwards=0,optimizer_updates=0,representation='native/logit',RCA_evaluated=False))
 print(json.dumps(dict(status='complete',systems=len(rows),manifest_SHA256=sha256(a.output/'manifest.json'))))
