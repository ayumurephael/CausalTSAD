"""Original four metrics on predeclared primary conditions; no re-inference."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,pickle,argparse,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from causal_tsad.io import atomic_json,sha256
def selected(tag):
    return (tag.startswith('E4_') or tag.startswith('prefix__') or
            tag.startswith('online__') or
            (tag.startswith(('xfit__','period_parent32__')) and tag.rsplit('__',1)[1] in ['point','center_case','center_channel']))
def one(args):
    root,plan,task=args;root=Path(root);row=task['row'];ident=task['id'];p=root/'records'/ident
    meta=json.loads(p.with_suffix('.json').read_text());assert sha256(p.with_suffix('.npz'))==meta['scores_SHA256']
    with (Path(plan['benchmark'])/'data'/row['shard']).open('rb') as f:s=pickle.load(f)[row['offset']]
    y=np.asarray(s['labels'])[320:];sys.path.insert(0,plan['metric_code']);from evaluate_table1 import compute_metrics
    result={};memo={}
    with np.load(p.with_suffix('.npz')) as arrays,(root/'logs'/('metrics_'+ident+'.log')).open('x') as log,redirect_stdout(log),redirect_stderr(log):
        for tag in arrays.files:
            if not selected(tag):continue
            score=arrays[tag][0];key=score.tobytes()
            if key not in memo:
                raw=compute_metrics(score,y,row['period']) if y.any() and not y.all() else {}
                memo[key]={k:float(raw[k]) if k in raw else None for k in ['Affiliation-F','F1_T','Standard-F1','VUS-PR']}
            result[tag]=memo[key]
    dst=root/'metrics'/(ident+'.json');atomic_json(dst,dict(task=task,methods=result,scores_SHA256=meta['scores_SHA256']))
    return dict(path=str(dst.relative_to(root)),SHA256=sha256(dst))
def main(a):
    plan=json.loads(a.plan.read_text());tasks=[t for t in plan['tasks'] if t['row']['kind']=='target'];records=[];failed=[]
    with ProcessPoolExecutor(max_workers=12) as pool:
        jobs={pool.submit(one,(str(a.root),plan,t)):t for t in tasks}
        for f in as_completed(jobs):
            t=jobs[f]
            try:records.append(f.result())
            except Exception:failed.append(t['id']);atomic_json(a.root/'errors'/('metric_'+t['id']+'.json'),dict(error=traceback.format_exc()))
            atomic_json(a.root/'metric_progress.json',dict(status='running',completed=len(records),total=len(tasks),failed=failed))
    atomic_json(a.root/'metric_manifest.json',dict(status='needs_attention' if failed else 'complete',records=records))
    atomic_json(a.root/'metric_progress.json',dict(status='needs_attention' if failed else 'complete',completed=len(records),total=len(tasks),failed=failed));assert not failed
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);main(p.parse_args())
