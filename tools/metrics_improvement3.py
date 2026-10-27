"""Original four target metrics; same first32 suffix and source-frozen scores."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,pickle,argparse,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from causal_tsad.io import atomic_json,sha256,array_sha

def one(args):
    output,freeze_path,row=args;out=Path(output);freeze=json.loads(Path(freeze_path).read_text())
    assert row['split']=='development';path=out/row['system_id'];rec=json.loads(path.with_suffix('.json').read_text());assert sha256(path.with_suffix('.npz'))==rec['scores_SHA256']
    assert rec['source_freeze_SHA256']==sha256(freeze_path)
    with (Path(freeze['benchmark'])/'data'/row['shard']).open('rb') as f:s=pickle.load(f)[row['offset']]
    assert s['system_id']==row['system_id'] and array_sha(s['labels'])==row['labels_sha256'];y=np.asarray(s['labels'])[32:]
    assert sha256(Path(freeze['metric_code'])/'evaluate_table1.py')==freeze['metric_entry_SHA256']
    sys.path.insert(0,freeze['metric_code']);from evaluate_table1 import compute_metrics
    methods={};memo={}
    with np.load(path.with_suffix('.npz')) as f:
        for k in f.files:
            if not k.endswith('__0'):continue
            v=f[k];h=hashlib.sha256(v.tobytes()).hexdigest()
            if h not in memo:
                raw=compute_metrics(v,y,row['period']) if y.any() and not y.all() else {}
                memo[h]={name:float(raw[name]) if name in raw else None for name in ['Affiliation-F','F1_T','Standard-F1','VUS-PR']}
            methods[k[:-3]]=memo[h]
    p=out/'metrics'/(row['system_id']+'.json');assert not p.exists();atomic_json(p,dict(status='complete',row=row,methods=methods,scores_SHA256=rec['scores_SHA256'],freeze_SHA256=sha256(freeze_path),first=32))
    return dict(path=str(p),SHA256=sha256(p))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze',type=Path,required=True);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    freeze=json.loads(a.freeze.read_text());assert not freeze['locked_opened']
    assigned=[freeze['development_rows'][i] for i in freeze['task_indices']];rows=[]
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            jobs=[pool.submit(one,(str(a.output),str(a.freeze),r)) for r in assigned]
            for fut in as_completed(jobs):
                rows.append(fut.result());atomic_json(a.output/'metric_progress.json',dict(status='running',completed=len(rows),total=len(assigned)))
        atomic_json(a.output/'metric_manifest.json',dict(status='complete',rows=rows,freeze_SHA256=sha256(a.freeze)))
        atomic_json(a.output/'metric_progress.json',dict(status='complete',completed=len(rows),total=len(assigned)))
    except Exception:
        import traceback
        atomic_json(a.output/'metric_progress.json',dict(status='needs_attention',completed=len(rows),total=len(assigned),error=traceback.format_exc()));raise
