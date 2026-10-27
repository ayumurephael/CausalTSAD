"""Original four target metrics; same first32 suffix and source-frozen scores."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,pickle,argparse,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from causal_tsad.io import atomic_json,sha256

def one(args):
    output,freeze_path,row=args;out=Path(output);freeze=json.loads(Path(freeze_path).read_text())
    assert row['split']=='development';path=out/row['system_id'];rec=json.loads(path.with_suffix('.json').read_text());assert sha256(path.with_suffix('.npz'))==rec['scores_SHA256']
    with (Path(freeze['benchmark'])/'data'/row['shard']).open('rb') as f:s=pickle.load(f)[row['offset']]
    assert s['system_id']==row['system_id'];y=np.asarray(s['labels'])[32:]
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
    p=out/'metrics'/(row['system_id']+'.json');assert not p.exists();atomic_json(p,dict(row=row,methods=methods,scores_SHA256=rec['scores_SHA256'],first=32))
    return dict(path=str(p),SHA256=sha256(p))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze',type=Path,required=True);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    freeze=json.loads(a.freeze.read_text());assert not freeze['locked_opened']
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(one,[(str(a.output),str(a.freeze),r) for r in freeze['development_rows']]))
    atomic_json(a.output/'metric_manifest.json',dict(status='complete',rows=rows,freeze_SHA256=sha256(a.freeze)))
