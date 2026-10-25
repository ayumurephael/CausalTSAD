"""Only independent source pool files are accepted. No benchmark path option."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256
from proposals.c03.data import encode


def prepare(job):
    source,dest,row=job;path=Path(source)/row['path'];assert sha256(path)==row['SHA256']
    with np.load(path) as f:pack={k:np.array(f[k]) for k in f.files}
    fields={};counts={}
    for view,key in [('x','x'),('normal','normal')]:
        encoded=[encode(v) for v in pack[key]]
        for field in encoded[0][0]:fields[view+'_'+field]=np.stack([v[0][field] for v in encoded])
        counts[view]={k:sum(v[1][k] for v in encoded) for k in encoded[0][1]}
    fields.update(graph=pack['adjacency'],labels=pack['labels'].reshape(len(pack['labels']),64,16))
    target=Path(dest)/row['path']
    with target.open('xb') as f:np.savez_compressed(f,**fields)
    return dict(row,path=target.name,SHA256=sha256(target),source_file_SHA256=row['SHA256'],counts=counts)


def main(a):
    assert sha256(a.source/'manifest.json')=='bd24005add8469c106b89ee69a68bf1f6a20e7070d4dcf7f6a4e3b74dd4c2ac6'
    m=json.loads((a.source/'manifest.json').read_text());assert m['status']=='complete'
    assert not a.output.exists();a.output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        files=list(pool.map(prepare,[(str(a.source),str(a.output),r) for r in m['files']]))
    atomic_json(a.output/'manifest.json',dict(status='complete',files=files,source_manifest_sha256=sha256(a.source/'manifest.json'),
        source_only=True,target_arrays_opened=False,normal_training_query_scalars=4976640,source_validation_scalars=1968128,
        observation_train_gradient_scalars=0,normal_twin_not_a_target_input=True,
        context='past192 ending64before query, per-edge correlations [0,1,2,4,8], per-channel auto/coverage, 3 chunks of64',
        generation_or_encoder_repeats=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
