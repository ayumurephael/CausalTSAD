"""112 new systems, disjoint source64/16/32; no benchmark arrays or training reuse."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,argparse,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.prepare_innovation_probe import build
from causal_tsad.io import atomic_json
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    jobs=[(c,i,str(a.output)) for c in [8,12,16,20] for i in range(100,128)]
    with concurrent.futures.ProcessPoolExecutor(8) as pool:rows=list(pool.map(build,jobs))
    for row,job in zip(rows,jobs):row['split']='train' if job[1]<116 else 'calibration' if job[1]<120 else 'source_validation'
    assert len(rows)==112 and len({v['key'] for v in rows})==112
    atomic_json(a.output/'manifest.json',dict(status='complete',files=rows,systems=112,split_systems=dict(train=64,calibration=16,source_validation=32),
        stored_input_slots=sum(v['stored_input_slots'] for v in rows),seed_namespace='348004000000+C*10000+i, i100..127; old diagnostic i0..7 disjoint',
        labels_detail=dict(effect=None,anomaly_channel=None,anomaly_time=None),benchmark_arrays_opened=False,full_retained_array_overlap_scan=False))
