"""Isolated oracle lag audit. These metadata never enter the model loader."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256,array_sha


def replay_one(job):
    generator,task=job;sys.path.insert(0,generator)
    from rcd_schema import make_sample
    s=make_sample(1024,task['c'],task['seed'],task['ident'],permute=True,activate=False)
    lags=np.asarray(s['mechanism']['edge_lag']);times=np.arange(64,1024);start=times//16*16
    counts={k:0 for k in ['block','past','current']};total=len(lags)*len(times)
    for lag in lags:
        required=times-int(lag)
        counts['block']+=int(((required>=start-64)&(required<start)).sum())
        counts['past']+=int(((required>=times-64)&(required<times)).sum())
        counts['current']+=int(((required>=times-63)&(required<=times)).sum())
    return dict(key=task['ident'],channels=task['c'],edges=len(lags),lag_histogram={str(int(v)):int((lags==v).sum()) for v in np.unique(lags)},
        exact_required_parent_observation_available=counts,edge_time_pairs=total,
        original_normal_sha=array_sha(s['normal_time_series']),original_observed_sha=array_sha(s['time_series']),
        extra_CPU_generated_scalars_per_view=int(s['time_series'].size))


def main(a):
    config=a.generator.parent/'config/synthetic.json';os.environ['CHATTS_CONFIG_PATH']=str(config)
    manifest=json.loads((a.source/'manifest.json').read_text());plan=json.loads((a.source/'plan.json').read_text())
    assert sha256(a.source/'manifest.json')=='bd24005add8469c106b89ee69a68bf1f6a20e7070d4dcf7f6a4e3b74dd4c2ac6'
    for name,value in manifest['generator_sha'].items():assert sha256(a.generator/name)==value
    selected=[]
    for c in [8,12,16,20]:
        group=[t for t in plan if t['split']=='source_validation' and t['c']==c]
        selected+=sorted(group,key=lambda t:hashlib.sha256(t['ident'].encode()).hexdigest())[:8]
    lookup={r['ident']:r for r in manifest['records']}
    with ProcessPoolExecutor(max_workers=8) as pool:rows=list(pool.map(replay_one,[(str(a.generator),t) for t in selected]))
    for row in rows:
        assert row['original_normal_sha']==lookup[row['key']]['original_normal_sha']
        assert row['original_observed_sha']==lookup[row['key']]['original_x_sha']
    totals={k:sum(r['exact_required_parent_observation_available'][k] for r in rows) for k in ['block','past','current']}
    denominator=sum(r['edge_time_pairs'] for r in rows)
    hist={}
    for r in rows:
        for k,v in r['lag_histogram'].items():hist[k]=hist.get(k,0)+v
    atomic_json(a.output,dict(status='passed',systems=len(rows),selection='label-free SHA first8 source-validation systems per C>1',
        lag_histogram=hist,edge_time_pairs=denominator,availability_fraction={k:v/denominator for k,v in totals.items()},records=rows,
        extra_CPU_generation_scalars_per_view=sum(r['extra_CPU_generated_scalars_per_view'] for r in rows),
        pinned_generator_config_sha256=sha256(config),source_manifest_sha256=sha256(a.source/'manifest.json'),
        model_input_metadata=False,target_arrays_opened=False,
        limitation='Availability of exact structural parent value, not a proof of detection improvement or predictive information. Autocorrelation may substitute for a missing value.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--generator',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
