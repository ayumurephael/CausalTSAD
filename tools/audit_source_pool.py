"""Replay every new source system with an explicitly pinned generator config."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256
from tools.prepare_source_pool import generate


def main(a):
    config=a.generator_root.parent/'config/synthetic.json'
    assert config.exists();os.environ['CHATTS_CONFIG_PATH']=str(config)
    manifest=json.loads((a.source/'manifest.json').read_text());plan=json.loads((a.source/'plan.json').read_text())
    lookup={r['ident']:r for r in manifest['records']}
    files={}
    for row in manifest['files']:
        assert sha256(a.source/row['path'])==row['SHA256']
        with np.load(a.source/row['path']) as z:files[(row['split'],row['channels'])]={k:np.array(z[k]) for k in z.files}
    offsets={};done=0
    args=[(str(a.generator_root),r['c'],r['seed'],r['ident']) for r in plan]
    with ProcessPoolExecutor(max_workers=16) as pool:
        for task,result in zip(plan,pool.map(generate,args,chunksize=2)):
            prior=lookup[task['ident']]
            assert prior['original_x_sha']==result['original_x_sha']
            assert prior['original_normal_sha']==result['original_normal_sha']
            key=(task['split'],task['c']);i=offsets.get(key,0);offsets[key]=i+1
            for name in ('x','normal','labels','adjacency','root','root_window','seed'):
                assert np.array_equal(files[key][name][i],result[name]),(task['ident'],name)
            done+=1
    report=dict(status='passed',systems=done,source_manifest_sha256=sha256(a.source/'manifest.json'),
        pinned_generator_config_sha256=sha256(config),all_full_float64_branches_replayed=True,
        all_saved_float32_inputs_labels_graphs_roots_identical=True,extra_CPU_generated_observed_scalars=
        manifest['training_original_observed_scalars']+manifest['source_validation_original_observed_scalars'],
        target_arrays_opened=False)
    atomic_json(a.output,report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--generator-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
