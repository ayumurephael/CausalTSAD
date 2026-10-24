"""Independent original-mechanism source pool; never train on benchmark arrays.

Requires the user's versioned replayable generator via --generator-root. Its
source hashes are recorded; no generator parameters enter published model inputs.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import adjacency, array_sha, atomic_json, sha256


def generate(task):
    generator, c, seed, ident = task
    sys.path.insert(0, generator)
    from rcd_schema import make_sample, replay
    s = make_sample(1024, c, seed, ident, permute=True, activate=False)
    x = s['time_series']; n = s['normal_time_series']
    assert all(s['labels_detail'][k] is None for k in ('effect', 'anomaly_channel', 'anomaly_time'))
    # Replay a deterministic subset, retaining original full float64 array hashes.
    if seed % 23 == 0:
        rebuilt = replay(s)
        assert np.allclose(rebuilt[0], n, rtol=0, atol=1e-10)
        assert np.allclose(rebuilt[1], x, rtol=0, atol=1e-10)
    event_mask = np.zeros(1024, np.int8)
    for e in s['events']:
        if e['applied']:
            for start, end in e['annotation_intervals']: event_mask[start:end] = 1
    return dict(x=x.astype('float32'), normal=n.astype('float32'), labels=s['labels'].astype('int8'),
                adjacency=adjacency(s['graph'], c), root=int(np.flatnonzero(s['labels_detail']['root_channel'])[0]),
                root_window=event_mask, seed=seed, ident=ident, original_x_sha=array_sha(x), original_normal_sha=array_sha(n))


def main(a):
    assert not a.output.exists(), 'A source pool is immutable; choose a new revision'
    forbidden = set(np.frombuffer(a.forbidden_seeds.read_bytes(), dtype='<u4').tolist())
    benchmark = json.loads(a.benchmark_index.read_text())
    hashes = {r[k] for r in benchmark for k in ('time_series_sha256', 'normal_sha256')}
    used = set(forbidden); plan = []
    for split, budget in [('train', 5000000), ('source_validation', 2000000)]:
        for c in [1, 8, 12, 16, 20]:
            count = (budget // 5) // (1024 * c)
            for i in range(count):
                ident = f'causal-b1-v1/{split}/c{c}/{i}'
                nonce = 0
                while True:
                    seed = int.from_bytes(hashlib.sha256(f'{ident}/{nonce}'.encode()).digest()[:4], 'little')
                    if seed not in used: break
                    nonce += 1
                used.add(seed); plan.append(dict(split=split, c=c, seed=seed, ident=ident))
    a.output.mkdir(parents=True)
    generator_hashes = {p.name: sha256(p) for p in sorted(a.generator_root.glob('*.py'))}
    atomic_json(a.output/'plan.json', plan)
    atomic_json(a.output/'preparation.json', dict(status='running', total=len(plan), completed=0))
    rows = []; packs = {}; starts = {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        args = [(str(a.generator_root), r['c'], r['seed'], r['ident']) for r in plan]
        for task, s in zip(plan, pool.map(generate, args, chunksize=2)):
            assert s['original_x_sha'] not in hashes and s['original_normal_sha'] not in hashes
            key = (task['split'], task['c']); packs.setdefault(key, []).append(s)
            if len(rows) % 50 == 0:
                atomic_json(a.output/'preparation.json', dict(status='running', total=len(plan), completed=len(rows)))
            rows.append({k: s[k] for k in ('ident', 'seed', 'original_x_sha', 'original_normal_sha')})
    files = []
    for (split, c), records in packs.items():
        p = a.output / f'{split}_c{c:02d}.npz'
        np.savez_compressed(p, **{k: np.stack([r[k] for r in records]) for k in ('x','normal','labels','adjacency','root','root_window','seed')})
        files.append(dict(path=p.name, split=split, channels=c, systems=len(records), scalars=len(records)*1024*c, SHA256=sha256(p)))
    train = sum(r['scalars'] for r in files if r['split']=='train')
    val = sum(r['scalars'] for r in files if r['split']=='source_validation')
    atomic_json(a.output/'manifest.json', dict(status='complete', files=files, records=rows,
        plan_sha=sha256(a.output/'plan.json'), generator_sha=generator_hashes,
        training_original_observed_scalars=train, source_validation_original_observed_scalars=val,
        normal_twin_forward_scalars_per_pass=dict(train=train, source_validation=val),
        source_seed_disjoint_from_original_corpus=True, source_seed_disjoint_train_validation=True,
        benchmark_full_array_hash_matches=0, benchmark_index_sha=sha256(a.benchmark_index),
        mechanism_exposure='original official attribute-set generator; activation off; new systems and RNG; not held-out dynamics',
        labels_unchanged=True, new_channel_effect_labels=None))
    atomic_json(a.output/'preparation.json', dict(status='complete', total=len(plan), completed=len(rows)))
    print(json.dumps(dict(status='complete', train=train, validation=val, systems=len(rows))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--generator-root', type=Path, required=True)
    p.add_argument('--forbidden-seeds', type=Path, required=True); p.add_argument('--benchmark-index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--workers', type=int, default=16)
    main(p.parse_args())
