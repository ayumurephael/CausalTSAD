"""Run the immutable original source pipeline on this GPU, without target access.

This is deliberately a separate process: original modules, including the cache
writer and CandidateMethod, are imported from the original experiment snapshot.
It provides a hardware-matched reference; historical A100 parity stays recorded.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main(a):
    plan = json.loads(a.plan.read_text())
    original = Path(plan['original_source_code'])
    assert original.name == 'code_13b2294fc21910df3a1d0439f5bdb5c1c8b57fb5', original
    sys.path.insert(0, str(original))
    import numpy as np
    import torch
    from tools.run_improvement3 import cache_task, load
    from proposals.improvement3.models import CandidateMethod
    from causal_tsad.io import sha256, atomic_json
    from causal_tsad.frozen_detector import parameter_sha

    a.output.mkdir(parents=True, exist_ok=False)
    for name in ['task_progress', 'records']:
        (a.output / name).mkdir()
    source = Path(plan['source'])
    manifest = json.loads((source / 'manifest.json').read_text())
    row = next(r for r in manifest['rows'] if r['split'] == 'confirmation')
    runtime = dict(plan, code=str(original), primary_profile='true',
                   i11_frozen=[], deadline_unix=time.time() + 600)
    task = dict(id='hardware_reference', row=row, profile='true',
                graph='true', nuisance='none', technical_smoke_views=3)
    cache_task(a.output, runtime, task)
    record = json.loads((a.output / 'records/hardware_reference.json').read_text())
    item = next(x for x in record['files'] if x['view'] == 2)
    z, _, _ = load(a.output, item)
    arrays = {}
    exposure = 0
    for model in plan['models']:
        cp = Path(model['checkpoint'])
        assert sha256(cp) == model['SHA256']
        w = torch.load(cp, map_location='cpu', weights_only=False)
        m = CandidateMethod(w['arm'], w['seed']).cuda().eval().requires_grad_(False)
        m.load_state_dict(w['model'])
        before = parameter_sha(m)
        with torch.no_grad():
            out = m(z)
        arrays[model['name']] = out['score'].cpu().numpy()
        if out['root'] is not None:
            arrays['root__' + model['name']] = out['root'].cpu().numpy()
        assert parameter_sha(m) == before
        exposure += item['input_scalars']
    path = a.output / 'scores.npz'
    np.savez_compressed(path, **arrays)
    code_files = {str(p.relative_to(original)): sha256(p)
                  for p in original.rglob('*.py')}
    atomic_json(a.output / 'reference.json', dict(
        status='complete', original_source_code=str(original),
        original_code_files_SHA256=code_files, source_row=row,
        source_only=True, target_arrays_read=False, optimizer_steps=0,
        scores_SHA256=sha256(path), cache_record=record,
        additional_head_input_scalars=exposure,
        gpu=torch.cuda.get_device_name(0)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
