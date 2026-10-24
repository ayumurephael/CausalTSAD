"""GPU information-boundary checks before any optimizer run."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json, sha256, model_inputs, graph_control


def main(a):
    import torch
    from baselines.shared_conditional import SharedConditional
    from tools.train_b1 import load_pool
    from tools.evaluate_b1 import predict,selected_rows
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=importlib.util.spec_from_file_location('boundaries',Path(__file__).resolve().parents[1]/'tests/test_shared_conditional.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    report=tests.run('cuda');pool,manifest=load_pool(a.source,'train','parents')
    checks=[];technical_forward=0;technical_backward=0
    torch.manual_seed(348);model=SharedConditional().cuda().eval()
    for c in [1,8,12,16,20]:
        row=pool[c];x=row['x'][0,:160].copy();g=row['controlled_graph'][0]
        xt=torch.from_numpy(x)[None].cuda();gt=torch.from_numpy(g)[None].cuda()
        with torch.no_grad():full=model(xt,gt)
        parts=predict(model,x,g,chunk=37)
        error=float(np.max(np.abs(parts[0]-torch.sigmoid(full['logit'])[0].cpu().numpy())))
        assert error<1e-5,(c,error)
        technical_forward+=xt.numel()+parts[3]
        # Backward on normal density only; no optimizer and no weight update.
        model.zero_grad();out=model(xt,gt);loss=out['nll'].mean()
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        technical_forward+=xt.numel();technical_backward+=xt.numel()
        checks.append(dict(channels=c,chunk_probability_max_error=error,finite_backward=True))
    assert not any(name in model.state_dict() for name in ['root','labels','mechanism'])
    selected=selected_rows(a.benchmark)
    report.update(status='passed',checks=checks,full_model_parameters=sum(p.numel() for p in model.parameters()),
        source_manifest_sha256=sha256(a.source/'manifest.json'),benchmark_selection_systems=len(selected),
        benchmark_arrays_opened=False,technical_forward_scalars=technical_forward,
        technical_backward_scalars=technical_backward,optimizer_steps=0,
        cuda_peak_bytes=torch.cuda.max_memory_allocated(),torch_version=torch.__version__,
        source_training_scalars=manifest['training_original_observed_scalars'],
        source_validation_scalars=manifest['source_validation_original_observed_scalars'])
    atomic_json(a.output,report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
