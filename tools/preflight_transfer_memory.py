"""Source-only worst-shape resource probe; no target labels or model changes."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from tools.run_improvement3 import init
from causal_tsad.io import sha256,atomic_json
from causal_tsad.frozen_detector import parameter_sha

def main(a):
    p=json.loads(a.freeze.read_text());runtime=dict(p,worker_GiB=a.worker_memory_GiB)
    basis,det=init(runtime);before=parameter_sha(basis)
    source=Path(p['source']);m=json.loads((source/'manifest.json').read_text())
    row=next(r for r in m['rows'] if r['split']=='confirmation');path=source/row['path'];assert sha256(path)==row['SHA256']
    with np.load(path) as f:small=f['values'][2].copy()
    # The largest actual concat is 5000 x (20 observed + 20 residual channels).
    # Content is a tiled source example solely to exercise allocation shape.
    x=np.tile(small,(int(np.ceil(5000/len(small))),int(np.ceil(40/small.shape[1]))))[:5000,:40]
    assert np.isfinite(det.predict(x,'selftrained_multi','native')['logit']).all()
    det.verify_frozen();assert parameter_sha(basis)==before
    atomic_json(a.output,dict(status='passed',shape=list(x.shape),source_SHA256=row['SHA256'],
        target_arrays_read=False,optimizer_steps=0,worker_GiB=a.worker_memory_GiB,
        detector_scalars=det.actual_input_scalars,detector_calls=det.encoder_calls,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu=torch.cuda.get_device_name(0)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker-memory-GiB',type=int,required=True);main(p.parse_args())
