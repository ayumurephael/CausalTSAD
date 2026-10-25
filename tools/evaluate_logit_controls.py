"""Frozen cached-head readout controls; original probability results stay intact."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,pickle,sys,time,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256
from proposals.c01.data import batches,tensor_batch
from tools.train_c01 import forward
from tools.cache_c01 import module_sha


def evaluate(a):
    import torch
    from sklearn.metrics import average_precision_score
    from proposals.c01.model import CausalResidualDetector
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(0).total_memory)
    saved=torch.load(a.run/'final.pth',map_location='cpu',weights_only=False);cfg=saved['config']
    model=CausalResidualDetector(saved['feature_dim'],cfg['width'],cfg['message_width'],cfg['horizon']).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['model'],strict=True);initial=module_sha(model)
    training=json.loads((a.run/'progress.json').read_text())
    assert sha256(a.run/'final.pth')==training['checkpoint_sha256'] and initial==training['final_parameter_sha256']
    assert not a.output.exists();a.output.mkdir(parents=True);began=time.monotonic()
    cache=a.parent/'cache';manifest=json.loads((cache/'manifest.json').read_text());assert sha256(cache/'manifest.json')==saved['cache_manifest_sha256']
    source=[r for r in manifest['files'] if r['kind']=='source_validation'];rows=[r for r in manifest['files'] if r['kind']=='development'];assert len(rows)==100
    calibration=[];base_calibration=[];source_scalars=0
    with torch.no_grad():
        for group in batches(source,cfg['batch_input_scalars'],9002):
            b=tensor_batch(cache,group,saved['arm'],'cuda');valid=b['query_valid'].bool();result=forward(model,b,'normal')
            calibration.append(result['logit'][valid].cpu().numpy());base_calibration.append(b['normal_base'][valid].cpu().numpy())
            source_scalars+=int(valid.sum())*b['normal_query'].shape[-1]
    assert source_scalars==1968128
    thresholds=dict(fused=float(np.quantile(np.concatenate(calibration),.99)),base=float(np.quantile(np.concatenate(base_calibration),.99)))
    from scipy.special import expit
    original_probability_threshold=float(np.quantile(expit(np.concatenate(calibration).astype(np.float64)),.99))
    # Quantization can affect thresholds too; retain both independently calibrated representations.
    atomic_json(a.output/'source_logit_calibration.json',dict(thresholds=thresholds,
        source_normal_query_scalars=source_scalars,extra_encoder_forwards=0,parameter_updates=0,
        float64_probability_threshold_diagnostic=original_probability_threshold))
    sys.path.insert(0,str(a.metric_code))
    from evaluate_table1 import compute_metrics
    from utils.slidingWindows import find_length_rank
    current=None;raw_records=None;results=[];base_results=[];forward_scalars=0;max_null=0.
    for r in rows:
        if time.monotonic()-began>3*3600:raise TimeoutError('Fixed readout worker deadline')
        assert sha256(cache/r['path'])==r['SHA256'];b=tensor_batch(cache,[r],saved['arm'],'cuda');length=r['length']
        with torch.no_grad():
            obs=forward(model,b,'x')['logit'][0].cpu().numpy().ravel()[:length]
            normal=forward(model,b,'normal')['logit'][0].cpu().numpy().ravel()[:length]
            base=b['x_base'][0].cpu().numpy().ravel()[:length];base_normal=b['normal_base'][0].cpu().numpy().ravel()[:length]
            probability=torch.sigmoid(torch.from_numpy(obs)).numpy()
        with np.load(a.run/'development_scores'/(r['name']+'.npz')) as z:prior_prob=np.array(z['x_fused'])
        error=float(np.max(np.abs(probability-prior_prob)));assert error<1e-6;max_null=max(max_null,error)
        forward_scalars+=2*r['scalars']
        with np.load(cache/r['path']) as z:y=z['labels'].ravel()[:length]
        row=r['row'];assert row['split']=='development'
        if current!=row['shard']:
            with (a.benchmark/'data'/row['shard']).open('rb') as f:raw_records=pickle.load(f)
            current=row['shard']
        sample=raw_records[row['offset']];assert sample['system_id']==r['key']
        assert np.array_equal(y,sample['labels']);period=max(1,int(find_length_rank(sample['time_series'],rank=1)))
        valid=len(np.unique(y))>1
        def metrics(score):
            keys=['Affiliation-F','F1_T','Standard-F1','VUS-PR']
            raw=compute_metrics(score,y,period) if valid else {k:None for k in keys}
            return dict(metrics={k:float(raw[k]) if raw[k] is not None else None for k in keys},AP=float(average_precision_score(y,score)) if valid else None)
        result=dict(system_id=r['key'],track=row['track'],channels=r['channels'],scores=metrics(obs),
            normal_FPR=float(np.mean(normal>thresholds['fused'])),probability_null_max_error=error)
        if a.compute_base:
            base_result=dict(system_id=r['key'],track=row['track'],channels=r['channels'],scores=metrics(base),
                normal_FPR=float(np.mean(base_normal>thresholds['base'])))
            base_results.append(base_result)
        path=a.output/'scores'/(r['name']+'.npz');path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,fused_logit=obs,normal_fused_logit=normal)
        result['scores_sha256']=sha256(path);results.append(result)
        atomic_json(a.output/'progress.json',dict(status='running',completed=len(results),total=100,
            extra_development_head_query_scalars=forward_scalars,extra_source_normal_head_query_scalars=source_scalars,
            extra_encoder_forwards=0,parameter_updates=0,max_probability_null_error=max_null))
    assert initial==module_sha(model)
    atomic_json(a.output/'results.json',dict(status='complete',arm=saved['arm'],seed=saved['seed'],records=results,
        base_logit_records=base_results,checkpoint_sha256=sha256(a.run/'final.pth'),
        extra_development_head_query_scalars=forward_scalars,extra_source_normal_head_query_scalars=source_scalars,
        extra_encoder_forwards=0,parameter_updates=0,frozen_head_digest=initial,head_hash_unchanged=True,
        max_probability_null_error=max_null,original_probability_results_untouched=True,
        protocol='supplementary raw-logit controls for original/fused models, fixed100development, no validation/locked',
        interpretation='AP ranks expose rounding ties. VUS/threshold metrics also depend on score representation; do not describe every change as pure quantization.'))
    atomic_json(a.output/'progress.json',dict(status='complete',completed=100,total=100,
        extra_development_head_query_scalars=forward_scalars,extra_source_normal_head_query_scalars=source_scalars,
        extra_encoder_forwards=0,parameter_updates=0,max_probability_null_error=max_null))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--metric-code',type=Path,required=True);p.add_argument('--compute-base',action='store_true');a=p.parse_args()
    try:evaluate(a)
    except Exception:
        atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
