"""Fixed100development C01 evaluation. Metadata is only used after inference."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,pickle,sys,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256
from proposals.c01.data import tensor_batch
from tools.train_c01 import forward


def evaluate(a):
    import torch
    from sklearn.metrics import average_precision_score
    from proposals.c01.model import CausalResidualDetector
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(a.run/'final.pth',map_location='cpu',weights_only=False);config=saved['config']
    torch.cuda.set_per_process_memory_fraction(config['memory_gib_per_worker']*2**30/torch.cuda.get_device_properties(0).total_memory)
    model=CausalResidualDetector(feature_dim=saved['feature_dim'],width=config['width'],message_width=config['message_width'],horizon=config['horizon']).cuda().eval()
    model.load_state_dict(saved['model'],strict=True)
    manifest=json.loads((a.cache/'manifest.json').read_text());assert sha256(a.cache/'manifest.json')==saved['cache_manifest_sha256']
    rows=[r for r in manifest['files'] if r['kind']=='development'];assert len(rows)==100
    sys.path.insert(0,str(a.metric_code))
    from evaluate_table1 import compute_metrics
    from utils.slidingWindows import find_length_rank
    current=None;records=None;results=[];head_slots=0
    for r in rows:
        assert sha256(a.cache/r['path'])==r['SHA256'];b=tensor_batch(a.cache,[r],saved['arm'],'cuda');length=r['length']
        predicted={}
        with torch.no_grad():
            for view in ['x','normal']:
                out=forward(model,b,view)
                for key,value in [('fused',torch.sigmoid(out['logit'])),('base',torch.sigmoid(b[view+'_base'])),('density',out['density_score']),('node',out['innovation'])]:
                    array=value[0].cpu().numpy()
                    predicted[view+'_'+key]=array.reshape(-1,array.shape[-1])[:length] if key=='node' else array.ravel()[:length]
                head_slots+=int(b['query_valid'].sum())*r['channels']
        # Inference is complete before ground-truth events or labels are read.
        row=r['row'];assert row['split']=='development'
        if current!=row['shard']:
            with (a.benchmark/'data'/row['shard']).open('rb') as f:records=pickle.load(f)
            current=row['shard']
        sample=records[row['offset']];assert sample['system_id']==r['key']
        y=np.asarray(sample['labels'],int);period=max(1,int(find_length_rank(sample['time_series'],rank=1)))
        outcome={}
        for name in ['base','fused','density']:
            score=predicted['x_'+name];valid=len(np.unique(y))>1
            metrics={k:None for k in ['Affiliation-F','F1_T','Standard-F1','VUS-PR']}
            if valid:
                raw=compute_metrics(score,y,period);metrics={k:float(raw[k]) for k in metrics}
            threshold=saved['source_calibration'][{'base':'base_threshold','fused':'threshold','density':'density_threshold'}[name]]
            outcome[name]=dict(metrics=metrics,AP=float(average_precision_score(y,score)) if valid else None,
                normal_FPR=float(np.mean(predicted['normal_'+name]>threshold)))
        rca=[]
        if r['channels']>1:
            for event in sample['events']:
                if not event.get('applied',True):continue
                mask=np.zeros(length,bool)
                for start,end in event['annotation_intervals']:mask[start:end]=True
                if not mask.any():continue
                rank=np.argsort(-predicted['x_node'][mask].mean(0),kind='stable')
                ranks=[int(np.flatnonzero(rank==root)[0])+1 for root in event['root_channels']]
                rca.append(dict(event_id=event['event_id'],root_ranks=ranks,Top1=int(min(ranks)==1),Top3=int(min(ranks)<=min(3,r['channels'])),MRR=1/min(ranks)))
        path=a.run/'development_scores'/(r['name']+'.npz');path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,**predicted)
        result=dict(system_id=r['key'],track=row['track'],family=row['family'],channels=r['channels'],event_types=row['event_types'],
            scores=outcome,RCA=rca,C1_RCA_excluded=r['channels']==1,scores_sha256=sha256(path),
            input_sha256=row['time_series_sha256'],normal_input_sha256=row['normal_sha256'])
        atomic_json(a.run/'development_records'/(r['name']+'.json'),result);results.append(result)
        atomic_json(a.run/'evaluation_progress.json',dict(status='running',completed=len(results),total=100,
            head_query_scalar_forward_slots=head_slots,encoder_forward_scalars=0,target_validation_or_locked_arrays_opened=False))
    atomic_json(a.run/'development_results.json',dict(status='complete',records=results,parameter_updates=0,
        checkpoint_sha256=sha256(a.run/'final.pth'),head_query_scalar_forward_slots=head_slots,
        note='Common cache encoder costs separately counted; only fixed100development; RCA conditional on true windows'))
    atomic_json(a.run/'evaluation_progress.json',dict(status='complete',completed=100,total=100,
        head_query_scalar_forward_slots=head_slots,target_validation_or_locked_arrays_opened=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--metric-code',type=Path,required=True)
    a=p.parse_args()
    try:evaluate(a)
    except Exception:
        atomic_json(a.run/'evaluation_error.json',dict(error=traceback.format_exc()));raise
