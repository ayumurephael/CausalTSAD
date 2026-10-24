"""Fixed development probe, never benchmark validation or locked-test arrays."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import model_inputs, graph_control, sha256, atomic_json
from baselines.shared_conditional import SharedConditional


def selected_rows(root):
    rows=json.loads((root/'data/system_index.json').read_text());chosen=[]
    for track in ['id','mechanism_holdout']:
        for c in [1,8,12,16,20]:
            group=[r for r in rows if r['split']=='development' and r['track']==track and r['channels']==c]
            group.sort(key=lambda r:hashlib.sha256(r['system_id'].encode()).hexdigest())
            chosen+=group[:10]
    assert len(chosen)==100
    forbidden_shards={r['shard'] for r in rows if r['split']!='development'}
    assert not {r['shard'] for r in chosen}&forbidden_shards,'A chosen shard mixes development with a forbidden split'
    return chosen


def predict(model,values,graph,chunk=256):
    import torch
    t,c=values.shape;out=[];density=[];innovation=[];h=model.history;actual_input_scalars=0
    for start in range(0,t,chunk):
        begin=max(0,start-h);end=min(t,start+chunk);offset=start-begin
        with torch.no_grad():
            x=torch.from_numpy(values[begin:end]).cuda()[None]
            actual_input_scalars+=x.numel()
            result=model(x,torch.from_numpy(graph).cuda()[None])
            out.append(torch.sigmoid(result['logit'])[0,offset:].cpu().numpy())
            density.append(result['density_score'][0,offset:].cpu().numpy())
            innovation.append(result['innovation'][0,offset:].cpu().numpy())
    return np.concatenate(out),np.concatenate(density),np.concatenate(innovation),actual_input_scalars


def main(a):
    import torch
    from sklearn.metrics import average_precision_score
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(a.run/'final.pth',map_location='cpu',weights_only=False);config=saved['config']
    torch.cuda.set_per_process_memory_fraction(config['memory_gib_per_worker']*2**30/torch.cuda.get_device_properties(0).total_memory)
    model=SharedConditional(config['history'],config['width'],config['message_width']).cuda().eval()
    model.load_state_dict(saved['model'],strict=True)
    rows=selected_rows(a.benchmark);atomic_json(a.run/'development_selection.json',rows)
    code_root=a.metric_code
    sys.path.insert(0,str(code_root))
    from evaluate_table1 import compute_metrics
    from utils.slidingWindows import find_length_rank
    results=[];current=None;records=None;forward_scalars=0
    for row in rows:
        assert row['split']=='development'
        if row['shard']!=current:
            with (a.benchmark/'data'/row['shard']).open('rb') as f:records=pickle.load(f)
            current=row['shard']
        s=records[row['offset']];assert s['system_id']==row['system_id']
        inputs=model_inputs(s,'anomalous');normal_inputs=model_inputs(s,'normal')
        # Mapping is label-free and frozen for a given system in all optimization seeds.
        seed=int.from_bytes(hashlib.sha256(row['system_id'].encode()).digest()[:4],'little')^0x46113
        permutation=np.random.RandomState(seed).permutation(row['channels'])
        graph=graph_control(inputs['adjacency'],saved['mode'],permutation)
        started=time.monotonic()
        score,density,node,ns=predict(model,inputs['values'],graph)
        normal,normal_density,_,nn=predict(model,normal_inputs['values'],graph)
        forward_scalars+=ns+nn
        y=np.asarray(s['labels'],int);valid=len(np.unique(y))>1
        period=max(1,int(find_length_rank(s['time_series'],rank=1)))
        metric={k:None for k in ['Affiliation-F','F1_T','Standard-F1','VUS-PR']}
        if valid:
            raw=compute_metrics(score,y,period);metric={k:float(raw[k]) for k in metric}
        rca=[]
        if row['channels']>1:
            for event in s['events']:
                if not event.get('applied',True):continue
                mask=np.zeros(len(y),bool)
                for start,end in event['annotation_intervals']:mask[start:end]=True
                if not mask.any():continue
                rank=np.argsort(-node[mask].mean(0),kind='stable')
                roots=set(event['root_channels']);ranks=[int(np.flatnonzero(rank==r)[0])+1 for r in roots]
                rca.append(dict(event_id=event['event_id'],root_ranks=ranks,Top1=int(min(ranks)==1),
                    Top3=int(min(ranks)<=min(3,row['channels'])),MRR=1/min(ranks)))
        key=hashlib.sha256(row['system_id'].encode()).hexdigest()[:24]
        path=a.run/'development_scores'/(key+'.npz');path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,score=score,normal=normal,density=density,normal_density=normal_density,innovation=node)
        result=dict(system_id=row['system_id'],track=row['track'],channels=row['channels'],family=row['family'],
            event_types=row['event_types'],metrics=metric,AP=float(average_precision_score(y,score)) if valid else None,
            density_AP=float(average_precision_score(y,density)) if valid else None,
            normal_FPR=float(np.mean(normal>saved['source_calibration']['threshold'])),
            density_normal_FPR=float(np.mean(normal_density>saved['source_calibration']['density_threshold'])),
            RCA=rca,C1_RCA_excluded=row['channels']==1,scores_sha256=sha256(path),
            input_sha256=row['time_series_sha256'],normal_input_sha256=row['normal_sha256'],
            seconds=time.monotonic()-started)
        atomic_json(a.run/'development_records'/(key+'.json'),result);results.append(result)
        atomic_json(a.run/'evaluation_progress.json',dict(status='running',completed=len(results),total=len(rows),
            actual_frozen_forward_scalars=forward_scalars,validation_and_locked_arrays_read=False))
    atomic_json(a.run/'development_results.json',dict(status='complete',records=results,
        checkpoint_sha256=sha256(a.run/'final.pth'),parameter_updates=0,source_calibration_only=True,
        evaluation_scope='fixed100development systems, no validation/locked',actual_frozen_forward_scalars=forward_scalars))
    atomic_json(a.run/'evaluation_progress.json',dict(status='complete',completed=len(results),total=len(rows),
        actual_frozen_forward_scalars=forward_scalars,validation_and_locked_arrays_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--metric-code',type=Path,required=True);main(p.parse_args())
