"""Source-frozen development-only transfer. Never opens validation/locked.

The freeze file must be written BEFORE loading benchmark arrays. Development
is a transfer check, not checkpoint/hyperparameter/threshold selection.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,pickle,argparse,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from causal_tsad.io import atomic_json,sha256,array_sha,adjacency
from tools.run_improvement3 import init,save_npz
from proposals.improvement3.components import components
from proposals.improvement3.models import CandidateMethod
from proposals.improvement3.evidence import frozen
from proposals.improvement3.metrics import evaluate

def infer(values,graph,basis,detector,methods,nuisance='none',chunk=5000):
    # Complete eligible suffix: 32 history points for every subsequent chunk;
    # each output point32..T-1 appears exactly once. Never label-driven chunks.
    T,C=values.shape;raw=detector.predict(values,'selftrained_multi','native')['logit']
    outputs={};roots={};budget=0
    for lo in range(32,T,chunk):
        hi=min(T,lo+chunk);start=lo-32
        # A last small chunk still needs >=64 context rows: extend LEFT only,
        # predict the whole extra context but only emit its requested suffix.
        if hi-start<256:start=max(0,hi-256)
        x=values[start:hi];z=components(basis,torch.as_tensor(x,device='cuda',dtype=torch.float64),torch.as_tensor(graph,device='cuda',dtype=torch.float64),nuisance)
        z['raw_logit']=torch.as_tensor(raw[start+32:hi],device='cuda',dtype=torch.float64)
        r=(z['response']-z['mu'][...,0])/z['variance'][...,0].sqrt()
        cat=np.concatenate([x[32:],r.cpu().numpy()],1)
        z['concat_logit']=torch.as_tensor(detector.predict(cat,'selftrained_multi','native')['logit'],device='cuda',dtype=torch.float64)
        ss,rr,_=frozen(z);take=lo-(start+32)
        for name,model in methods.items():
            with torch.no_grad():p=model(z)
            ss[name]=p['score']
            if p['root'] is not None:rr[name]=p['root']
        for k,v in ss.items():outputs.setdefault(k,[]).append(v[take:].cpu().numpy())
        for k,v in rr.items():roots.setdefault(k,[]).append(v[take:].cpu().numpy())
        budget+=x.size
    return {k:np.concatenate(v) for k,v in outputs.items()},{k:np.concatenate(v) for k,v in roots.items()},budget

def main(a):
    freeze=json.loads(a.freeze.read_text());assert freeze['source_only_selection'] and not freeze['locked_opened']
    assert freeze['frozen_before_target'] and freeze['max_target_split']=='development'
    assert len(freeze['development_rows'])==432 and len({r['system_id'] for r in freeze['development_rows']})==432
    row=freeze['development_rows'][a.task];assert row['split']=='development' and row['channels']>1
    assert all(r['split']=='development' for r in freeze['development_rows'])
    basis,detector=init(freeze);methods={}
    for r in freeze['models']:
        p=Path(r['checkpoint']);assert sha256(p)==r['SHA256'];cp=torch.load(p,map_location='cpu',weights_only=False)
        m=CandidateMethod(cp['arm'],cp['seed']).cuda().eval();m.load_state_dict(cp['model']);m.requires_grad_(False);methods[r['name']]=m
    shard=Path(freeze['benchmark'])/'data'/row['shard'];assert sha256(shard)==row['shard_SHA256']
    with shard.open('rb') as f:s=pickle.load(f)[row['offset']]
    assert s['system_id']==row['system_id'] and array_sha(s['time_series'])==row['time_series_sha256']
    g=adjacency(s['graph'],row['channels']);metrics=[];scores={};rca=[];exposure=0
    for vi,x in enumerate([s['time_series'],s['normal_time_series']]):
        # Each case gets independent context fitting, including the normal case.
        y=np.asarray(s['labels'][32:]) if vi==0 else np.zeros(len(x)-32,np.int8)
        ss,rr,b=infer(x,g,basis,detector,methods,freeze.get('nuisance','none'));exposure+=b
        for name,score in ss.items():
            if name not in freeze['thresholds']:continue
            assert len(score)==len(y);scores[name+'__'+str(vi)]=score
            metrics.append(dict(method=name,view=vi,**evaluate(score,y,freeze['thresholds'][name])))
            if vi==0 and name in rr:
                # Multi-event roots evaluated per actual event, not one root per
                # whole system. Full end-to-end matching is separately stored.
                for event in s['events']:
                    if not event.get('applied',True) or not event['root_channels']:continue
                    mask=np.zeros(len(x),np.int8)
                    for lo,hi in event['annotation_intervals']:mask[lo:hi]=1
                    if not mask[32:].any():continue
                    per=[evaluate(score,mask[32:],freeze['thresholds'][name],rr[name],int(root)) for root in event['root_channels']]
                    rca.append(dict(method=name,event_id=event['event_id'],per_root=per))
    detector.verify_frozen();a.output.mkdir(parents=True,exist_ok=True)
    dst=a.output/row['system_id'];save_npz(dst.with_suffix('.npz'),**scores)
    atomic_json(dst.with_suffix('.json'),dict(status='complete',row=row,metrics=metrics,conditional_and_detected_RCA=rca,
      first=32,context_chunk=5000,source_freeze_SHA256=sha256(a.freeze),scores_SHA256=sha256(dst.with_suffix('.npz')),
      input_scalars=exposure,global_updates=0,TimeRCD_RCA=False,locked_opened=False,
      metric_followup='Original four-metric worker must run on these same suffix scores; AP/PR-AUC are already separate'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--task',type=int,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
