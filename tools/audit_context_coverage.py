"""Source-normal calibrated context conflict and operational RCA abstention.

Read-only postprocessing of immutable caches. AP is NOT recomputed on a
label-dependent selected subset. Abstained events count as failures in the
all-event success rate. This is not a proof of causal identifiability.
"""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import softmax
from causal_tsad.io import sha256,atomic_json
from proposals.improvement3.metrics import spans

def confidence(root,rec,view):
    item=rec['files'][view];p=root/item['path'];assert sha256(p)==item['SHA256']
    with np.load(p) as f:
        valid=f['valid'][...,1:];distance=f['features'][...,1:,18]
        quorum=valid.sum(-1)>=2
        minimum=np.where(valid,distance,np.inf).min(-1)
        labels=f['labels']
    return minimum,quorum,labels

def main(a):
    plan=json.loads(a.plan.read_text());records=[];cal=[]
    for task in plan['cache_tasks']:
        if task['profile']!=a.profile:continue
        p=a.root/'records'/(task['id']+'.json');r=json.loads(p.read_text());assert r['status']=='complete';records.append(r)
        if task['row']['split']=='calibration':
            for v in [0,1,8,9]:
                d,q,_=confidence(a.root,r,v);cal.extend(d[q].tolist())
    assert len(records)==112 and cal
    threshold=float(np.quantile(cal,.99));rows=[]
    for rec in records:
        row=rec['task']['row']
        if row['split']!='confirmation':continue
        p=a.root/rec['frozen_path'];assert sha256(p)==rec['frozen_SHA256']
        with np.load(p) as f:all_nodes=f['root__consistency']
        for view in range(16):
            distance,quorum,y=confidence(a.root,rec,view)
            nodes=all_nodes[view];prob=softmax(nodes,axis=-1);pred=prob.argmax(-1)
            covered=quorum[np.arange(len(pred)),pred]&(distance[np.arange(len(pred)),pred]<=threshold)
            events=[]
            for lo,hi in spans(y>0):
                available=covered[lo:hi];attempted=bool(available.mean()>=.5)
                ranking=np.argsort(-prob[lo:hi][available].mean(0),kind='stable') if attempted else []
                rank=int(np.flatnonzero(ranking==row['root'])[0])+1 if attempted else None
                events.append(dict(onset=lo,offset=hi,attempted=attempted,rank=rank,coverage=float(available.mean()),
                                   Top1_all_events=int(attempted and rank==1)))
            rows.append(dict(key=row['key'],view=view,C=row['C'],fault=row['fault_names'][view%8],background='plain' if view<8 else 'smooth',
                             point_coverage=float(covered.mean()),events=events))
    assert not a.output.exists()
    atomic_json(a.output,dict(status='complete',profile=a.profile,threshold=threshold,calibration='normal source candidate-conflict99',
      minimum_quarter_candidates=2,event_coverage_required=.5,rows=rows,global_updates=0,extra_basis_forwards=0,
      all_event_denominator_keeps_abstentions=True,AP_selected_subset_reported=False,
      claim='Operational refusal under high context conflict; not a Chow F-test or identifiability theorem'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--profile',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
