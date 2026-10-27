"""Offline any-root localization with deterministic one-to-one event matching."""
import numpy as np
from .metrics import spans

def localization(score,nodes,events,threshold,first=32):
    T=len(score);eligible=[];excluded=[]
    for index,event in enumerate(events):
        roots=sorted(set(int(v) for v in event.get('root_channels',[])))
        mask=np.zeros(T,bool)
        for lo,hi in event.get('annotation_intervals',[]):
            a=max(0,int(lo)-first);b=min(T,int(hi)-first)
            if b>a:mask[a:b]=True
        eid=event.get('event_id',index)
        if not event.get('applied',True) or not roots or not mask.any():
            excluded.append(dict(event_id=eid,reason='not_applied_no_roots_or_no_eligible_points'));continue
        assert all(0<=v<nodes.shape[1] for v in roots)
        eligible.append(dict(event_id=eid,roots=roots,mask=mask))
    alarms=spans(np.asarray(score)>threshold)
    # Global overlap-descending greedy matching. Stable event/alarm order breaks
    # ties, no root score or root label enters matching.
    pairs=[]
    for i,e in enumerate(eligible):
        for j,(lo,hi) in enumerate(alarms):
            overlap=int(e['mask'][lo:hi].sum())
            if overlap:pairs.append((-overlap,i,j))
    matches={};used=set()
    for neg,i,j in sorted(pairs):
        if i not in matches and j not in used:matches[i]=j;used.add(j)
    z=nodes-nodes.max(1,keepdims=True);prob=np.exp(z);prob/=prob.sum(1,keepdims=True)
    def rank(mask,roots):
        order=np.argsort(-prob[mask].mean(0),kind='stable');ranks=[int(np.flatnonzero(order==r)[0])+1 for r in roots]
        return min(ranks),ranks
    records=[]
    for i,e in enumerate(eligible):
        cr,per=rank(e['mask'],e['roots']);j=matches.get(i);dr=None
        if j is not None:
            lo,hi=alarms[j];mask=np.zeros(T,bool);mask[lo:hi]=True;dr,_=rank(mask,e['roots'])
        records.append(dict(event_id=e['event_id'],root_count=len(e['roots']),conditional_rank=cr,per_root_ranks=per,
          alarm_index=j,detected=j is not None,detected_window_rank=dr,joint_correct=dr==1,
          onset_error=alarms[j][0]-int(np.flatnonzero(e['mask'])[0]) if j is not None else None))
    n=len(records)
    return dict(events=records,excluded=excluded,eligible_events=n,multi_root_events=sum(len(e['roots'])>1 for e in eligible),
      conditional_Top1=float(np.mean([e['conditional_rank']==1 for e in records])) if n else None,
      conditional_MRR=float(np.mean([1/e['conditional_rank'] for e in records])) if n else None,
      event_recall=len(matches)/n if n else None,joint_detected_root_recall=sum(e['joint_correct'] for e in records)/n if n else None,
      alarm_events=len(alarms),unmatched_alarm_events=len(alarms)-len(used),root_definition='any_true_root',
      matching='global_overlap_descending_greedy_one_to_one')
