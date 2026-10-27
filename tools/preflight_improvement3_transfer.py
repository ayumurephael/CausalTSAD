"""Source-only writer/inference and event matching checks before target access."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from tools.transfer_improvement3 import infer
from tools.run_improvement3 import init
from proposals.improvement3.models import CandidateMethod
from proposals.improvement3.event_localization import localization
from causal_tsad.io import sha256,atomic_json
from causal_tsad.frozen_detector import parameter_sha

def main(a):
    plan=json.loads(a.plan.read_text());basis,det=init(plan);before=parameter_sha(basis)
    source=Path(plan['source']);m=json.loads((source/'manifest.json').read_text());row=next(r for r in m['rows'] if r['split']=='confirmation')
    p=source/row['path'];assert sha256(p)==row['SHA256']
    with np.load(p) as f:x=f['values'][2].copy();g=f['graph'].copy()
    models={}
    for item in plan['models']:
        cp=Path(item['checkpoint']);assert sha256(cp)==item['SHA256'];w=torch.load(cp,map_location='cpu',weights_only=False)
        model=CandidateMethod(w['arm'],w['seed']).cuda().eval().requires_grad_(False);model.load_state_dict(w['model']);models[item['name']]=model
    scores,roots,budget=infer(x,g,basis,det,models)
    # One 992-point source block must agree with actual source cached inference.
    errors={}
    for name in models:
        with np.load(plan['source_scores'][name]) as f:ref=f[row['key']+'__2'];errors[name]=float(np.max(np.abs(ref-scores[name])))
    assert max(errors.values())<1e-7,errors
    # Force a small last block to exercise left-context extension; every output
    # point still emitted once, with finite scores and matching root lengths.
    ss,rr,small=infer(x,g,basis,det,models,chunk=450)
    assert all(len(v)==len(x)-32 and np.isfinite(v).all() for v in ss.values())
    assert all(len(v)==len(x)-32 for v in rr.values())
    # Two true events share one continuous alarm: only one is matched.
    events=[dict(event_id=i,applied=True,root_channels=[1],annotation_intervals=[(a,b)]) for i,(a,b) in enumerate([(2,4),(6,8)])]
    result=localization(np.ones(10),np.tile([[0.,1.]],(10,1)),events,.5,first=0)
    assert result['eligible_events']==2 and result['event_recall']==.5 and result['joint_detected_root_recall']==.5
    none=localization(np.zeros(10),np.tile([[0.,1.]],(10,1)),events,.5,first=0)
    assert none['event_recall']==0 and none['conditional_Top1']==1
    det.verify_frozen();assert parameter_sha(basis)==before
    atomic_json(a.output,dict(status='passed',source_only=True,target_arrays_read=False,source_parity_max=errors,
      budgets=[budget,small],detector_input_scalars=det.actual_input_scalars,detector_calls=det.encoder_calls,
      one_to_one_alarm_matching=True,all_suffix_points_emitted_once=True,weight_updated=False,optimizer_steps=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
