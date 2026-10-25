"""No new inference: audit quantized probabilities vs cached original logits."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256


def main(a):
    m=json.loads((a.parent/'cache/manifest.json').read_text());rows=[r for r in m['files'] if r['kind']=='development'];out=[]
    for r in rows:
        assert sha256(a.parent/'cache'/r['path'])==r['SHA256']
        with np.load(a.parent/'cache'/r['path']) as z:
            logit=z['x_base'].ravel()[:r['length']];labels=z['labels'].ravel()[:r['length']]
        with np.load(a.parent/'recovery_v1/runs/pretrained_none_s348/development_scores'/(r['name']+'.npz')) as z:prob=z['x_base']
        valid=len(np.unique(labels))>1
        row=dict(system_id=r['key'],track=r['row']['track'],channels=r['channels'],length=r['length'],
            float32_probability_AP=float(average_precision_score(labels,prob)) if valid else None,
            original_logit_AP=float(average_precision_score(labels,logit)) if valid else None,
            float64_probability_AP=float(average_precision_score(labels,expit(logit.astype(np.float64)))) if valid else None,
            probability_one_points=int((prob==1).sum()),probability_zero_points=int((prob==0).sum()),
            distinct_logit_values=len(np.unique(logit)),distinct_probability_values=len(np.unique(prob)),
            logit_min=float(logit.min()),logit_max=float(logit.max()))
        out.append(row)
    groups={}
    for name in ['all','multi','id_multi','heldout_multi','C1']:
        selected=[r for r in out if name=='all' or name=='multi' and r['channels']>1 or name=='C1' and r['channels']==1 or
            name=='id_multi' and r['channels']>1 and r['track']=='id' or name=='heldout_multi' and r['channels']>1 and r['track']=='mechanism_holdout']
        groups[name]=dict(systems=len(selected),**{k:float(np.mean([r[k] for r in selected if r[k] is not None])) for k in ['float32_probability_AP','original_logit_AP','float64_probability_AP']},
            probability_one_points=sum(r['probability_one_points'] for r in selected),total_points=sum(r['length'] for r in selected))
    atomic_json(a.output,dict(status='complete',records=out,groups=groups,extra_model_forwards=0,optimizer_updates=0,
        target_scope='same fixed100development, cached arrays only; no validation/locked',
        primary_protocol_changed=False,limitation='Representation diagnostic only; AP rank/rounding effect does not itself establish a VUS-PR difference or causal gain.'))
    print(json.dumps(groups))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
