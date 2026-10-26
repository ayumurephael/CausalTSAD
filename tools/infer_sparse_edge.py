"""Source-only frozen readouts over SHA-verified old innovation caches."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,argparse,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from proposals.sparse_edge_alternatives import heads,METHODS,FIRST
from tools.train_meta_basis import nuisance
from causal_tsad.io import sha256,array_sha,atomic_json
def load(row,old):
    path=Path(old['source'] if row['kind']=='source' else old['calibration_source'])/row['path'];assert sha256(path)==row['SHA256']
    with np.load(path) as f:x=f['cases'];g=f['graph'];labels=f['labels'] if row['kind']=='source' else None
    if row['kind']=='source':events=row['case_metadata']
    else:
        x=np.concatenate([nuisance(x,row['key'],bg)[:2] for bg in ['plain','smooth']]);labels=np.zeros(x.shape[:2],np.int8);events=[dict(kind='normal',duration=0,root=None) for _ in x]
    return x,g,labels,events
def run(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(int(1.5*2**30)/torch.cuda.get_device_properties(0).total_memory)
    plan=json.loads(a.plan.read_text());probe=Path(plan['probe']);old=json.loads((probe/'job_plan.json').read_text());records=[];slots=queries=apps=0;start=time.time();replay=0.
    for row in plan['tasks'][a.worker::a.workers]:
        assert row['kind'] in ['source','calibration'] and time.time()<plan['deadline_unix'];ident=row['task_id'];oldmeta=json.loads((probe/'records'/(ident+'.json')).read_text());x,g,labels,events=load(row,old);assert array_sha(x)==oldmeta['input_SHA256']
        metrics={};scores={};audits={}
        oldscore=probe/'records'/(ident+'.npz');assert sha256(oldscore)==oldmeta['scores_SHA256']
        with np.load(oldscore) as sf:old_energies={str(s):sf['s'+str(s)+'__true__conditional_energy'].copy() for s in [348,349,350]}
        for seed in [348,349,350]:
            cache=probe/'innovation_cache'/(ident+'__s'+str(seed)+'__true.npz');cm=next(v for v in oldmeta['innovation_cache'] if v['path']==str(cache.relative_to(probe)));assert sha256(cache)==cm['SHA256']
            with np.load(cache) as f:r=f['residual'].copy();v=f['variance'].copy()
            with torch.no_grad():out,audit=heads(torch.tensor(x,device='cuda'),torch.tensor(g,device='cuda'),torch.tensor(r,device='cuda'),torch.tensor(v,device='cuda'),str(row['key']))
            assert set(out)==set(METHODS);audits[str(seed)]=audit;apps+=audit['candidate_scale_time_applications'];slots+=x.size;queries+=r.size
            for name,o in out.items():
                s=o['score'].cpu().numpy();rp=o['root'].cpu().numpy();assert np.isfinite(s).all() and np.isfinite(rp).all();tag='s'+str(seed)+'__'+name;scores[tag]=s;items=[]
                if name=='energy16':delta=float(np.max(np.abs(s-old_energies[str(seed)])));assert delta<1e-7;replay=max(replay,delta)
                for j,meta in enumerate(events):
                    y=labels[j,FIRST:]>0;item=dict(view=j,kind=meta['kind'],duration=meta['duration'],AP=float(average_precision_score(y,s[j])) if y.any() else None,Top1=None,Top3=None,MRR=None)
                    if y.any():
                        order=np.argsort(-rp[j,y].mean(0),kind='stable');rank=int(np.flatnonzero(order==meta['root'])[0])+1;item.update(Top1=int(rank==1),Top3=int(rank<=min(3,len(g))),MRR=1/rank)
                    items.append(item)
                metrics[tag]=items
            del out
        path=a.root/'records'/ident;assert not path.with_suffix('.npz').exists();np.savez_compressed(path.with_suffix('.npz'),**scores)
        atomic_json(path.with_suffix('.json'),dict(task=row,input_SHA256=array_sha(x),old_record_SHA256=sha256(probe/'records'/(ident+'.json')),metrics=metrics,audits=audits,scores_SHA256=sha256(path.with_suffix('.npz')),old_energy16_max_error=replay,target_arrays_opened=False))
        records.append(dict(path=str(path.with_suffix('.json').relative_to(a.root)),SHA256=sha256(path.with_suffix('.json')),scores_SHA256=sha256(path.with_suffix('.npz'))))
        atomic_json(a.root/('worker'+str(a.worker)+'_progress.json'),dict(status='running',completed=len(records)))
    atomic_json(a.root/('worker'+str(a.worker)+'_manifest.json'),dict(status='complete',records=records,reused_raw_input_equivalent_slots=slots,reused_query_slots=queries,candidate_scale_time_applications=apps,score_head_query_applications=queries*len(METHODS),new_backbone_forwards=0,optimizer_updates=0,target_arrays_opened=False,seconds=time.time()-start,old_energy16_max_error=replay,plan_SHA256=sha256(a.plan)))
    atomic_json(a.root/('worker'+str(a.worker)+'_progress.json'),dict(status='complete',completed=len(records)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,required=True);a=p.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors'/('worker'+str(a.worker)+'.json'),dict(error=traceback.format_exc()));raise
