"""Frozen source-final models on the pre-existing80 development systems only."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,argparse,json,pickle,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from proposals.causal_meta_basis import CausalMetaBasis,FIRST
from proposals.raw_context import parameter_sha
from causal_tsad.io import atomic_json,sha256,array_sha,adjacency
ARMS=['frozen_basis','normal_only','normal_relation'];MODES=['true','empty','rewired'];SEEDS=[348,349,350]
def wrong(g,ident):
    import hashlib
    rng=np.random.RandomState(int.from_bytes(hashlib.sha256(ident.encode()).digest()[:4],'little'));p=rng.permutation(len(g));return g[p][:,p]
def rca(root,events,length,c):
    out=[];excluded=[]
    for e in events:
        if not e.get('applied',True):continue
        mask=np.zeros(length,bool)
        for lo,hi in e['annotation_intervals']:mask[lo:hi]=True
        mask=mask[FIRST:]
        if not mask.any() or not e['root_channels']:excluded.append(e['event_id']);continue
        mean=root[mask].mean(0);order=np.argsort(-mean,kind='stable');ranks=[int(np.flatnonzero(order==k)[0])+1 for k in e['root_channels']]
        out.append(dict(event_id=e['event_id'],root_ranks=ranks,Top1=int(min(ranks)==1),Top3=int(min(ranks)<=min(3,c)),MRR=1/min(ranks),ready_points=int(mask.sum()),mean_root_probabilities=mean.tolist()))
    return out,excluded
def run(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    plan=json.loads(a.plan.read_text());models={};hashes={};weightsha={};records=[];slots=queries=factorizations=0;start=time.time()
    for arm in ARMS:
        for seed in SEEDS:
            name=arm+'_s'+str(seed);p=Path(plan['weights'])/(name+'.pth');assert sha256(p)==plan['weight_SHA256'][name];ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['updates']==128 and ck['input_exposure']==7340032
            m=CausalMetaBasis(seed).cuda().eval().requires_grad_(False);m.load_state_dict(ck['model'],strict=True);models[name]=m;hashes[name]=parameter_sha(m);weightsha[name]=sha256(p)
    for row in plan['rows'][a.worker::a.workers]:
        assert time.time()<plan['deadline_unix'];assert row['split']=='development' and row['pilot100'] and row['channels']>1
        with (Path(plan['benchmark'])/'data'/row['shard']).open('rb') as f:sample=pickle.load(f)[row['offset']]
        assert sample['system_id']==row['system_id'];x=np.asarray(sample['time_series']);normal=np.asarray(sample['normal_time_series']);assert array_sha(x)==row['time_series_sha256'] and array_sha(normal)==row['normal_sha256']
        values=torch.tensor(np.stack([x,normal]),device='cuda',dtype=torch.float64);g=adjacency(sample['graph'],row['channels']);scores={};roots={};stats={};graphsha={}
        for mode in MODES:
            gg=g if mode=='true' else np.zeros_like(g) if mode=='empty' else wrong(g,row['system_id']);graph=torch.tensor(gg,dtype=torch.float64,device='cuda');graphsha[mode]=array_sha(gg)
            for name,m in models.items():
                with torch.no_grad():o=m(values,graph)
                tag=name+'__'+mode;scores[tag]=o['score'].cpu().numpy();roots[tag]=o['root'][0].cpu().numpy();q=(o['residual'].square()/o['variance']).cpu().numpy();stats[tag]=dict(normal_NLL=float(o['normal_nll'][1].mean()),normal_mean_whitened_square=float(q[1].mean()),normal_q99=float(np.quantile(q[1],.99)))
                slots+=values.numel();queries+=2*(len(x)-FIRST)*row['channels'];factorizations+=o['context_factorizations']
        # Baseline is copied after SHA checks, never recalculated or evaluated for RCA.
        old=Path(plan['baseline'])/'development';oldmeta=json.loads((old/'inference'/(row['id']+'.json')).read_text());p=old/'scores'/(row['id']+'.npz');assert sha256(p)==oldmeta['scores_SHA256']
        with np.load(p) as f:scores['B0']=np.stack([f['B0'],f['B0_normal']])
        root_records={};excluded={}
        for name,r in roots.items():root_records[name],excluded[name]=rca(r,sample['events'],len(x),row['channels'])
        p=a.root/'inference'/row['id'];p.parent.mkdir(exist_ok=True);assert not p.with_suffix('.npz').exists();np.savez_compressed(p.with_suffix('.npz'),**scores)
        record=dict(id=row['id'],system_id=row['system_id'],input_observed_SHA256=array_sha(x),input_normal_SHA256=array_sha(normal),used_graph_SHA256=graphsha,weight_SHA256=weightsha,scores_SHA256=sha256(p.with_suffix('.npz')),B0_array_SHA256=array_sha(scores['B0']),B0_reference_file_SHA256=oldmeta['scores_SHA256'],RCA=root_records,RCA_excluded=excluded,normal_prediction_stats=stats,first_query=320,target_updates=0,RCA_scope='conditional true event window; full per-time root arrays not persisted; deterministic rerun possible from checkpoints/input hashes')
        atomic_json(p.with_suffix('.json'),record);records.append(dict(id=row['id'],SHA256=sha256(p.with_suffix('.json')),scores_SHA256=record['scores_SHA256']));atomic_json(a.root/('worker'+str(a.worker)+'_progress.json'),dict(status='running',completed=len(records),full_input_scalar_slots=slots,query_scalar_forwards=queries,context_factorizations=factorizations))
    assert all(parameter_sha(m)==hashes[k] for k,m in models.items());atomic_json(a.root/('worker'+str(a.worker)+'_manifest.json'),dict(status='complete',records=records,full_input_scalar_slots=slots,query_scalar_forwards=queries,context_factorizations=factorizations,parameters_unchanged=True,global_parameter_SHA256=hashes,weight_SHA256=weightsha,plan_SHA256=sha256(a.plan),target_updates=0,TimeRCD_new_forwards=0,validation_or_locked_arrays_opened=False,seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    atomic_json(a.root/('worker'+str(a.worker)+'_progress.json'),dict(status='complete',completed=len(records),full_input_scalar_slots=slots,query_scalar_forwards=queries,context_factorizations=factorizations))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,required=True);a=p.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors'/('worker'+str(a.worker)+'.json'),dict(error=traceback.format_exc()));raise
