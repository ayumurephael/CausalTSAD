"""Shared frozen forwards, independent timing and parent-feature readouts."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,pickle,traceback,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from causal_tsad.io import atomic_json,sha256,array_sha,adjacency
from proposals.causal_meta_basis import CausalMetaBasis,FIRST
from proposals.innovation_readout import readouts
from tools.train_meta_basis import nuisance,graph_for,psha

def extension(path):
    spec=importlib.util.spec_from_file_location('conditional_change_evidence',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def wrong(g,key):
    import hashlib
    rng=np.random.RandomState(int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little'));p=rng.permutation(len(g));return g[p][:,p]
def summarize(score,root,labels,events):
    out=[]
    for i in range(len(score)):
        y=labels[i,FIRST:];valid=bool(y.any() and not y.all());row=dict(view=i,AP=float(average_precision_score(y,score[i])) if valid else None,
            ROC_AUC=float(roc_auc_score(y,score[i])) if valid else None,finite=bool(np.isfinite(score[i]).all()))
        rca=[]
        for event in events[i]:
            mask=np.zeros(labels.shape[1],bool)
            for lo,hi in event['intervals']:mask[lo:hi]=True
            mask=mask[FIRST:]
            if mask.any() and event['roots']:
                rp=root[i,mask].mean(0);order=np.argsort(-rp,kind='stable');rank=min(int(np.flatnonzero(order==j)[0])+1 for j in event['roots'])
                rca.append(dict(Top1=int(rank==1),Top3=int(rank<=min(3,len(order))),MRR=1/rank,points=int(mask.sum()),root_probabilities=rp.tolist()))
        row['RCA']=rca;out.append(row)
    return out
def load(row,plan):
    kind=row['kind']
    if kind=='source':
        p=Path(plan['source'])/row['path'];assert sha256(p)==row['SHA256']
        with np.load(p) as f:x=f['cases'];g=f['graph'];labels=f['labels']
        assert array_sha(x)==row['cases_SHA256'] and array_sha(g)==row['graph_SHA256']
        events=[[dict(intervals=[m['interval']],roots=[m['root']])] if m['root'] is not None else [] for m in row['case_metadata']]
    elif kind=='calibration':
        p=Path(plan['calibration_source'])/row['path'];assert sha256(p)==row['SHA256']
        with np.load(p) as f:original=f['cases'];g=f['graph']
        x=np.concatenate([nuisance(original,row['key'],bg)[:2] for bg in ['plain','smooth']]);labels=np.zeros(x.shape[:2],np.int8);events=[[] for _ in x]
    else:
        assert kind=='target' and row['split']=='development' and row['pilot100'] and row['channels']>1
        with (Path(plan['benchmark'])/'data'/row['shard']).open('rb') as f:s=pickle.load(f)[row['offset']]
        assert s['system_id']==row['system_id'];x=np.stack([s['time_series'],s['normal_time_series']]);g=adjacency(s['graph'],row['channels']);labels=np.stack([s['labels'],np.zeros(len(x[0]),np.int8)])
        assert array_sha(x[0])==row['time_series_sha256'] and array_sha(x[1])==row['normal_sha256']
        events=[[dict(intervals=e['annotation_intervals'],roots=e['root_channels']) for e in s['events'] if e.get('applied',True)],[]]
    assert len(x)==len(events) and np.isfinite(x).all()
    return x,g,labels,events

def run(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    plan=json.loads(a.plan.read_text());ext=extension(plan['conditional_code']+'/proposals/conditional_change_evidence.py')
    models={};initial={};slots=queries=factors=headfactors=0;records=[];start=time.time();cachebytes=0;maximum_old_error=0.
    for seed in [348,349,350]:
        name='normal_only_s'+str(seed);p=Path(plan['weights'])/(name+'.pth');assert sha256(p)==plan['weight_SHA256'][name]
        ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['updates']==128 and ck['input_exposure']==7340032
        m=CausalMetaBasis(seed).cuda().eval().requires_grad_(False);m.load_state_dict(ck['model']);models[seed]=m;initial[seed]=psha(m)
    for row in plan['tasks'][a.worker::a.workers]:
        assert time.time()<plan['deadline_unix'];x,g,labels,events=load(row,plan);key=row.get('key',row.get('system_id'));ident=row['task_id'];xgpu=torch.tensor(x,device='cuda',dtype=torch.float64)
        scorecache={};metrics={};cachemeta=[];audits={};normalstats={};inputsha=array_sha(x)
        for mode in ['true','empty','rewired']:
            gg=(g if mode=='true' else np.zeros_like(g) if mode=='empty' else wrong(g,row['system_id'])) if row['kind']=='target' else graph_for(g,row['key'],mode)
            graph=torch.tensor(gg,device='cuda',dtype=torch.float64)
            for seed,model in models.items():
                with torch.no_grad():o=model(xgpu,graph);heads,error=readouts(model,o)
                assert all(torch.isfinite(o[k]).all() for k in ['score','mean','variance','residual']);assert (o['variance']>0).all()
                if row['kind']=='target':
                    oldpath=Path(plan['old_target'])/'inference'/(row['id']+'.npz');oldmeta=json.loads(oldpath.with_suffix('.json').read_text());assert sha256(oldpath)==oldmeta['scores_SHA256']
                    with np.load(oldpath) as f:old=f['normal_only_s'+str(seed)+'__'+mode]
                    delta=float(np.max(np.abs(heads['trailing16']['score'].cpu().numpy()-old)));assert delta<=1e-8;maximum_old_error=max(maximum_old_error,delta)
                else:maximum_old_error=max(maximum_old_error,error)
                if row['kind']!='target' and mode=='true':
                    with torch.no_grad():more,audit=ext.evidence_heads(xgpu,graph,o['residual'],o['variance'],str(key))
                    heads.update({'conditional_'+k:v for k,v in more.items()});audits[str(seed)]=audit;headfactors+=audit['factorization_count']
                for h,v in heads.items():
                    tag='s'+str(seed)+'__'+mode+'__'+h;score=v['score'].cpu().numpy();root=v['root'].cpu().numpy()
                    assert np.isfinite(score).all() and np.isfinite(root).all();scorecache[tag]=score
                    metrics[tag]=summarize(score,root,labels,events)
                normalviews=[0,1] if row['kind']=='source' else [1] if row['kind']=='target' else list(range(len(x)))
                normalstats['s'+str(seed)+'__'+mode]=dict(normal_NLL=float(o['normal_nll'][normalviews].mean()),normal_mean_whitened_square=float((o['residual'][normalviews].square()/o['variance'][normalviews]).mean()))
                cache=a.root/'innovation_cache'/(ident+'__s'+str(seed)+'__'+mode+'.npz');assert not cache.exists()
                np.savez_compressed(cache,residual=o['residual'].cpu().numpy(),variance=o['variance'].cpu().numpy())
                cachemeta.append(dict(path=str(cache.relative_to(a.root)),SHA256=sha256(cache),bytes=cache.stat().st_size));cachebytes+=cache.stat().st_size
                slots+=x.size;queries+=len(x)*(x.shape[1]-FIRST)*x.shape[-1];factors+=o['context_factorizations']
        dst=a.root/'records'/ident;assert not dst.with_suffix('.npz').exists();np.savez_compressed(dst.with_suffix('.npz'),**scorecache)
        atomic_json(dst.with_suffix('.json'),dict(task=row,input_SHA256=inputsha,graph_SHA256=array_sha(g),labels_SHA256=array_sha(labels),metrics=metrics,
            scores_SHA256=sha256(dst.with_suffix('.npz')),innovation_cache=cachemeta,conditional_audit=audits,normal_stats=normalstats,
            old_head_maximum_error=maximum_old_error,weight_SHA256=plan['weight_SHA256'],target_updates=0,
            RCA_scope='conditional true injection/annotation window; not automatic detection to localization'))
        records.append(dict(id=ident,path=str(dst.with_suffix('.json').relative_to(a.root)),SHA256=sha256(dst.with_suffix('.json')),scores_SHA256=sha256(dst.with_suffix('.npz'))))
        atomic_json(a.root/('worker'+str(a.worker)+'_progress.json'),dict(status='running',completed=len(records),full_input_slots=slots,query_slots=queries))
    assert all(psha(m)==initial[seed] for seed,m in models.items())
    manifest=dict(status='complete',records=records,full_input_slots=slots,query_slots=queries,context_factorizations=factors,
        conditional_head_factorizations=headfactors,innovation_cache_bytes=cachebytes,old_head_maximum_error=maximum_old_error,
        parameters_unchanged=True,global_parameter_SHA256=initial,weight_SHA256=plan['weight_SHA256'],plan_SHA256=sha256(a.plan),
        optimizer_updates=0,target_updates=0,validation_or_locked_arrays_opened=False,seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
    atomic_json(a.root/('worker'+str(a.worker)+'_manifest.json'),manifest)
    atomic_json(a.root/('worker'+str(a.worker)+'_progress.json'),dict(status='complete',completed=len(records),full_input_slots=slots,query_slots=queries))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,required=True);a=p.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors'/('worker'+str(a.worker)+'.json'),dict(error=traceback.format_exc()));raise
