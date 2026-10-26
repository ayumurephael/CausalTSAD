"""One fixed row/seed task: E1/E2 frozen inference, E3 cached readouts, E4 TSAD.
Metadata enters evaluation only. Calibration uses independent source normal
views, never the benchmark's paired normal view to fit an anomalous case.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import sys,json,pickle,time,argparse,traceback,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from causal_tsad.io import atomic_json,sha256,array_sha,adjacency
from causal_tsad.frozen_detector import FrozenDetector,parameter_sha
from proposals.causal_meta_basis import CausalMetaBasis
from proposals.context_crossfit import predict,configurations,FIRST
from proposals.relative_context_scores import readouts
from tools.train_meta_basis import nuisance,graph_for

def wrong(g,key):
    rng=np.random.RandomState(int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:4],'little'))
    p=rng.permutation(len(g));return g[p][:,p]

def rca(root,events,length,c):
    out=[];excluded=[]
    for event in events:
        if not event.get('applied',True):continue
        mask=np.zeros(length,bool)
        for lo,hi in event['annotation_intervals']:mask[lo:hi]=True
        mask=mask[FIRST:]
        if not mask.any() or not event['root_channels']:excluded.append(event['event_id']);continue
        mean=root[mask].mean(0);order=np.argsort(-mean,kind='stable')
        ranks=[int(np.flatnonzero(order==k)[0])+1 for k in event['root_channels']]
        out.append(dict(event_id=event['event_id'],root_ranks=ranks,Top1=int(min(ranks)==1),Top3=int(min(ranks)<=min(3,c)),
                        MRR=1/min(ranks),ready_points=int(mask.sum()),mean_root_probabilities=mean.tolist()))
    return out,excluded

def load(row,plan):
    if row['kind']=='calibration':
        p=Path(plan['calibration_source'])/row['path'];assert sha256(p)==row['SHA256']
        with np.load(p) as f:raw=f['cases'];g=f['graph']
        x=np.concatenate([nuisance(raw,row['key'],bg)[:2] for bg in ['plain','smooth']])
        y=np.zeros(x.shape[:2],np.int8);events=[[] for _ in x]
    else:
        assert row['split']=='development' and row['pilot100'] and row['channels']>1
        with (Path(plan['benchmark'])/'data'/row['shard']).open('rb') as f:s=pickle.load(f)[row['offset']]
        assert s['system_id']==row['system_id'];x=np.stack([s['time_series'],s['normal_time_series']])
        assert array_sha(x[0])==row['time_series_sha256'] and array_sha(x[1])==row['normal_sha256']
        g=adjacency(s['graph'],row['channels']);y=np.stack([s['labels'],np.zeros(len(x[0]),np.int8)])
        events=[s['events'],[]]
    assert np.isfinite(x).all()
    return x,g,y,events

def evaluate(score,root,y,events):
    rows=[]
    for view,s in enumerate(score):
        yy=y[view,FIRST:];eligible=bool(yy.any() and not yy.all())
        out=dict(view=view,AP=float(average_precision_score(yy,s)) if eligible else None,
                 ROC_AUC=float(roc_auc_score(yy,s)) if eligible else None)
        if root is not None:out['RCA'],out['excluded_RCA']=rca(root[view],events[view],len(y[view]),root.shape[-1])
        rows.append(out)
    return rows

def run(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    plan=json.loads(a.plan.read_text());task=plan['tasks'][a.task];row=task['row'];seed=task['seed'];ident=task['id'];start=time.time()
    torch.cuda.set_per_process_memory_fraction(plan.get('worker_GiB',8)*2**30/torch.cuda.get_device_properties(0).total_memory)
    dst=a.root/'records'/ident;assert not dst.with_suffix('.json').exists()
    x,g,y,events=load(row,plan);source=row['kind']=='calibration';key=str(row.get('system_id',row.get('key')))
    weight=Path(plan['weights'])/f'normal_only_s{seed}.pth';assert sha256(weight)==plan['weight_SHA256'][f'normal_only_s{seed}']
    model=CausalMetaBasis(seed).cuda().eval().requires_grad_(False);model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False)['model'])
    initial=parameter_sha(model);temp=float(model.log_temperature.exp().clamp(.1,10));bias=float(model.bias)
    scores={};metrics={};audits={};contexts={};factors=feature_rows=raw_slots=0;equivalence=0.
    def put(tag,read):
        score=np.stack([v['score'] for v in read]);root=np.stack([v['root'] for v in read])
        assert np.isfinite(score).all() and np.isfinite(root).all();scores[tag]=score;metrics[tag]=evaluate(score,root,y,events)
    def progress(stage):atomic_json(a.root/'task_progress'/(ident+'.json'),dict(status='running',stage=stage,seconds=time.time()-start))
    configs=configurations()
    for mode in ['true','empty','rewired']:
        gg=graph_for(g,row['key'],mode) if source else g if mode=='true' else np.zeros_like(g) if mode=='empty' else wrong(g,key)
        graph=torch.tensor(gg,dtype=torch.float64,device='cuda')
        # Reuse exact old prefix residuals where available; verify file SHA.
        oldident=row.get('task_id','target_'+row.get('id',''))
        old=Path(plan['probe'])/'innovation_cache'/f'{oldident}__s{seed}__{mode}.npz'
        oldrecord=Path(plan['probe'])/'records'/(oldident+'.json')
        prefix=[]
        if old.is_file() and oldrecord.is_file():
            meta=json.loads(oldrecord.read_text());entry=next(e for e in meta['innovation_cache'] if Path(e['path']).name==old.name);assert sha256(old)==entry['SHA256']
            assert meta['input_SHA256']==array_sha(x)
            with np.load(old) as f:rr=f['residual'];vv=f['variance']
            prefix=[dict(residual=rr[i],variance=vv[i]) for i in range(len(x))]
            audits['prefix_'+mode]=dict(cache=str(old),SHA256=sha256(old),reused=True)
        else:
            for v in x:
                with torch.no_grad():out=model(torch.tensor(v,device='cuda',dtype=torch.float64)[None],graph)
                prefix.append(dict(residual=out['residual'][0].cpu().numpy(),variance=out['variance'][0].cpu().numpy()))
                factors+=out['context_factorizations'];raw_slots+=v.size
            audits['prefix_'+mode]=dict(reused=False)
        put('prefix__'+mode+'__point',[readouts(v['residual'],v['variance'],temp,bias)['point'] for v in prefix])
        selected=list(configs) if mode=='true' else ['xfit','online','period_parent32']
        for name in selected:
            progress(name+'_'+mode);outviews=[]
            for v in x:
                out=predict(model,torch.tensor(v,device='cuda',dtype=torch.float64),graph,configs[name])
                factors+=out['audit']['factorizations'];feature_rows+=out['audit']['feature_node_rows'];raw_slots+=v.size
                outviews.append({k:out[k].cpu().numpy() if k!='audit' else out[k] for k in out})
            audits[name+'__'+mode]=[v['audit'] for v in outviews]
            reads=[readouts(v['residual'],v['variance'],temp,bias) for v in outviews]
            for readname in reads[0] if name in ['xfit','period_parent32'] else ['point']:
                put(name+'__'+mode+'__'+readname,[r[readname] for r in reads])
            if name=='xfit':contexts[mode]=outviews
    # E4 uses the common primary xfit residual. No root outputs from TimeRCD.
    detector=FrozenDetector(plan['detector_code'],plan['detector_weights']);progress('E4')
    def detput(tag,values,kind):
        output=[]
        for v in values:
            if kind=='uni':
                per=[detector.predict(v[:,j:j+1],'selftrained_uni','native')['logit'] for j in range(v.shape[1])]
                output.append(np.max(np.stack(per),axis=0))
            else:output.append(detector.predict(v,'selftrained_multi','native')['logit'])
        scores[tag]=np.stack(output);metrics[tag]=evaluate(scores[tag],None,y,events)
    # Matched suffix input for raw/residual/parent-removed inputs. Original full
    # sequence TimeRCD reference is added separately in analysis, not conflated.
    raw=x[:,FIRST:]
    if seed==348:
        # These inputs do not depend on the normal-basis seed. Run once only;
        # analysis shares this reference with the two other paired seeds.
        detput('E4_raw_uni',raw,'uni');detput('E4_raw_multi',raw,'multi')
        detput('E4_duplicate_multi',[np.concatenate([v,v],axis=1) for v in raw],'multi')
    for mode,outs in contexts.items():
        res=[v['residual']/np.sqrt(v['variance']) for v in outs]
        detput('E4_res_uni__'+mode,res,'uni')
        if mode!='empty':detput('E4_parentremoved_uni__'+mode,[v['baseline'] for v in outs],'uni')
        detput('E4_x_res_multi__'+mode,[np.concatenate([v,r],axis=1) for v,r in zip(raw,res)],'multi')
    detector.verify_frozen();assert parameter_sha(model)==initial
    # Persist residuals for secondary diagnostics; never regenerate normal basis
    # just to alter a readout. Keep per-view inference independent.
    cache={}
    for mode,outs in contexts.items():
        for k in ['residual','variance','baseline']:cache[mode+'__'+k]=np.stack([v[k] for v in outs])
    np.savez_compressed(a.root/'context_cache'/(ident+'.npz'),**cache)
    np.savez_compressed(dst.with_suffix('.npz'),**scores)
    atomic_json(dst.with_suffix('.json'),dict(task=task,input_SHA256=array_sha(x),graph_SHA256=array_sha(g),labels_SHA256=array_sha(y),
        weight_SHA256=sha256(weight),parameter_SHA256=initial,parameters_unchanged=True,scores_SHA256=sha256(dst.with_suffix('.npz')),
        context_cache_SHA256=sha256(a.root/'context_cache'/(ident+'.npz')),metrics=metrics,audits=audits,
        budget=dict(raw_input_slots=raw_slots,feature_node_rows=feature_rows,context_factorizations=factors,
                    TimeRCD_actual_input_scalars=detector.actual_input_scalars,TimeRCD_padded_scalars=detector.padded_forward_scalars,
                    TimeRCD_encoder_calls=detector.encoder_calls,optimizer_updates=0),seconds=time.time()-start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),TimeRCD_RCA=False,validation_locked_opened=False,
        execution_code=plan['code'],worker_GiB=plan.get('worker_GiB',8),
        RCA_scope='conditional supplied true event window, not detected event'))
    atomic_json(a.root/'task_progress'/(ident+'.json'),dict(status='complete',seconds=time.time()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--task',type=int,required=True);a=p.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors'/f'task{a.task}.json',dict(error=traceback.format_exc()));raise
