"""I06: graph, alignment, input scope and TSAD pooling controls; no training."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from causal_tsad.io import atomic_json,sha256,array_sha
from causal_tsad.frozen_detector import FrozenDetector,parameter_sha
from proposals.causal_meta_basis import CausalMetaBasis
from proposals.context_crossfit import predict,configurations
from proposals.i06_input_controls import detector_scores,changed_residual
from tools.train_meta_basis import graph_for

def metrics(scores,y,row):
    if row['split']=='calibration':return {}
    groups={'relation':[0,1,2,3],'short_measurement':[0,1,4,5],'long_measurement':[0,1,6,7]};out={}
    for bi,bg in enumerate(['plain','smooth']):
        for name,ids in groups.items():
            ids=np.array(ids)+bi*8;yy=y[ids,320:].reshape(-1);s=scores[ids].reshape(-1)
            out[bg+'__'+name]=dict(AP=float(average_precision_score(yy,s)),ROC_AUC=float(roc_auc_score(yy,s)),
                positive_points=int(yy.sum()),total_points=len(yy),prevalence=float(yy.mean()))
    return out

def run(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    plan=json.loads(a.plan.read_text());task=plan['tasks'][a.task];row=task['row'];seed=task['seed'];ident=task['id'];start=time.time()
    torch.cuda.set_per_process_memory_fraction(plan['worker_GiB']*2**30/torch.cuda.get_device_properties(0).total_memory)
    dest=a.root/'records'/ident;assert not dest.with_suffix('.json').exists()
    source=Path(plan['source'])/row['path'];assert sha256(source)==row['SHA256']
    with np.load(source) as z:values=z['values'];labels=z['labels'];g=z['graph']
    assert array_sha(values)==row['input_SHA256'] and array_sha(g)==row['graph_SHA256']
    weight=Path(plan['weights'])/f'normal_only_s{seed}.pth';assert sha256(weight)==plan['weight_SHA256'][str(seed)]
    model=CausalMetaBasis(seed).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False)['model']);initial=parameter_sha(model)
    detector=FrozenDetector(plan['detector_code'],plan['detector_weights']);scores={};cache={};roots={};audits={}
    factors=features=slots=0
    def progress(stage):atomic_json(a.root/'task_progress'/(ident+'.json'),dict(stage=stage,seconds=time.time()-start))
    def detect(tag,vs,first):
        collected={}
        for v in vs:
            out=detector_scores(detector,v,row['C'])
            for k,s in out.items():collected.setdefault(k,[]).append(s[320-first:])
        for k,s in collected.items():scores[f'{tag}__{k}']=np.stack(s)
    for mode in ['true','empty','wrong']:
        progress('causal_'+mode);gg=graph_for(g,row['key'],mode)
        graph=torch.as_tensor(gg,dtype=torch.float64,device='cuda');outs=[]
        for v in values:
            o=predict(model,torch.as_tensor(v,dtype=torch.float64,device='cuda'),graph,dict(configurations()['xfit'],first=32))
            factors+=o['audit']['factorizations'];features+=o['audit']['feature_node_rows'];slots+=v.size
            outs.append({k:o[k].cpu().numpy() for k in ['residual','variance','baseline']})
        for k in outs[0]:cache[mode+'__'+k]=np.stack([v[k] for v in outs])
        res=cache[mode+'__residual']/np.sqrt(cache[mode+'__variance'])
        e=.5*((2/3)*res[:,288:]**2-np.log(3));temp=float(model.log_temperature.exp().clamp(.1,10));bias=float(model.bias)
        from scipy.special import logsumexp,softmax
        scores['causal__'+mode]=temp*(logsumexp(e/temp,axis=-1)-np.log(row['C']))+bias
        if row['split']=='source_validation':
            rr=[]
            for b in [0,8]:
                for view,lo,hi in [(2,320,1024),(3,320,1024),(4,512,513),(5,512,513),(6,512,640),(7,512,640)]:
                    p=softmax(e[b+view,lo-320:hi-320],axis=-1).mean(0)
                    rank=int(np.flatnonzero(np.argsort(-p,kind='stable')==row['root'])[0])+1
                    rr.append(dict(view=b+view,Top1=int(rank==1),Top3=int(rank<=3),MRR=1/rank))
            roots[mode]=rr
        audits[mode]=dict(graph_SHA256=array_sha(gg),same_as_true=bool(np.array_equal(gg,g)))
        for first in [320,32]:
            r=res[:,first-32:];x=values[:,first:]
            for alignment in (['aligned','channel','time'] if mode=='true' else ['aligned']):
                progress(f'detector_{mode}_{alignment}_{first}')
                pair=[np.concatenate([v,changed_residual(z,row['key'],alignment)],axis=1) for v,z in zip(x,r)]
                detect(f'xres_{mode}_{alignment}__from{first}',pair,first)
    if seed==348:
        for first in [320,32]:
            raw=values[:,first:]
            for name,vs in [('raw',raw),('duplicate',[np.concatenate([v,v],axis=1) for v in raw]),('zero',[np.concatenate([v,np.zeros_like(v)],axis=1) for v in raw])]:
                progress(name+str(first));detect(f'{name}__from{first}',vs,first)
    detector.verify_frozen();assert parameter_sha(model)==initial
    assert all(np.isfinite(v).all() and v.shape==(len(values),704) for v in scores.values())
    np.savez_compressed(dest.with_suffix('.npz'),**scores)
    np.savez_compressed(a.root/'context_cache'/(ident+'.npz'),**cache)
    atomic_json(dest.with_suffix('.json'),dict(task=task,input_SHA256=array_sha(values),labels_SHA256=array_sha(labels),
        basis_weight_SHA256=sha256(weight),basis_parameter_SHA256=initial,parameters_unchanged=True,
        TimeRCD_weight_SHA256=sha256(Path(plan['detector_weights'])/'selftrained_multi.pth'),
        metrics={k:metrics(v,labels,row) for k,v in scores.items()},conditional_causal_RCA=roots,graph_audits=audits,
        scores_SHA256=sha256(dest.with_suffix('.npz')),context_cache_SHA256=sha256(a.root/'context_cache'/(ident+'.npz')),
        budget=dict(causal_input_scalars=slots,feature_node_rows=features,factorizations=factors,
            detector_input_scalars=detector.actual_input_scalars,detector_padded_scalars=detector.padded_forward_scalars,
            detector_calls=detector.encoder_calls,training_exposure=0,optimizer_updates=0),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-start,code=plan['code'],
        benchmark_arrays_opened=False,TimeRCD_RCA=False,target_context_inference='offline purged 8-fold IRLS; future context outside block allowed'))
    progress('complete')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--task',type=int,required=True);a=p.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors'/f'task{a.task}.json',dict(error=traceback.format_exc()));raise
