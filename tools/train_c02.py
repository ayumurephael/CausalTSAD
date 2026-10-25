"""Fixed normal-mechanism timing probe, independent source systems only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256
from proposals.c02.data import batch_order,tensors


def state_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load(source,split,expected):
    assert sha256(source/'manifest.json')==expected
    manifest=json.loads((source/'manifest.json').read_text());assert manifest['status']=='complete'
    pool={}
    for r in manifest['files']:
        if r['split']!=split:continue
        assert sha256(source/r['path'])==r['SHA256']
        with np.load(source/r['path']) as f:pool[r['channels']]={k:np.array(f[k]) for k in f.files}
    return pool


def evaluate(model,pool,arm,cfg):
    import torch
    from sklearn.metrics import average_precision_score
    from proposals.c02.model import loss_and_score
    model.eval();records={};ys=[];scores=[];normal_scores=[];query_slots=0
    with torch.no_grad():
        for c,indices in batch_order(pool,cfg['batch_input_scalars'],9002):
            pack=pool[c];normal=tensors(pack,indices,arm,'normal');observed=tensors(pack,indices,arm,'x')
            nll,sn,_=loss_and_score(model,normal);_,sx,node=loss_and_score(model,observed)
            interventions={}
            for mode in ['block','past','current']:
                alt,_,_=loss_and_score(model,normal,timing=mode);interventions[mode]=alt[:,64:].mean((-1,-2)).cpu().numpy()
            query_slots+=5*normal['values'].numel()
            yn=pack['labels'][indices]
            ys.extend([np.zeros(yn.size,np.int8),yn.ravel()]);scores.extend([sn.cpu().numpy().ravel(),sx.cpu().numpy().ravel()])
            normal_scores.append(sn.cpu().numpy().ravel())
            for local,index in enumerate(indices):
                y=yn[local];event=pack['root_window'][index].astype(bool);ranks=None
                if c>1 and event.any():
                    rank=np.argsort(-node[local].cpu().numpy()[event].mean(0),kind='stable')
                    position=int(np.flatnonzero(rank==int(pack['root'][index]))[0])+1
                    ranks=dict(Top1=int(position==1),Top3=int(position<=min(3,c)),MRR=1/position)
                records[f'c{c}/{int(index)}']=dict(channels=c,normal_NLL=float(nll[local].mean()),ready_NLL=float(nll[local,64:].mean()),
                    timing_intervention_ready_NLL={k:float(v[local]) for k,v in interventions.items()},
                    observed_AP=float(average_precision_score(y,sx[local].cpu().numpy())) if len(np.unique(y))>1 else None,
                    conditional_RCA=ranks,normal_score_mean=float(sn[local].mean()))
    if arm=='none':
        assert all(max(r['timing_intervention_ready_NLL'].values())-min(r['timing_intervention_ready_NLL'].values())<1e-6 for r in records.values())
    threshold=float(np.quantile(np.concatenate(normal_scores),.99))
    return dict(normal_NLL_macro=float(np.mean([r['normal_NLL'] for r in records.values()])),
        ready_NLL_macro=float(np.mean([r['ready_NLL'] for r in records.values()])),records=records,
        innovation_AP=float(average_precision_score(np.concatenate(ys),np.concatenate(scores))),threshold=threshold,
        normal_FPR=float(np.mean(np.concatenate(normal_scores)>threshold)),
        actual_frozen_query_scalar_forwards=query_slots,
        logical_head_history_scalar_slots=2*query_slots*cfg['history'],target_arrays_opened=False,
        score_rule='max log1p standardized squared conditional residual; not raw NLL score',
        RCA_scope='source only, true injection interval supplied for scoring, C1 excluded')


def train(a):
    import torch
    from proposals.c02.model import ParentTiming,loss_and_score
    cfg=json.loads(a.config.read_text());torch.set_num_threads(1);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(cfg['memory_gib_per_worker']*2**30/torch.cuda.get_device_properties(0).total_memory)
    assert not a.output.exists();a.output.mkdir(parents=True)
    training=load(a.source,'train',cfg['source_manifest_sha256']);validation=load(a.source,'source_validation',cfg['source_manifest_sha256'])
    model=ParentTiming(cfg['history'],cfg['block'],cfg['width'],cfg['message_width']).cuda()
    order=batch_order(training,cfg['batch_input_scalars'],cfg['data_seed']);initial=state_sha(model)
    h,w,m,p=cfg['history'],cfg['width'],cfg['message_width'],cfg['block']
    flops=sum(2*len(pack['normal'])*1024*(c*((h+1)*w+(w+m+1)*w+w*2*p)+c*c*(2*h*m+m*m)) for c,pack in training.items())
    status=dict(status='running',arm=a.arm,seed=a.seed,pid=os.getpid(),updates=0,total_updates=len(order),
        initial_parameter_sha256=initial,parameters=sum(p.numel() for p in model.parameters()),
        source_manifest_sha256=cfg['source_manifest_sha256'],config_sha256=sha256(a.config),
        optimized_normal_query_scalars=0,anomalous_observed_training_scalars=0,extra_source_evaluation_query_scalars=0,
        logical_training_head_history_scalar_slots=0,logical_training_own_valid_history_uses=0,
        logical_training_selected_parent_valid_history_uses=0,
        normal_training_Linear_forward_FLOPs=flops,local_commit=a.local_commit,github_commit=a.github_commit,
        target_arrays_in_gradient=False,TimeRCD_weights_used=False,
        FLOPs_note='Dense Linear forward only; all modes evaluated per point with same16 output slots. No encoder; normalization, gathers, backward and validation separate.')
    atomic_json(a.output/'progress.json',status);started=time.monotonic()
    initial_eval=evaluate(model,validation,a.arm,cfg);atomic_json(a.output/'initial_source_validation.json',initial_eval)
    status['extra_source_evaluation_query_scalars']+=initial_eval['actual_frozen_query_scalar_forwards']
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    for i,(c,indices) in enumerate(order,1):
        if time.monotonic()-started>cfg['max_train_hours_per_run']*3600:raise TimeoutError('Fixed per-run deadline')
        model.train();b=tensors(training[c],indices,a.arm,'normal');optimizer.zero_grad(set_to_none=True)
        nll,_,_=loss_and_score(model,b);loss=nll.mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        optimizer.step();status['optimized_normal_query_scalars']+=b['values'].numel()
        times=np.arange(b['values'].shape[1]);timing=b['timing'];batch_c=len(indices)*c
        status['logical_training_head_history_scalar_slots']+=2*b['values'].numel()*h
        status['logical_training_own_valid_history_uses']+=int(np.minimum(times,h).sum())*batch_c
        parent_counts=np.minimum(times//p*p,h) if timing=='block' else np.minimum(times+(timing=='current'),h)
        status['logical_training_selected_parent_valid_history_uses']+=int(parent_counts.sum())*batch_c
        status.update(updates=i,last_normal_NLL=float(loss),seconds=time.monotonic()-started)
        with (a.output/'learning_curve.jsonl').open('a') as f:f.write(json.dumps({k:status[k] for k in ['updates','last_normal_NLL','optimized_normal_query_scalars','seconds']})+'\n')
        if i==1 or i%20==0 or i==len(order):atomic_json(a.output/'progress.json',status)
    assert status['optimized_normal_query_scalars']==cfg['normal_training_scalars_per_run']
    final=evaluate(model,validation,a.arm,cfg);atomic_json(a.output/'final_source_validation.json',final)
    status['extra_source_evaluation_query_scalars']+=final['actual_frozen_query_scalar_forwards']
    temp=a.output/'final.partial';torch.save(dict(model=model.state_dict(),config=cfg,arm=a.arm,seed=a.seed,source_calibration=final,
        source_manifest_sha256=cfg['source_manifest_sha256']),temp);os.replace(temp,a.output/'final.pth')
    status.update(status='complete',final_parameter_sha256=state_sha(model),checkpoint_sha256=sha256(a.output/'final.pth'),
        final_normal_NLL=final['normal_NLL_macro'],final_ready_NLL=final['ready_NLL_macro'],
        final_source_innovation_AP=final['innovation_AP'],peak_cuda_bytes=torch.cuda.max_memory_allocated())
    atomic_json(a.output/'progress.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--local-commit',required=True);p.add_argument('--github-commit',required=True);a=p.parse_args()
    try:train(a)
    except Exception:
        atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
