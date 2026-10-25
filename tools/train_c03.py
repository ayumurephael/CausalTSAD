"""C03 normal-mechanism source probe; no target or TimeRCD dependency."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256
from proposals.c03.data import batch_order,tensors


def model_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load(cache,split):
    manifest=json.loads((cache/'manifest.json').read_text());assert manifest['status']=='complete' and manifest['source_only']
    pool={}
    for row in manifest['files']:
        if row['split']!=split:continue
        path=cache/row['path'];assert sha256(path)==row['SHA256']
        with np.load(path) as f:pool[row['channels']]={k:np.array(f[k]) for k in f.files}
    return pool


def evaluate(model,pool,arm,config):
    import torch
    from sklearn.metrics import average_precision_score
    from proposals.c03.model import loss_and_score
    records={};ys=[];scores=[];normal=[];query_slots=0;swap_slots=0
    model.eval()
    with torch.no_grad():
        for c,indices in batch_order(pool,config['batch_input_scalars'],9002):
            pack=pool[c];normal_batch=tensors(pack,indices,arm,'normal');nll,sn=loss_and_score(model,normal_batch)
            observed=tensors(pack,indices,arm,'x');_,sx=loss_and_score(model,observed)
            swapped=tensors(pack,indices,arm,'normal',swap=True);swap_nll,_=loss_and_score(model,swapped)
            valid=normal_batch['valid'].bool();ready=normal_batch['ready'].bool()&valid
            ys.extend([np.zeros(int(valid.sum()),np.int8),observed['labels'][valid].cpu().numpy()])
            scores.extend([sn[valid].cpu().numpy(),sx[valid].cpu().numpy()]);normal.append(sn[valid].cpu().numpy())
            for local,index in enumerate(indices):
                v=valid[local];r=ready[local]
                records[f'c{c}/{int(index)}']=dict(channels=c,normal_NLL=float(nll[local][v].mean()),
                    ready_NLL=float(nll[local][r].mean()),swap_ready_NLL=float(swap_nll[local][r].mean()),
                    normal_score_mean=float(sn[local][v].mean()),ready_scalar_count=int(r.sum())*c)
            query_slots+=int(valid.sum())*c*2;swap_slots+=int(valid.sum())*c
    threshold=float(np.quantile(np.concatenate(normal),.99))
    return dict(normal_NLL_macro=float(np.mean([r['normal_NLL'] for r in records.values()])),
        ready_NLL_macro=float(np.mean([r['ready_NLL'] for r in records.values()])),
        swap_ready_NLL_macro=float(np.mean([r['swap_ready_NLL'] for r in records.values()])),
        density_AP=float(average_precision_score(np.concatenate(ys),np.concatenate(scores))),threshold=threshold,
        normal_FPR=float(np.mean(np.concatenate(normal)>threshold)),records=records,
        source_normal_and_observed_frozen_query_scalars=query_slots,extra_swap_frozen_query_scalars=swap_slots,
        target_arrays_opened=False,note='Normal/swapped clean contexts are source-only diagnostics. Test input is never a normal twin.')


def train(a):
    import torch
    from proposals.c03.model import ContextMechanism,loss_and_score
    cfg=json.loads(a.config.read_text());torch.set_num_threads(1);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(cfg['memory_gib_per_worker']*2**30/torch.cuda.get_device_properties(0).total_memory)
    assert not a.output.exists();a.output.mkdir(parents=True)
    training=load(a.cache,'train');validation=load(a.cache,'source_validation')
    model=ContextMechanism(cfg['history'],cfg['horizon'],cfg['width'],cfg['message_width']).cuda()
    order=batch_order(training,cfg['batch_input_scalars'],cfg['data_seed']);initial=model_sha(model)
    c,w,m,h=1,cfg['width'],cfg['message_width'],cfg['history']
    linear_flops=sum(2*len(p['normal_raw'])*64*(channels*((h+1)*w+(w+m+1)*w+w*2*cfg['horizon'])+
        channels**2*((2*h+8)*m+m*m)) for channels,p in training.items())
    status=dict(status='running',arm=a.arm,seed=a.seed,pid=os.getpid(),updates=0,total_updates=len(order),
        optimized_normal_query_scalars=0,anomalous_observed_training_scalars=0,extra_source_evaluation_query_scalars=0,
        cache_manifest_sha256=sha256(a.cache/'manifest.json'),config_sha256=sha256(a.config),initial_parameter_sha256=initial,
        parameters=sum(p.numel() for p in model.parameters()),normal_training_Linear_forward_FLOPs=linear_flops,
        local_commit=a.local_commit,github_commit=a.github_commit,target_arrays_in_gradient=False,
        scope='source-only normal mechanism probe, not a TimeRCD improvement')
    atomic_json(a.output/'progress.json',status);started=time.monotonic()
    init=evaluate(model,validation,a.arm,cfg);atomic_json(a.output/'initial_source_validation.json',init)
    status['extra_source_evaluation_query_scalars']+=init['source_normal_and_observed_frozen_query_scalars']+init['extra_swap_frozen_query_scalars']
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    for i,(channels,indices) in enumerate(order,1):
        if time.monotonic()-started>cfg['max_train_hours_per_run']*3600:raise TimeoutError('Fixed source-probe budget')
        model.train();b=tensors(training[channels],indices,a.arm,'normal');optimizer.zero_grad(set_to_none=True)
        nll,_=loss_and_score(model,b);loss=nll[b['valid'].bool()].mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        optimizer.step();status['optimized_normal_query_scalars']+=int(b['valid'].sum())*channels
        status.update(updates=i,last_normal_NLL=float(loss),seconds=time.monotonic()-started)
        with (a.output/'learning_curve.jsonl').open('a') as f:f.write(json.dumps({k:status[k] for k in ['updates','last_normal_NLL','optimized_normal_query_scalars','seconds']})+'\n')
        if i==1 or i%20==0 or i==len(order):atomic_json(a.output/'progress.json',status)
    assert status['optimized_normal_query_scalars']==cfg['normal_training_scalars_per_run']
    final=evaluate(model,validation,a.arm,cfg);atomic_json(a.output/'final_source_validation.json',final)
    status['extra_source_evaluation_query_scalars']+=final['source_normal_and_observed_frozen_query_scalars']+final['extra_swap_frozen_query_scalars']
    temp=a.output/'final.partial';torch.save(dict(model=model.state_dict(),config=cfg,arm=a.arm,seed=a.seed,
        source_calibration=final,cache_manifest_sha256=sha256(a.cache/'manifest.json')),temp);os.replace(temp,a.output/'final.pth')
    status.update(status='complete',final_parameter_sha256=model_sha(model),checkpoint_sha256=sha256(a.output/'final.pth'),
        final_normal_NLL=final['normal_NLL_macro'],final_ready_NLL=final['ready_NLL_macro'],
        final_swap_ready_NLL=final['swap_ready_NLL_macro'],final_source_density_AP=final['density_AP'],peak_cuda_bytes=torch.cuda.max_memory_allocated())
    atomic_json(a.output/'progress.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--local-commit',required=True);p.add_argument('--github-commit',required=True);a=p.parse_args()
    try:train(a)
    except Exception:
        atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
