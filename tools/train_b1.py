"""Bounded B1 training, fixed final checkpoint and source-only calibration."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import sha256, atomic_json, graph_control


def now(): return datetime.now(timezone.utc).isoformat()


def state_sha(model):
    h=hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_pool(root, split, mode):
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='complete'
    pool={}
    for row in manifest['files']:
        if row['split']!=split: continue
        path=root/row['path'];assert sha256(path)==row['SHA256']
        with np.load(path) as z: pack={k:np.array(z[k]) for k in z.files}
        graphs=[]
        for a, seed in zip(pack['adjacency'],pack['seed']):
            permutation=np.random.RandomState(int(seed)^0x46113).permutation(row['channels'])
            graphs.append(graph_control(a,mode,permutation))
        pack['controlled_graph']=np.stack(graphs)
        pool[row['channels']]=pack
    return pool,manifest


def batches(pool, scalar_batch, data_seed):
    out=[]
    for c,pack in sorted(pool.items()):
        rng=np.random.RandomState(data_seed+c);order=rng.permutation(len(pack['x']))
        size=max(1,scalar_batch//(c*pack['x'].shape[1]))
        out += [(c,order[i:i+size]) for i in range(0,len(order),size)]
    rng=np.random.RandomState(data_seed);rng.shuffle(out)
    return out


def tensors(pack, indices, device):
    import torch
    return {k:torch.from_numpy(pack[k][indices]).to(device) for k in ('x','normal','labels','controlled_graph')}


def evaluate_source(model,pool,config,device):
    import torch
    from sklearn.metrics import average_precision_score
    model.eval();labels=[];scores=[];normal_scores=[];density_scores=[];normal_density=[];roots=[]
    density_sum=0.;density_count=0;scalars=0
    with torch.no_grad():
        for c,ind in batches(pool,config['batch_input_scalars'],9002):
            pack=pool[c];b=tensors(pack,ind,device);a=b['controlled_graph']
            obs=model(b['x'],a);norm=model(b['normal'],a)
            p=torch.sigmoid(obs['logit']).cpu().numpy();pn=torch.sigmoid(norm['logit']).cpu().numpy()
            d=obs['density_score'].cpu().numpy();dn=norm['density_score'].cpu().numpy()
            labels.extend([np.zeros_like(pack['labels'][ind]).ravel(),pack['labels'][ind].ravel()])
            scores.extend([pn.ravel(),p.ravel()]);density_scores.extend([dn.ravel(),d.ravel()])
            normal_scores.append(pn.ravel());normal_density.append(dn.ravel())
            density_sum+=float(norm['nll'].sum());density_count+=norm['nll'].numel()
            scalars+=b['x'].numel()+b['normal'].numel()
            innovation=obs['innovation'].cpu().numpy()
            for local,idx in enumerate(ind):
                mask=pack['root_window'][idx].astype(bool)
                if c>1 and mask.any():
                    rank=np.argsort(-innovation[local,mask].mean(0),kind='stable');r=int(pack['root'][idx])
                    position=int(np.flatnonzero(rank==r)[0])+1
                    roots.append((position,c))
    y=np.concatenate(labels);s=np.concatenate(scores);d=np.concatenate(density_scores)
    threshold=float(np.quantile(np.concatenate(normal_scores),1-config['normal_FPR_calibration']))
    density_threshold=float(np.quantile(np.concatenate(normal_density),1-config['normal_FPR_calibration']))
    return dict(normal_NLL=density_sum/density_count,AP=float(average_precision_score(y,s)),
        density_AP=float(average_precision_score(y,d)),threshold=threshold,density_threshold=density_threshold,
        normal_FPR=float(np.mean(np.concatenate(normal_scores)>threshold)),
        RCA=dict(systems=len(roots),Top1=float(np.mean([r==1 for r,c in roots])) if roots else None,
            Top3=float(np.mean([r<=min(3,c) for r,c in roots])) if roots else None,
            MRR=float(np.mean([1/r for r,c in roots])) if roots else None,conditional_on_true_root_window=True),
        frozen_forward_scalars=scalars)


def train(a):
    import torch
    from torch.nn import functional as F
    from baselines.shared_conditional import SharedConditional
    config=json.loads(a.config.read_text());torch.set_num_threads(1)
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(config['memory_gib_per_worker']*2**30/torch.cuda.get_device_properties(0).total_memory)
    device=torch.device('cuda:0');assert not a.output.exists(),'Never overwrite a run'
    a.output.mkdir(parents=True);started=time.monotonic()
    train_pool,manifest=load_pool(a.source,'train',a.mode)
    validation,_=load_pool(a.source,'source_validation',a.mode)
    model=SharedConditional(config['history'],config['width'],config['message_width']).to(device)
    initial_sha=state_sha(model);parameter_count=sum(p.numel() for p in model.parameters())
    source_sha=sha256(a.source/'manifest.json')
    positives=sum(int(p['labels'].sum()) for p in train_pool.values())
    total_points=sum(p['labels'].size for p in train_pool.values())
    pos_weight=min(100.,max(1.,(2*total_points-positives)/max(1,positives)))
    order=batches(train_pool,config['batch_input_scalars'],config['data_seed'])
    status=dict(status='running',started_at=now(),mode=a.mode,seed=a.seed,pid=os.getpid(),
        parameters=parameter_count,initial_parameter_sha256=initial_sha,source_manifest_sha256=source_sha,
        config_sha256=sha256(a.config),code_sha256=sha256(Path(__file__)),
        local_commit=a.local_commit,github_commit=a.github_commit,updates=0,total_updates=len(order),
        optimized_original_observed_scalars=0,normal_twin_training_forward_scalars=0,
        density_loss_scalars=0,detection_loss_points=0,extra_source_evaluation_forward_scalars=0,
        positive_weight=pos_weight,pretrained_weights=None,benchmark_arrays_in_training=False)
    atomic_json(a.output/'progress.json',status)
    initial=evaluate_source(model,validation,config,device)
    atomic_json(a.output/'initial_source_validation.json',initial)
    status['extra_source_evaluation_forward_scalars']+=initial['frozen_forward_scalars']
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    curve=a.output/'learning_curve.jsonl'
    for update,(c,ind) in enumerate(order,1):
        if time.monotonic()-started>config['max_hours_per_run']*3600: raise TimeoutError('Fixed technical deadline reached; budget not silently extended')
        model.train();b=tensors(train_pool[c],ind,device);graph=b['controlled_graph'];optimizer.zero_grad(set_to_none=True)
        normal=model(b['normal'],graph);observed=model(b['x'],graph)
        density=normal['nll'].mean()
        weight=torch.tensor(pos_weight,device=device)
        ce=.5*(F.binary_cross_entropy_with_logits(observed['logit'],b['labels'].float(),pos_weight=weight)+
                F.binary_cross_entropy_with_logits(normal['logit'],torch.zeros_like(normal['logit']),pos_weight=weight))
        loss=config['density_loss_weight']*density+config['detection_loss_weight']*ce
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'])
        if not torch.isfinite(norm): raise FloatingPointError('Non-finite gradient')
        optimizer.step()
        status['optimized_original_observed_scalars']+=b['x'].numel()
        status['normal_twin_training_forward_scalars']+=b['normal'].numel()
        status['density_loss_scalars']+=normal['nll'].numel()
        status['detection_loss_points']+=2*b['labels'].numel()
        status.update(updates=update,at=now(),elapsed_seconds=time.monotonic()-started,
            last_loss=float(loss),last_normal_NLL=float(density),last_detection_BCE=float(ce))
        with curve.open('a') as f:f.write(json.dumps(dict(update=update,channels=c,loss=float(loss),normal_NLL=float(density),
            detection_BCE=float(ce),original_scalars=status['optimized_original_observed_scalars'],seconds=status['elapsed_seconds']))+'\n')
        if update==1 or update%20==0 or update==len(order):atomic_json(a.output/'progress.json',status)
    assert status['optimized_original_observed_scalars']==manifest['training_original_observed_scalars']
    assert status['optimized_original_observed_scalars']<=config['original_observed_train_budget_per_run']
    final=evaluate_source(model,validation,config,device);atomic_json(a.output/'final_source_validation.json',final)
    status['extra_source_evaluation_forward_scalars']+=final['frozen_forward_scalars']
    final_sha=state_sha(model)
    ckpt=a.output/'final.pth';tmp=ckpt.with_suffix('.partial')
    torch.save(dict(model=model.state_dict(),config=config,mode=a.mode,seed=a.seed,
        source_manifest_sha256=source_sha,initial_parameter_sha256=initial_sha,final_parameter_sha256=final_sha,
        source_calibration=final,local_commit=a.local_commit,github_commit=a.github_commit),tmp)
    os.replace(tmp,ckpt)
    status.update(status='training_complete',completed_at=now(),checkpoint_sha256=sha256(ckpt),
        final_parameter_sha256=final_sha,actual_peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        final_source_AP=final['AP'],final_source_density_AP=final['density_AP'],
        all_configured_updates_completed=True,checkpoint_rule=config['checkpoint_rule'])
    atomic_json(a.output/'progress.json',status)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--mode',required=True,choices=['none','parents','full','rewired'])
    p.add_argument('--seed',type=int,required=True);p.add_argument('--local-commit',required=True);p.add_argument('--github-commit',required=True)
    a=p.parse_args()
    try:print(json.dumps(train(a)))
    except Exception:
        a.output.mkdir(parents=True,exist_ok=True)
        atomic_json(a.output/'error.json',dict(error=traceback.format_exc(),at=now()));raise
