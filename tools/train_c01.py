"""Fixed one-pass C01 pilot. Optimizers can only access independent source rows."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,sys,time,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256
from proposals.c01.data import batches,tensor_batch
from tools.cache_c01 import module_sha


def forward(model,b,view):
    return model(b[view+'_features'].float(),b[view+'_fraction'].float(),b['graph'].float(),
                 b[view+'_query'].float(),b[view+'_base'].float())


def source_evaluation(model,root,rows,arm,config):
    import torch
    from sklearn.metrics import average_precision_score
    labels=[];scores=[];densities=[];normal_scores=[];base_scores=[];normal_base=[];normal_density=[]
    total_nll=0.;count=0;slots=0
    model.eval()
    with torch.no_grad():
        for batch in batches(rows,config['batch_input_scalars'],9002):
            assert all(r['kind']=='source_validation' for r in batch)
            b=tensor_batch(root,batch,arm,'cuda');valid=b['query_valid'].bool()
            for view in ['x','normal']:
                out=forward(model,b,view);s=torch.sigmoid(out['logit'])[valid].cpu().numpy()
                base=torch.sigmoid(b[view+'_base'])[valid].cpu().numpy()
                d=out['density_score'][valid].cpu().numpy()
                y=b['labels'][valid].cpu().numpy() if view=='x' else np.zeros(len(s),np.int8)
                labels.append(y);scores.append(s);base_scores.append(base);densities.append(d)
                slots+=int(valid.sum())*b[view+'_query'].shape[-1]
                if view=='normal':
                    normal_scores.append(s);normal_base.append(base);normal_density.append(d)
                    nll=out['nll'][valid];total_nll+=float(nll.sum());count+=nll.numel()
    labels=np.concatenate(labels)
    result=dict(AP=float(average_precision_score(labels,np.concatenate(scores))),
        base_AP=float(average_precision_score(labels,np.concatenate(base_scores))),
        density_AP=float(average_precision_score(labels,np.concatenate(densities))),normal_NLL=total_nll/count,
        threshold=float(np.quantile(np.concatenate(normal_scores),1-config['normal_FPR_calibration'])),
        base_threshold=float(np.quantile(np.concatenate(normal_base),1-config['normal_FPR_calibration'])),
        density_threshold=float(np.quantile(np.concatenate(normal_density),1-config['normal_FPR_calibration'])),
        head_query_scalar_forward_slots=slots,encoder_forward_slots=0,source_only=True)
    assert all(np.isfinite(v) for v in result.values() if isinstance(v,float))
    return result


def train(a):
    import torch
    from torch.nn import functional as F
    from proposals.c01.model import CausalResidualDetector
    config=json.loads(a.config.read_text());torch.set_num_threads(1)
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(config['memory_gib_per_worker']*2**30/torch.cuda.get_device_properties(0).total_memory)
    assert not a.output.exists(),'Never overwrite a run';a.output.mkdir(parents=True)
    manifest=json.loads((a.cache/'manifest.json').read_text());assert manifest['status']=='complete'
    assert manifest['source_manifest_sha256']==config['source_manifest_sha256']
    training=[r for r in manifest['files'] if r['kind']=='train']
    validation=[r for r in manifest['files'] if r['kind']=='source_validation']
    # Deliberately do not open development files from this optimizer process.
    for r in training+validation:assert sha256(a.cache/r['path'])==r['SHA256']
    model=CausalResidualDetector(feature_dim=manifest['feature_dim'],width=config['width'],
        message_width=config['message_width'],horizon=config['horizon']).cuda()
    initial=module_sha(model);order=batches(training,config['batch_input_scalars'],config['data_seed'])
    expected=sum(r['scalars'] for r in training);assert expected<=config['original_observed_train_budget_per_run']
    d,w,m,h=manifest['feature_dim'],config['width'],config['message_width'],config['horizon']
    # Every C^2 pair is evaluated before masking; counts are equal in graph arms.
    linear_flops=sum(2*((r['length']+h-1)//h)*(r['channels']*((d+1)*w+(w+m+1)*w+w*2*h)+
        r['channels']**2*(2*d*m+m*m)+h*(5*16+16)) for r in training)
    positives=0;points=0
    for r in training:
        with np.load(a.cache/r['path']) as z:positives+=int(z['labels'].sum());points+=int(z['query_valid'].sum())
    pos_weight=min(100.,max(1.,(2*points-positives)/max(1,positives)))
    started=time.monotonic();status=dict(status='running',pid=os.getpid(),arm=a.arm,seed=a.seed,
        local_commit=a.local_commit,github_commit=a.github_commit,source_manifest_sha256=config['source_manifest_sha256'],
        cache_manifest_sha256=sha256(a.cache/'manifest.json'),config_sha256=sha256(a.config),
        original_checkpoint_sha256=config['checkpoint_sha256'],initial_parameter_sha256=initial,
        trainable_parameters=sum(p.numel() for p in model.parameters()),updates=0,total_updates=len(order),
        optimized_original_observed_scalars=0,normal_twin_training_query_scalars=0,
        additional_encoder_forward_scalars=0,extra_source_evaluation_head_query_scalars=0,
        original_parameters_updated=False,target_arrays_in_gradient=False,positive_weight=pos_weight,
        two_view_training_head_Linear_forward_FLOPs=2*linear_flops,
        FLOPs_note='Dense Linear only, excludes frozen encoder/cache, activations, normalization, backward and evaluation; cache costs separately recorded. Raw control not native end-to-end FLOPs matched.',
        graph_mode=a.arm.split('_',1)[1],representation=a.arm.split('_',1)[0],
        scope='cached frozen features; encoder costs amortized and separately recorded in cache manifest')
    atomic_json(a.output/'progress.json',status)
    before=source_evaluation(model,a.cache,validation,a.arm,config)
    atomic_json(a.output/'initial_source_validation.json',before)
    status['extra_source_evaluation_head_query_scalars']+=before['head_query_scalar_forward_slots']
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    for i,rows in enumerate(order,1):
        if time.monotonic()-started>3600*config['max_train_hours_per_run']:raise TimeoutError('Fixed train deadline, no silent extension')
        assert all(r['kind']=='train' for r in rows)
        model.train();b=tensor_batch(a.cache,rows,a.arm,'cuda');valid=b['query_valid'].bool()
        optimizer.zero_grad(set_to_none=True);observed=forward(model,b,'x');normal=forward(model,b,'normal')
        density=normal['nll'][valid].mean()
        weight=torch.tensor(pos_weight,device='cuda')
        ce=.5*(F.binary_cross_entropy_with_logits(observed['logit'][valid],b['labels'][valid].float(),pos_weight=weight)+
            F.binary_cross_entropy_with_logits(normal['logit'][valid],torch.zeros_like(normal['logit'][valid]),pos_weight=weight))
        loss=config['density_loss_weight']*density+config['detection_loss_weight']*ce
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'])
        if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
        optimizer.step();scalars=sum(r['scalars'] for r in rows)
        status['optimized_original_observed_scalars']+=scalars;status['normal_twin_training_query_scalars']+=scalars
        status.update(updates=i,last_loss=float(loss),last_normal_NLL=float(density),last_detection_BCE=float(ce),seconds=time.monotonic()-started)
        with (a.output/'learning_curve.jsonl').open('a') as f:f.write(json.dumps({k:status[k] for k in ['updates','last_loss','last_normal_NLL','last_detection_BCE','optimized_original_observed_scalars','seconds']})+'\n')
        if i==1 or i%20==0 or i==len(order):atomic_json(a.output/'progress.json',status)
    assert status['optimized_original_observed_scalars']==expected
    final=source_evaluation(model,a.cache,validation,a.arm,config)
    atomic_json(a.output/'final_source_validation.json',final)
    status['extra_source_evaluation_head_query_scalars']+=final['head_query_scalar_forward_slots']
    checkpoint=dict(model=model.state_dict(),config=config,arm=a.arm,seed=a.seed,source_calibration=final,
        feature_dim=manifest['feature_dim'],cache_manifest_sha256=sha256(a.cache/'manifest.json'),
        local_commit=a.local_commit,github_commit=a.github_commit)
    temp=a.output/'final.partial';torch.save(checkpoint,temp);os.replace(temp,a.output/'final.pth')
    status.update(status='training_complete',final_parameter_sha256=module_sha(model),
        checkpoint_sha256=sha256(a.output/'final.pth'),final_source_AP=final['AP'],
        final_source_density_AP=final['density_AP'],peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        fixed_final_all_updates_complete=True)
    atomic_json(a.output/'progress.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--arm',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--local-commit',required=True);p.add_argument('--github-commit',required=True)
    a=p.parse_args()
    try:train(a)
    except Exception:
        atomic_json(a.output/'error.json',dict(error=traceback.format_exc()));raise
