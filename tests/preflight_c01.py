"""G0 checks on random/source arrays, before target cache or optimizer launch."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256
from proposals.c01.data import blocks,selected_rows
from tools.cache_c01 import FrozenViews,module_sha


def main(a):
    import torch
    from proposals.c01.model import CausalResidualDetector
    torch.set_num_threads(1);torch.manual_seed(348)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
    config=json.loads(a.config.read_text());manifest=json.loads((a.source/'manifest.json').read_text())
    assert sha256(a.source/'manifest.json')==config['source_manifest_sha256']
    chosen=selected_rows(a.benchmark)
    assert all((a.benchmark/'data'/r['shard']).exists() for r in chosen)
    views=FrozenViews(a.original_code,a.checkpoint,config);before=module_sha(views.history)
    h,p=config['history'],config['horizon'];d=views.history.feature_dim
    # Each prediction of the entire next block uses only positions < block start.
    x=np.random.RandomState(348).normal(size=(128,4)).astype(np.float32)
    past,valid,raw,query,qvalid=blocks(x,h,p)
    modified=x.copy();modified[64:]+=123
    p2,v2,_,_,_=blocks(modified,h,p)
    assert np.array_equal(past[:5],p2[:5]) and np.array_equal(valid[:5],v2[:5])
    f=views.history(torch.from_numpy(past)[None].cuda(),torch.from_numpy(valid)[None].cuda())
    z=torch.from_numpy(past)[None].cuda();mask=torch.from_numpy(valid)[None].cuda()
    changed=z.clone();changed[:,:,3]+=13
    assert torch.allclose(f[:,:,:3],views.history(changed,mask)[:,:,:3],atol=1e-5,rtol=0)
    changed=z.clone();changed[~mask]=2345
    assert torch.allclose(f,views.history(changed,mask),atol=1e-5,rtol=0)
    fraction=mask.float().mean(-1);graph=torch.zeros(1,4,4,device='cuda');graph[:,0,1]=1;graph[:,1,2]=1
    model=CausalResidualDetector(d,horizon=p).cuda()
    mu,lv=model.density(f,fraction,graph);other=f.clone();other[:,:,3]+=32
    assert torch.allclose(mu[:,:,:,1],model.density(other,fraction,graph)[0][:,:,:,1],atol=1e-6,rtol=0)
    perm=torch.tensor([2,0,3,1],device='cuda')
    assert torch.allclose(mu[:,:,:,perm],model.density(f[:,:,perm],fraction[:,:,perm],graph[:,perm][:,:,perm])[0],atol=1e-5,rtol=0)
    q=torch.from_numpy(query)[None].cuda();base=torch.randn(1,len(past),p,device='cuda')
    out=model(f,fraction,graph,q,base)
    assert torch.equal(out['logit'],base),'Zero correction must preserve original logits exactly'
    out['logit'].sum().backward()
    assert all(param.grad is None for param in model.density.parameters()),'CE must not train normal mechanisms'
    model.zero_grad(set_to_none=True);model(f,fraction,graph,q,base)['nll'].mean().backward()
    assert any(param.grad is not None for param in model.density.parameters())
    assert before==module_sha(views.history)
    # Compare the frozen branch with the original dataset+model implementation.
    sys.path.insert(0,a.original_code)
    from models.time_rcd.TimeRCD_pretrain_multi import TimeSeriesPretrainModel
    from training_runtime import make_config
    from utils.dataset import TimeRCDDataset
    cfg=make_config(20);reference=TimeSeriesPretrainModel(cfg).cuda().eval()
    reference.load_state_dict(views.base.state_dict(),strict=True)
    checks=[];source_probe_scalars=0;reference_forward_slots=0
    for c in [1,8,20]:
        row=next(r for r in manifest['files'] if r['split']=='train' and r['channels']==c)
        assert sha256(a.source/row['path'])==row['SHA256']
        with np.load(a.source/row['path']) as pack:values=np.array(pack['x'][0])
        # Also test tail padding in the original offline detector.
        if c==8:values=np.concatenate([values]*5)[:5017]
        dataset=TimeRCDDataset(values,window_size=min(len(values),5000),stride=min(len(values),5000),normalize=True)
        reference.ts_encoder.num_features=c;prob=[]
        with torch.no_grad():
            for i in range(len(dataset)):
                xx,mm=dataset[i];embedding=reference(time_series=xx[None].cuda(),mask=mm[None].cuda())
                logits=reference.anomaly_head(embedding).mean(-2)
                prob.append(torch.softmax(logits.float(),-1)[0,:,1].cpu().numpy())
                reference_forward_slots+=xx.numel()
        expected=np.concatenate(prob)[:len(values)]
        actual,_=views.base_logits(values);actual=torch.sigmoid(torch.from_numpy(actual)).numpy()
        error=float(np.max(np.abs(expected-actual)));assert error<1e-4
        checks.append(dict(channels=c,length=len(values),base_probability_max_error=error));source_probe_scalars+=values.size
        encoded,vld,_=views.encode(values[:128],config)
        feature=torch.from_numpy(encoded['features'])[None].cuda();fraction=torch.from_numpy(encoded['fraction'])[None].cuda()
        query=torch.from_numpy(encoded['query'])[None].cuda();base=torch.from_numpy(encoded['base'])[None].cuda()
        empty=torch.zeros(1,c,c,device='cuda');value=model(feature,fraction,empty,query,base)
        assert all(torch.isfinite(t).all() for t in value.values())
    atomic_json(a.output,dict(status='passed',source_only=True,optimizer_updates=0,
        target_arrays_opened=False,query_block_blind=True,nonparent_history_invariance=True,
        invalid_invariance=True,head_channel_equivariance=True,CE_density_gradient_isolation=True,
        zero_correction_logit_error=0,encoder_hash_unchanged=True,base_equivalence=checks,
        trainable_parameters=sum(param.numel() for param in model.parameters()),feature_dim=d,
        random_history_array_scalars=int(past.size),source_probe_unique_scalars=source_probe_scalars,
        actual_history_forwards=views.history.forward_accounting,
        actual_runtime_base_forwards=views.base_forward_accounting,
        additional_original_reference_forward_scalar_slots=reference_forward_slots,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        scope='technical G0; cannot prove detection or RCA improvement'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--original-code',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
