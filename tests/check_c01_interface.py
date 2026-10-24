"""Interface-only GPU checks; no dataset, optimizer, or benchmark is used."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from proposals.c01.model import FrozenTimeRCDHistory,ParentInnovationHead
from causal_tsad.io import atomic_json


def digest(module):
    h=hashlib.sha256()
    for n,v in sorted(module.state_dict().items()):h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main(a):
    torch.set_num_threads(1);torch.manual_seed(348)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    encoder=FrozenTimeRCDHistory(a.original_code,a.checkpoint,a.weight_sha,microbatch=8).cuda()
    head=ParentInnovationHead(feature_dim=encoder.feature_dim).cuda()
    before=digest(encoder)
    past=torch.randn(1,3,4,64,device='cuda');valid=torch.ones_like(past,dtype=torch.bool)
    valid[:,0]=False;valid[:,1,:,:32]=False
    f=encoder(past,valid)
    z=past.clone();z[~valid]=1000
    assert torch.allclose(f,encoder(z,valid),atol=1e-6,rtol=0)
    z=past.clone();z[:,:,3]+=20
    assert torch.allclose(f[:,:,:3],encoder(z,valid)[:,:,:3],atol=1e-5,rtol=0)
    graph=torch.zeros(1,4,4,device='cuda');graph[:,0,1]=1;graph[:,1,2]=1
    fraction=valid.float().mean(-1)
    mu,lv=head(f,fraction,graph)
    other=f.clone();other[:,:,3]+=10
    assert torch.allclose(mu[:,:,1],head(other,fraction,graph)[0][:,:,1],atol=1e-6,rtol=0)
    p=torch.tensor([3,1,0,2],device='cuda')
    reordered=head(f[:,:,p],fraction[:,:,p],graph[:,p][:,:,p])[0]
    assert torch.allclose(mu[:,:,p],reordered,atol=1e-5,rtol=0)
    encoder.train();assert not encoder.training and not encoder.encoder.training
    assert not any(p.requires_grad for p in encoder.parameters())
    (mu.square().mean()+lv.square().mean()).backward()
    assert before==digest(encoder)
    assert all(torch.isfinite(v).all() for v in [f,mu,lv])
    report=dict(status='passed',scope='model interfaces only, not full C01 data/training/evaluation pipeline',
        no_optimizer=True,benchmark_arrays_opened=False,invalid_history_invariance=True,
        per_channel_frozen_encoding_independent=True,nonparent_head_invariance=True,head_equivariance=True,
        frozen_encoder_hash_unchanged=True,encoder_parameters=sum(p.numel() for p in encoder.parameters()),
        head_parameters=sum(p.numel() for p in head.parameters()),weight_sha256=a.weight_sha,
        technical_random_history_scalars=past.numel(),peak_cuda_bytes=torch.cuda.max_memory_allocated())
    atomic_json(a.output,report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original-code',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--weight-sha',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
