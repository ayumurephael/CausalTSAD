"""C01 model interface. Only legal history reaches the frozen encoder.

This initial revision defines the model, not a completed training experiment.
The B1 pilot is independent; no B1-specific implementation is inherited here.
"""
from pathlib import Path
import sys
import torch
from torch import nn
from causal_tsad.io import sha256


class FrozenTimeRCDHistory(nn.Module):
    def __init__(self, original_code, checkpoint, expected_sha256, microbatch=128):
        super().__init__()
        assert sha256(checkpoint)==expected_sha256
        sys.path.insert(0,str(Path(original_code)))
        from training_runtime import RuntimeModel,make_config
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        # A multi checkpoint owns same/different-channel bias parameters even
        # when this interface supplies one channel per independent batch row.
        # Preserve the saved architecture, rather than dropping those weights.
        metadata=saved.get('config',{}).get('ts_config',{})
        config=make_config(int(metadata.get('num_features',20)))
        for key,value in metadata.items():
            if hasattr(config.ts_config,key):setattr(config.ts_config,key,value)
        original=RuntimeModel(config,activation_checkpointing=False)
        original.load_state_dict(saved['model_state_dict'],strict=True)
        self.encoder=original.ts_encoder.eval().requires_grad_(False)
        self.feature_dim=self.encoder.d_proj
        self.microbatch=microbatch;self.checkpoint_sha256=expected_sha256

    def train(self,mode=True):
        # Parent .train() calls must not enable dropout in the frozen encoder.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self,past_values,past_valid):
        """Inputs [...,H] contain only times < query; no current value is accepted.

        Each channel is encoded in isolation. A global cross-channel encoding
        cannot be masked afterwards and called a parent-only mechanism.
        """
        assert past_values.shape==past_valid.shape
        shape=past_values.shape[:-1];h=past_values.shape[-1]
        x=past_values.reshape(-1,h);valid=past_valid.reshape(-1,h).bool()
        count=valid.sum(-1,keepdim=True).clamp_min(1)
        center=(x*valid).sum(-1,keepdim=True)/count
        variance=((x-center).square()*valid).sum(-1,keepdim=True)/count
        scale=variance.sqrt().clamp_min(1e-3)
        z=(x-center)/scale*valid
        output=x.new_zeros(len(x),self.feature_dim)
        indices=torch.where(valid.any(-1))[0]
        for part in indices.split(self.microbatch):
            v=z[part,:,None];mask=valid[part]
            embeddings=self.encoder(v,mask)[:,:,0]
            last=(mask.long()*torch.arange(1,h+1,device=x.device)).argmax(-1)
            output[part]=embeddings[torch.arange(len(part),device=x.device),last]
        return output.reshape(*shape,self.feature_dim)


class ParentInnovationHead(nn.Module):
    def __init__(self,feature_dim=256,width=64,message_width=32):
        super().__init__()
        self.own=nn.Sequential(nn.Linear(feature_dim+1,width),nn.GELU())
        self.message=nn.Sequential(nn.Linear(2*feature_dim,message_width),nn.GELU(),
                                   nn.Linear(message_width,message_width),nn.GELU())
        self.distribution=nn.Sequential(nn.Linear(width+message_width+1,width),nn.GELU(),nn.Linear(width,2))

    def forward(self,past_features,past_valid_fraction,adjacency):
        """Return normalized conditional Gaussian parameters before seeing query."""
        b,q,c,d=past_features.shape
        own=self.own(torch.cat([past_features,past_valid_fraction[...,None]],-1))
        target=past_features[:,:,:,None,:].expand(-1,-1,-1,c,-1)
        parent=past_features[:,:,None,:,:].expand(-1,-1,c,-1,-1)
        messages=self.message(torch.cat([target,parent],-1))
        mask=adjacency.transpose(1,2)[:,None,:,:,None]
        degree=mask.sum(-2)
        pooled=(messages*mask).sum(-2)/degree.clamp_min(1)
        stats=self.distribution(torch.cat([own,pooled,torch.log1p(degree).expand(b,q,c,1)],-1))
        mean,logvar=stats.unbind(-1)
        return mean,logvar.clamp(-6,6)
