"""Exact original C01 encoding/window arithmetic, repackaged without labels."""
import numpy as np
def blocks(values,history=64,horizon=16):
    t,c=values.shape;q=(t+horizon-1)//horizon
    past=np.zeros((q,c,history),np.float32);valid=np.zeros_like(past,bool)
    query=np.zeros((q,horizon,c),np.float32);query_valid=np.zeros((q,horizon),bool)
    for k,start in enumerate(range(0,t,horizon)):
        begin=max(0,start-history);length=start-begin
        if length:past[k,:,-length:]=values[begin:start].T;valid[k,:,-length:]=True
        end=min(t,start+horizon);query[k,:end-start]=values[start:end];query_valid[k,:end-start]=True
    count=valid.sum(-1,keepdims=True).clip(1)
    center=(past*valid).sum(-1,keepdims=True)/count
    scale=np.sqrt(((past-center)**2*valid).sum(-1,keepdims=True)/count).clip(1e-3)
    scale=np.where(valid.any(-1,keepdims=True),scale,1.)
    raw=(past-center)/scale*valid
    normalized=(query-center.transpose(0,2,1))/scale.transpose(0,2,1)
    assert np.isfinite(normalized).all()
    return past,valid,raw.astype(np.float32),normalized.astype(np.float32),query_valid



class FrozenViews:
    def __init__(self,code,weight,config):
        import torch
        from dependencies.c01_model import FrozenTimeRCDHistory
        self.history=FrozenTimeRCDHistory(code,weight,config['checkpoint_sha256'],microbatch=128).cuda()
        from training_runtime import RuntimeModel,make_config
        checkpoint=torch.load(weight,map_location='cpu',weights_only=False)
        saved=checkpoint.get('config',{}).get('ts_config',{})
        cfg=make_config(int(saved.get('num_features',20)))
        for k,v in saved.items():
            if hasattr(cfg.ts_config,k):setattr(cfg.ts_config,k,v)
        self.base=RuntimeModel(cfg,activation_checkpointing=False).cuda().eval().requires_grad_(False)
        self.base.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.base_forward_accounting=dict(calls=0,padded_scalar_slots=0)

    def base_logits(self,values):
        import torch
        # Identical normalization/window/tail/ChannelMean rule to zero_shot.
        x=values;mean=np.mean(x,axis=0);std=np.std(x,axis=0);std=np.where(std==0,1e-8,std)
        x=(x-mean)/std;t,c=x.shape;win=min(t,5000);out=[];n=0
        self.base.ts_encoder.num_features=c
        with torch.no_grad():
            for start in range(0,t,win):
                part=x[start:start+win];length=len(part)
                if length<win:part=np.concatenate([part,np.repeat(part[-1:],win-length,axis=0)])
                valid=torch.arange(win,device='cuda')[None]<length
                data=torch.as_tensor(part,dtype=torch.float32,device='cuda')[None]
                logits=self.base.anomaly_head(self.base.ts_encoder(data,valid)).mean(-2).float()
                out.append((logits[...,1]-logits[...,0])[0,:length].cpu().numpy());n+=data.numel()
        self.base_forward_accounting['calls']+=1;self.base_forward_accounting['padded_scalar_slots']+=n
        return np.concatenate(out),n

    def encode(self,values,config):
        import torch
        h,p=config['history'],config['horizon']
        past,valid,raw,query,qvalid=blocks(values,h,p)
        with torch.no_grad():
            features=self.history(torch.from_numpy(past)[None].cuda(),torch.from_numpy(valid)[None].cuda())[0].cpu().numpy()
        logit,base_input=self.base_logits(values)
        padded=np.pad(logit,(0,query.shape[0]*p-len(logit))).reshape(query.shape[:2])
        return dict(features=features,raw=raw,query=query,base=padded,
                    fraction=valid.mean(-1).astype(np.float32)),qvalid,dict(
            unique_observed_scalars=int(values.size),history_valid_scalars=int(valid.sum()),
            history_padded_slots=int(valid.size),history_encoded_channels=int(valid.any(-1).sum()),
            encoder_patch_tokens=int(valid.any(-1).sum())*((h+15)//16),
            detector_padded_scalars=base_input)


