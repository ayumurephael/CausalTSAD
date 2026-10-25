"""Immutable source/development feature cache. No optimizer exists here."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,hashlib,pickle,sys,time,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256,model_inputs
from proposals.c01.data import blocks,selected_rows


def module_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FrozenViews:
    def __init__(self,code,weight,config):
        import torch
        from proposals.c01.model import FrozenTimeRCDHistory
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


def make_tasks(source,benchmark):
    manifest=json.loads((source/'manifest.json').read_text());tasks=[]
    for row in manifest['files']:
        assert sha256(source/row['path'])==row['SHA256']
        for i in range(row['systems']):
            tasks.append(dict(kind=row['split'],pack=row['path'],offset=i,channels=row['channels'],
                key=f"source/{row['path']}/{i}",scalars=1024*row['channels']))
    for row in selected_rows(benchmark):
        tasks.append(dict(kind='development',key=row['system_id'],row=row,
                          channels=row['channels'],scalars=row['length']*row['channels'] if 'length' in row else None))
    for task in tasks:task['name']=hashlib.sha256(task['key'].encode()).hexdigest()[:24]
    return tasks


def worker(a):
    import torch
    config=json.loads(a.config.read_text());torch.set_num_threads(1)
    torch.manual_seed(348);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
    tasks=json.loads(a.tasks.read_text())[a.worker::a.workers]
    views=FrozenViews(a.original_code,a.checkpoint,config)
    before=dict(history=module_sha(views.history),detector=module_sha(views.base))
    started=time.monotonic();current=None;pack=None;rows=[];totals={}
    for task in tasks:
        if time.monotonic()-started>config['max_stage_hours']*3600:raise TimeoutError('Cache stage fixed deadline')
        name=task['name'];path=a.output/'arrays'/(name+'.npz');record=a.output/'records'/(name+'.json')
        assert not path.exists() and not record.exists(),'Immutable cache: use a new revision after a failure'
        if task['kind']=='development':
            row=task['row'];assert row['split']=='development'
            filename=str(a.benchmark/'data'/row['shard'])
            if current!=filename:
                with open(filename,'rb') as f:pack=pickle.load(f)
                current=filename
            sample=pack[row['offset']];assert sample['system_id']==row['system_id']
            x=model_inputs(sample,'anomalous');normal=model_inputs(sample,'normal')
            values=x['values'];normal_values=normal['values'];graph=x['adjacency'];labels=np.asarray(sample['labels'],np.int8)
        else:
            filename=str(a.source/task['pack'])
            if current!=filename:
                with np.load(filename) as f:pack={k:np.array(f[k]) for k in f.files}
                current=filename
            i=task['offset'];values=pack['x'][i];normal_values=pack['normal'][i];labels=pack['labels'][i];graph=pack['adjacency'][i]
        assert len(labels)==len(values) and np.isfinite(values).all() and np.isfinite(normal_values).all()
        saved={};counts={}
        for view,data in [('x',values),('normal',normal_values)]:
            encoded,qvalid,count=views.encode(data,config)
            saved.update({view+'_'+k:v for k,v in encoded.items()});counts[view]=count
        y=np.pad(labels,(0,qvalid.size-len(labels))).reshape(qvalid.shape)
        saved.update(graph=graph,labels=y,query_valid=qvalid)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.partial')
        with temporary.open('xb') as f:np.savez_compressed(f,**saved)
        os.replace(temporary,path)
        meta=dict(task,scalars=int(values.size),length=len(values),path=str(path.relative_to(a.output)),
                  SHA256=sha256(path),counts=counts,optimizer_updates=0)
        atomic_json(record,meta);rows.append(meta)
        for view,count in counts.items():
            for k,v in count.items():totals[view+'_'+k]=totals.get(view+'_'+k,0)+v
        atomic_json(a.output/f'worker_{a.worker}.json',dict(status='running',completed=len(rows),total=len(tasks),counts=totals))
    after=dict(history=module_sha(views.history),detector=module_sha(views.base));assert before==after
    result=dict(status='complete',worker=a.worker,completed=len(rows),total=len(tasks),counts=totals,
                frozen_parameter_digests_before=before,frozen_parameter_digests_after=after,
                actual_history_forwards=views.history.forward_accounting,
                actual_base_forwards=views.base_forward_accounting,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,
                checkpoint_sha256=sha256(a.checkpoint),config_sha256=sha256(a.config),
                target_validation_or_locked_arrays_opened=False,optimizer_updates=0)
    atomic_json(a.output/f'worker_{a.worker}.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--original-code',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',type=Path,required=True);p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,required=True)
    a=p.parse_args()
    try:worker(a)
    except Exception:
        atomic_json(a.output/f'worker_{a.worker}_error.json',dict(error=traceback.format_exc()));raise
