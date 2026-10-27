"""64 new SCM systems; normal twin is a source-training response only."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from proposals.matched_pairs import generate
from tools.train_meta_basis import nuisance
from causal_tsad.io import atomic_json,sha256,array_sha

def prepare(root):
    root.mkdir(parents=True,exist_ok=False);rows=[]
    for C in [8,12,16,20]:
        for i in range(16):
            seed=348009000000+C*10000+i;family=['linear','tanh','sin'][i%3]
            fork=[256,384,512,640][i%4]
            item=generate(C,seed,family,fork=fork);x=item['cases'];y=item['labels']
            values=np.concatenate([nuisance(x,seed,bg) for bg in ['plain','smooth']]);labels=np.concatenate([y,y])
            target=values[[0,1,0,1,4,5,4,5]].copy()
            for bg in [0,4]:
                for c in range(C):assert sorted(array_sha(values[bg+j,:,c]) for j in [0,1])==sorted(array_sha(values[bg+j,:,c]) for j in [2,3])
            p=root/f'{seed}.npz';np.savez_compressed(p,values=values,normal_response=target,labels=labels,graph=item['graph'])
            audit={k:v for k,v in item.items() if k not in ['cases','labels','graph']};audit['fork']=fork
            atomic_json(root/f'{seed}_audit.json',audit)
            rows.append(dict(key=str(seed),C=C,family=family,root=item['root'],fork=fork,path=p.name,SHA256=sha256(p),
                input_SHA256=array_sha(values),normal_response_SHA256=array_sha(target),graph_SHA256=array_sha(item['graph']),views=8))
    atomic_json(root/'manifest.json',dict(rows=rows,systems=64,unique_observed_scalars=7340032,normal_target_scalars=7340032,
        namespace='348009000000+C*10000+i',labels_detail=dict(effect=None,anomaly_channel=None,anomaly_time=None),
        benchmark_arrays_opened=False,train_only=True,normal_response='loss only; same shared prefix contamination retained; queries start320',
        familiar_families=True,new_systems=True,training_onset=[256,384,512,640]))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);prepare(p.parse_args().output)
