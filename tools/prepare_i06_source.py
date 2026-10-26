"""New independent SCM systems. This program never opens benchmark arrays."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from proposals.matched_pairs import generate
from tools.train_meta_basis import nuisance
from causal_tsad.io import atomic_json,sha256,array_sha

def prepare(root):
    root.mkdir(parents=True,exist_ok=False);rows=[]
    for split,n,offset in [('source_validation',8,1000000),('calibration',4,2000000)]:
        for C in [8,12,16,20]:
            for i in range(n):
                key=348006000000+offset+C*10000+i
                family=['linear','tanh','sin'][i%3]
                item=generate(C,key,family);cases=item['cases'];labels=item['labels'];rootnode=item['root']
                if split=='source_validation':
                    extra=[];ey=[]
                    for length in [1,128]:
                        for branch in range(2):
                            v=cases[branch].copy();y=labels[branch].copy()
                            ref=v[:256,rootnode];scale=max(1e-3,float(np.median(np.abs(ref-np.median(ref)))*1.4826))
                            v[512:512+length,rootnode]+=(1 if branch==0 else -1)*3*scale
                            y[512:512+length]=1;extra.append(v);ey.append(y)
                    cases=np.concatenate([cases,np.stack(extra)]);labels=np.concatenate([labels,np.stack(ey)])
                else:cases=cases[:2];labels=labels[:2]
                values=np.concatenate([nuisance(cases,key,bg) for bg in ['plain','smooth']])
                y=np.concatenate([labels,labels]);nv=len(cases)
                if split=='source_validation':
                    for b in [0,nv]:
                        for ch in range(C):
                            assert sorted(array_sha(values[b+j,:,ch]) for j in [0,1])==sorted(array_sha(values[b+j,:,ch]) for j in [2,3])
                p=root/f'{key}.npz';np.savez_compressed(p,values=values,labels=y,graph=item['graph'])
                metadata={k:v for k,v in item.items() if k not in ['cases','labels','graph']}
                atomic_json(root/f'{key}_audit.json',metadata)
                rows.append(dict(key=str(key),split=split,C=C,family=family,root=rootnode,path=p.name,SHA256=sha256(p),
                    input_SHA256=array_sha(values),graph_SHA256=array_sha(item['graph']),labels_SHA256=array_sha(y),views=len(values),
                    prefix_contaminated=item['prefix_contaminated'],audit_file=f'{key}_audit.json'))
    atomic_json(root/'manifest.json',dict(rows=rows,systems=48,unique_observed_scalars=sum(r['views']*1024*r['C'] for r in rows),
        training_exposure=0,benchmark_arrays_opened=False,labels_detail=dict(effect=None,anomaly_channel=None,anomaly_time=None),
        source_namespace='348006000000 + split(1M/2M) + C*10000 + i',families='linear/tanh/sin; familiar mechanism families, new systems',
        primary_groups=dict(relation=[0,1,2,3],short_measurement=[0,1,4,5],long_measurement=[0,1,6,7]),
        nuisance='plain and deterministic smooth; generator only uses baseline scale; no twin is passed to prediction'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);prepare(p.parse_args().output)
