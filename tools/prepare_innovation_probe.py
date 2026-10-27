"""Independent source systems with explicit injection-time labels; no benchmark reads."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from proposals.matched_pairs import generate,T
from causal_tsad.io import atomic_json,sha256,array_sha

DURATIONS=[1,4,16,64,256]
def build(job):
    c,i,out=job;seed=348004000000+c*10000+i;family=['linear','tanh','sin'][i%3]
    row=generate(c,seed,family);normal=row['cases'][:2];p=row['audit_parameters'];perm=np.asarray(p['new_to_old'])
    rho=np.asarray(p['rho'])[perm];b=np.asarray(p['coefficient_before_permutation'])[perm][:,perm]
    lag=np.asarray(p['lag_before_permutation'])[perm][:,perm];g=row['graph'];order=np.argsort(perm)
    f=(lambda x:x) if family=='linear' else np.tanh if family=='tanh' else np.sin
    rng=np.random.default_rng(seed+91);child=int(rng.choice(np.flatnonzero(g.sum(0)>0)));parent=int(rng.choice(np.flatnonzero(g[:,child])))
    root=row['root'];sign=float(rng.choice([-1,1]));scale=max(float(np.median(np.abs(normal[0,:256,root]-np.median(normal[0,:256,root])))*1.4826),.1)
    cases=[normal[0].copy(),normal[1].copy()];meta=[dict(kind='normal',duration=0,root=None,interval=None,base=v) for v in [0,1]]
    for base in range(2):
        n=normal[base]
        for duration in DURATIONS:
            for kind in ['measurement','physical','edge_coefficient','edge_lag']:
                x=n.copy();eventroot=root if kind in ['measurement','physical'] else child
                if kind=='measurement':x[512:512+duration,root]+=sign*3*scale
                else:
                    delta=np.zeros_like(n)
                    for t in range(512,T):
                        active=t<512+duration
                        for node in order:
                            delta[t,node]=rho[node]*delta[t-1,node]
                            for par in np.flatnonzero(g[:,node]):
                                tt=t-lag[par,node]
                                delta[t,node]+=b[par,node]*(f(x[tt,par])-f(n[tt,par]))
                            if active and kind=='physical' and node==root:delta[t,node]+=sign*3*scale
                            if active and node==child and kind=='edge_coefficient':delta[t,node]+=.25*b[parent,child]*f(x[t-lag[parent,child],parent])
                            if active and node==child and kind=='edge_lag':delta[t,node]+=b[parent,child]*(f(x[t-lag[parent,child]-8,parent])-f(x[t-lag[parent,child],parent]))
                            x[t,node]=n[t,node]+delta[t,node]
                cases.append(x);meta.append(dict(kind=kind,duration=duration,root=eventroot,interval=[512,512+duration],base=base))
    cases=np.stack(cases);labels=np.zeros((len(cases),T),np.int8)
    for v,m in enumerate(meta):
        if m['interval']:labels[v,slice(*m['interval'])]=1
    # Prefix sensor contamination from the original generator is retained and
    # disclosed. It is outside the >=320 evaluation point set, never called clean R.
    assert np.array_equal(cases[:,:256],np.repeat(cases[:1,:256],len(cases),0))
    assert np.isfinite(cases).all();key=str(seed);path=Path(out)/(key+'.npz')
    np.savez_compressed(path,cases=cases,labels=labels,graph=g)
    atomic_json(Path(out)/(key+'_audit_NOT_MODEL_INPUT.json'),dict(parameters=p,case_metadata=meta,
        prefix_contaminated=row['prefix_contaminated'],physical_tail_label_policy='labels mark direct injection time, not propagated effects',
        labels_detail=dict(effect=None,anomaly_channel=None,anomaly_time=None)))
    return dict(key=key,path=path.name,SHA256=sha256(path),C=c,family=family,seed=seed,case_metadata=meta,
        cases_SHA256=array_sha(cases),graph_SHA256=array_sha(g),labels_SHA256=array_sha(labels),prefix_contaminated=row['prefix_contaminated'],
        stored_input_slots=cases.size,base_generator_allocated_slots=4*T*c,replayed_state_updates=30*512*c)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    assert not a.output.exists();a.output.mkdir(parents=True)
    jobs=[(c,i,str(a.output)) for c in [8,12,16,20] for i in range(8)]
    with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:rows=list(pool.map(build,jobs))
    assert len({r['seed'] for r in rows})==32
    atomic_json(a.output/'manifest.json',dict(status='complete',systems=32,cases=1344,files=rows,
        stored_input_slots=sum(v['stored_input_slots'] for v in rows),base_generator_allocated_slots=sum(v['base_generator_allocated_slots'] for v in rows),
        replayed_state_updates=sum(v['replayed_state_updates'] for v in rows),benchmark_arrays_opened=False,optimizer_updates=0,
        seed_namespace=348004000000,full_original_training_array_overlap_scan_performed=False,
        labels_detail=dict(effect=None,anomaly_channel=None,anomaly_time=None)))
