"""Source-only 2x2 phase/period intervention, frozen normal/fixed models."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from sklearn.metrics import average_precision_score
from proposals.causal_meta_basis import CausalMetaBasis
from proposals.raw_context import parameter_sha
from tools.train_meta_basis import load,graph_for
from causal_tsad.io import atomic_json,sha256,array_sha
def background(cases,key,period,phase):
    scale=np.median(np.abs(cases[0,:256]-np.median(cases[0,:256],axis=0)),axis=0)*1.4826;scale=np.where(scale<1e-3,1,scale)
    sign=np.random.RandomState((int(key)+127)%2**32).choice([-1.,1.],cases.shape[-1]);omega=2*np.pi/(period*cases.shape[1]);angle=omega*(np.arange(cases.shape[1])-127.5)[:,None]+(0 if phase=='crossing' else np.pi/2)
    b=4*scale[None]*sign[None]*np.sin(angle);return cases+b[None]
def run(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    plan=json.loads(a.plan.read_text());assert sha256(Path(plan['source'])/'manifest.json')==plan['source_SHA256'];manifest=json.loads((Path(plan['source'])/'manifest.json').read_text());rows=[v for v in manifest['files'] if v['split']=='source_validation'];assert len(rows)==32
    models={};hashes={};weights={};records=[];slots=queries=factors=0;start=time.time()
    for seed in [348,349,350]:
        name=a.arm+'_s'+str(seed);p=Path(plan['weights'])/(name+'.pth');assert sha256(p)==plan['weight_SHA256'][name];ck=torch.load(p,map_location='cpu',weights_only=False);m=CausalMetaBasis(seed).cuda().eval().requires_grad_(False);m.load_state_dict(ck['model'],strict=True);models[seed]=m;hashes[seed]=parameter_sha(m);weights[name]=sha256(p)
    for row in rows:
        cases,g=load(row,plan['source']);gg=graph_for(g,row['key'],a.mode);graph=torch.tensor(gg,dtype=torch.float64,device='cuda')
        for period in [.75,1.5]:
            for phase in ['crossing','extremum']:
                assert time.time()<plan['deadline_unix'];values=background(cases,row['key'],period,phase);assert all(sorted(array_sha(values[i,:,ch]) for i in [0,1])==sorted(array_sha(values[i,:,ch]) for i in [2,3]) for ch in range(row['C']));x=torch.tensor(values,dtype=torch.float64,device='cuda');lo=values[:2,:256].min(1);hi=values[:2,:256].max(1);outside=(values[:2,320:]<lo[:,None])|(values[:2,320:]>hi[:,None]);methods={};scores=[]
                for seed,m in models.items():
                    with torch.no_grad():o=m(x,graph)
                    score=o['score'].cpu().numpy();root=o['root'].cpu().numpy();theta=plan['thresholds'][a.arm+'/'+str(seed)+'/'+a.mode]['threshold'];q=(o['residual'][:2].square()/o['variance'][:2]).cpu().numpy();rca=[]
                    for view in range(2,8):
                        order=np.argsort(-root[view].mean(0),kind='stable');rank=int(np.flatnonzero(order==row['root'])[0])+1;rca.append(dict(kind=row['case_kind'][view],Top1=int(rank==1),Top3=int(rank<=min(3,row['C'])),MRR=1/rank))
                    methods[str(seed)]=dict(AP=float(average_precision_score(np.repeat([0,0,1,1],704),score[:4].reshape(-1))),physical_AP=float(average_precision_score(np.repeat([0,0,1,1],704),score[[0,1,6,7]].reshape(-1))),FPR=float((score[:2]>theta).mean()),normal_NLL=float(o['normal_nll'][:2].mean()),normal_whitened_square=float(q.mean()),normal_whitened_q99=float(np.quantile(q,.99)),time_quarters_whitened=[float(z.mean()) for z in np.array_split(q,4,axis=1)],RCA=rca)
                    scores.append(score);slots+=values.size;queries+=8*704*row['C'];factors+=o['context_factorizations']
                p=a.root/'records'/a.arm/a.mode/(row['key']+'_p'+str(period).replace('.','p')+'_'+phase);p.parent.mkdir(parents=True,exist_ok=True);assert not p.with_suffix('.npz').exists();np.savez_compressed(p.with_suffix('.npz'),scores=np.stack(scores));record=dict(key=row['key'],C=row['C'],family=row['family'],arm=a.arm,mode=a.mode,period=period,phase=phase,full_amplitude_multiplier=4,prefix_center=127.5,input_SHA256=array_sha(values),graph_SHA256=array_sha(gg),source_file_SHA256=row['SHA256'],methods=methods,normal_outside_prefix_range_fraction=float(outside.mean()),normal_prefix_range_median=float(np.median(hi-lo)),normal_query_range_median=float(np.median(np.ptp(values[:2,320:],axis=1))),scores_SHA256=sha256(p.with_suffix('.npz')),source_only=True,target_updates=0,optimizer_updates=0)
                atomic_json(p.with_suffix('.json'),record);records.append(dict(path=str(p.with_suffix('.json').relative_to(a.root)),SHA256=sha256(p.with_suffix('.json')),scores_SHA256=record['scores_SHA256']));atomic_json(a.root/(a.arm+'_'+a.mode+'_progress.json'),dict(status='running',completed=len(records),full_input_scalar_slots=slots,query_scalar_forwards=queries))
    assert all(parameter_sha(m)==hashes[s] for s,m in models.items());atomic_json(a.root/(a.arm+'_'+a.mode+'_manifest.json'),dict(status='complete',records=records,weight_SHA256=weights,parameters_unchanged=True,full_input_scalar_slots=slots,query_scalar_forwards=queries,context_factorizations=factors,source_only=True,benchmark_arrays_opened=False,optimizer_updates=0,seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated()));atomic_json(a.root/(a.arm+'_'+a.mode+'_progress.json'),dict(status='complete',completed=len(records)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--arm',choices=['frozen_basis','normal_only'],required=True);p.add_argument('--mode',choices=['true','empty'],required=True);a=p.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors'/(a.arm+'_'+a.mode+'.json'),dict(error=traceback.format_exc()));raise
