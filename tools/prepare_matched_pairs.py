"""Generate independent source only; audit every quartet before allowing training."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import json,argparse,sys,hashlib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from proposals.matched_pairs import generate
from causal_tsad.io import atomic_json,sha256,array_sha

def one(task):
 split,c,idx=task;seed=348_000_000_000+{'train':0,'calibration':1_000_000,'source_validation':2_000_000}[split]+c*10000+idx
 family=(['linear','tanh','sin'] if split=='source_validation' else ['linear','tanh'])[idx%(3 if split=='source_validation' else 2)]
 return split,c,idx,generate(c,seed,family)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--benchmark-index',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 forbidden={r[k] for r in json.loads(a.benchmark_index.read_text()) for k in ['time_series_sha256','normal_sha256']};tasks=[(split,c,i) for split,n in [('train',64),('calibration',16),('source_validation',32)] for c in [8,12,16,20] for i in range(n)]
 by={};params={};matched=0
 with ProcessPoolExecutor(max_workers=16) as pool:
  for split,c,i,row in pool.map(one,tasks,chunksize=1):
   key=split+'/'+str(c);by.setdefault(key,[]).append(row);params[split+'/'+str(c)+'/'+str(i)]=dict(seed=row['seed'],family=row['family'],root=row['root'],prefix_contaminated=row['prefix_contaminated'],**row['audit_parameters'])
   for view in row['cases']:assert array_sha(view) not in forbidden
   matched+=1
   if matched%16==0:atomic_json(a.output/'progress.json',dict(status='generating',completed=matched,total=len(tasks)))
 files=[]
 for key,rows in by.items():
  split,c=key.split('/');rows.sort(key=lambda r:r['seed']);path=a.output/(split+'_c'+c+'.npz')
  np.savez_compressed(path,cases=np.stack([r['cases'] for r in rows]),labels=np.stack([r['labels'] for r in rows]),graph=np.stack([r['graph'] for r in rows]),root=np.array([r['root'] for r in rows]),
   seed=np.array([r['seed'] for r in rows]),family=np.array([r['family'] for r in rows]),prefix_contaminated=np.array([r['prefix_contaminated'] for r in rows]))
  files.append(dict(path=path.name,split=split,channels=int(c),systems=len(rows),SHA256=sha256(path),observed_scalars=len(rows)*4*1024*int(c)))
 atomic_json(a.output/'audit_parameters_NOT_MODEL_INPUT.json',params)
 atomic_json(a.output/'manifest.json',dict(status='complete',files=files,independent_mechanism_groups=448,cases=1792,observed_scalars=sum(r['observed_scalars'] for r in files),
  full_univariate_class_multiset_exact_matching_groups=448,source_train_groups=256,source_calibration_groups=64,source_validation_groups=128,
  benchmark_index_SHA256=sha256(a.benchmark_index),benchmark_raw_array_matches=0,benchmark_arrays_opened=False,original_2_5B_or_250M_modified=False,
  source_seed_namespace='uint64 >=348000000000, independent source generator; not original uint32 seeds',source_mechanism_exposure='train linear/tanh; calibration linear/tanh; source validation includes sin',
  generator_SHA256=sha256(Path(__file__).resolve().parents[1]/'proposals/matched_pairs.py'),normal_twins_and_parameters_not_predictor_inputs=True,new_channel_effect_labels=None,
  note='group is independent unit; 4 paired cases within a group are dependent. Only ready t>=320 source mechanism metrics, not full benchmark TSAD.'))
 atomic_json(a.output/'progress.json',dict(status='complete',completed=matched,total=len(tasks)));print('448 matched source groups generated and audited')
