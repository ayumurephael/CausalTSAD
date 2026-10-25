"""Mask legality and exact no-mask compatibility, no optimization."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from proposals.bayes_propagation import BayesianPropagation,generate,availability_mask
from causal_tsad.io import atomic_json,sha256
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(1.5*2**30/torch.cuda.get_device_properties(0).total_memory)
 folders=list((a.original/'runs').iterdir());folder=next(f for f in folders if (f/'scores/0_dag_delay_mixture.npz').exists());params=json.loads((folder/'source_fitted_parameters.json').read_text());m=BayesianPropagation(params);r=generate(0,'evaluation');x=r['cases'][6];g=r['graph'];mask=availability_mask(0,len(g));old=np.load(folder/'scores/0_dag_delay_mixture.npz')['score'][6]
 full=m.infer(x,g)['score'];allvalid=m.infer(x,g,observed_mask=np.ones_like(mask))['score'];assert np.array_equal(full,old) and np.array_equal(full,allvalid)
 clean=m.infer(x,g,observed_mask=mask);corrupt=x.copy();corrupt[~mask]=1e100;changed=m.infer(corrupt,g,observed_mask=mask);assert np.array_equal(clean['score'],changed['score']) and np.array_equal(clean['root'],changed['root'])
 future=x.copy();future[101:]*=1e3;out=m.infer(future,g,observed_mask=mask);assert np.array_equal(clean['score'][:101],out['score'][:101])
 perm=np.arange(len(g))[::-1].copy();out=m.infer(x[:,perm],g[perm][:,perm],observed_mask=mask[:,perm]);assert np.max(np.abs(clean['score']-out['score']))<1e-12 and np.max(np.abs(clean['root'][:,perm]-out['root']))<1e-12
 atomic_json(a.output,dict(status='passed',original_scores_exact=True,all_valid_exact=True,invalid_value_invariance_exact=True,future_invariance_exact=True,channel_permutation_error=float(np.max(np.abs(clean['score']-out['score']))),optimizer_updates=0,benchmark_arrays_opened=False,actual_technical_full_input_scalar_forwards=6*x.size,hypothesis_time_cells=m.hypothesis_time_cells,original_parameters_SHA256=sha256(folder/'source_fitted_parameters.json')))
