"""Same diagonal and channel marginals, independent direct Gaussian references."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from proposals.causal_syndromes import covariance_log_ratio,evidence
from causal_tsad.io import atomic_json,sha256
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--original-root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(1*2**30/torch.cuda.get_device_properties(0).total_memory);torch.manual_seed(348);d='cuda';r=torch.randn(2,12,dtype=torch.float64,device=d);v=torch.rand_like(r)+1;m=torch.randn(2,12,5,dtype=r.dtype,device=d);cov=torch.diag_embed(v)+2*m@m.transpose(-1,-2);base=torch.diag_embed(v);errors={}
 for structure in ['full','diagonal','channel_block']:
  alternative=cov.clone()
  if structure=='diagonal':alternative=torch.diag_embed(alternative.diagonal(dim1=-2,dim2=-1))
  elif structure=='channel_block':
   ch=torch.arange(12,device=d)%3;alternative=torch.where(ch[None,:,None]==ch[None,None,:],alternative,torch.zeros_like(alternative))
  direct=.5*(torch.linalg.slogdet(base).logabsdet-torch.linalg.slogdet(alternative).logabsdet+(r[...,None]*(torch.linalg.solve(base,r[...,None])-torch.linalg.solve(alternative,r[...,None]))).sum((-2,-1)));out=covariance_log_ratio(r,v,m,2,structure,3);errors[structure]=float((out-direct).abs().max());assert errors[structure]<1e-9
  assert torch.equal(alternative.diagonal(dim1=-2,dim2=-1),cov.diagonal(dim1=-2,dim2=-1))
 # Verify the untouched full path against a saved original source evidence array.
 import json
 physical=torch.zeros(2,12,4,dtype=r.dtype,device=d);physical[:,torch.arange(4,device=d)*3,torch.arange(4,device=d)]=1
 nulls=[covariance_log_ratio(r,v,physical,2,mode,3) for mode in ['full','diagonal','channel_block']];physical_error=max(float((z-nulls[0]).abs().max()) for z in nulls);assert physical_error<1e-9
 manifest=json.loads((a.original_root/'cache/manifest.json').read_text());entry=next(r for r in manifest['records'] if r['mode'] in ['true_signature','true_unit']);source=json.loads((a.original_root/'source/manifest.json').read_text());row=next(r for r in source['files'] if r['key']==entry['key'])
 with np.load(a.original_root/'source'/row['path']) as f:x=torch.tensor(f['cases'][0],dtype=torch.float64,device=d);g=torch.tensor(f['graph'],dtype=torch.float64,device=d)
 expected=np.load(a.original_root/'cache'/entry['path'])['log_factors'][0];actual=evidence(x,g,signature='unit' if entry['mode']=='true_unit' else 'full')['log_factors'].cpu().numpy();error=float(np.max(np.abs(actual-expected)));assert error<1e-6
 atomic_json(a.output,dict(status='passed',direct_Gaussian_max_errors=errors,same_pointwise_variances=True,channel_block_retains_each_channel_temporal_covariance=True,physical_only_null_max_error=physical_error,original_full_covariance_log_factor_max_error=error,original_cache_mode=entry['mode'],original_cache_SHA256=sha256(a.original_root/'cache'/entry['path']),optimizer_updates=0,benchmark_arrays_opened=False,technical_raw_input_slots=x.numel(),technical_Gaussian_factorizations=actual.shape[0]*actual.shape[1]*6,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
