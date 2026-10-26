"""Hard equality/causality checks for the one guarded-recency interface."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from proposals.context_probe import predict
from proposals.causal_syndromes import evidence,SyndromePrior
from proposals.raw_context import parameter_sha
from causal_tsad.io import atomic_json
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(348);torch.cuda.set_per_process_memory_fraction(1*2**30/torch.cuda.get_device_properties(0).total_memory);x=torch.randn(641,3,dtype=torch.float64,device='cuda');g=torch.zeros(3,3,dtype=x.dtype,device=x.device);g[0,1]=g[1,2]=1;original=evidence(x,g);fixed=predict(x,g,'fixed');rolling=predict(x,g,'rolling');assert torch.equal(original['log_factors'],fixed['log_factors']);first=float((fixed['log_factors'][:256]-rolling['log_factors'][:256]).abs().max());short=evidence(x[:576],g);assert torch.equal(short['log_factors'],rolling['log_factors'][:256]);assert first<=1e-12
 future=x.clone();future[600:]+=100;changed=predict(future,g,'rolling');assert torch.equal(rolling['log_factors'][:280],changed['log_factors'][:280]);perm=torch.tensor([2,0,1],device=x.device);permuted=predict(x[:,perm],g[perm][:,perm],'rolling');prior=SyndromePrior().cuda().requires_grad_(False);before=parameter_sha(prior);r=prior(rolling['log_factors']);pout=prior(permuted['log_factors']);error=float((r['logit']-pout['logit']).abs().max());assert error<1e-6 and before==parameter_sha(prior)
 assert rolling['query_scalar_forwards']==fixed['query_scalar_forwards'];assert rolling['full_input_scalar_slots']>fixed['full_input_scalar_slots'];atomic_json(a.output,dict(status='passed',fixed_kernel_exact=True,first_block_same_shape_reference_exact=True,first_block_original_whole_length_error=first,first_block_original_whole_length_tolerance=1e-12,future_isolation_exact=True,permutation_error=error,global_parameters_unchanged=True,query_points_matched=True,rolling_extra_input_accounted=True,technical_query=original['log_factors'].shape[0]*3+short['log_factors'].shape[0]*3+sum(z['query_scalar_forwards'] for z in [fixed,rolling,changed,permuted]),technical_full_input_slots=x.numel()+576*3+sum(z['full_input_scalar_slots'] for z in [fixed,rolling,changed,permuted]),optimizer_updates=0,benchmark_arrays_opened=False,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
