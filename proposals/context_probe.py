"""Fixed prefix versus equally sized, guarded recent prefix. No target gradients."""
import torch
from proposals.causal_syndromes import evidence,FIRST
@torch.no_grad()
def predict(values,graph,variant):
 assert variant in ['fixed','rolling'];outputs=[];slots=queries=factors=conditioning=0
 starts=[FIRST] if variant=='fixed' else range(FIRST,len(values),256)
 for start in starts:
  lo=0 if variant=='fixed' else start-FIRST;hi=len(values) if variant=='fixed' else min(len(values),start+256);x=values[lo:hi];out=evidence(x,graph);outputs.append(out);slots+=x.numel();queries+=len(out['log_factors'])*values.shape[1];factors+=out['factorizations'];conditioning+=3*values.shape[1]
 return dict(log_factors=torch.cat([o['log_factors'] for o in outputs]),residual=torch.cat([o['ready_residual'] for o in outputs]),variance=torch.cat([o['ready_variance'] for o in outputs]),query_scalar_forwards=queries,full_input_scalar_slots=slots,Gaussian_covariance_factorizations=factors,context_ridge_factorizations=conditioning)
