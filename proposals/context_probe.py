"""Only the variance-matched likelihood changes; normal predictions are untouched."""
import torch
from proposals.causal_syndromes import evidence
@torch.no_grad()
def predict(values,graph,variant):
 assert variant=='student';out=evidence(values,graph)
 return dict(log_factors=out['log_factors'],residual=out['ready_residual'],variance=out['ready_variance'],query_scalar_forwards=len(out['log_factors'])*values.shape[1],full_input_scalar_slots=values.numel(),Gaussian_covariance_factorizations=out['factorizations'],context_ridge_factorizations=3*values.shape[1])
