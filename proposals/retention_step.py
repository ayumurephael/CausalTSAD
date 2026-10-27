"""A-GEM half-space principle applied to actual AdamW displacement, per parameter block.

Mature projection, not a new continual-learning algorithm. A clean minibatch loss
direction is protected only to first order; this does not guarantee AP or generalization.
"""
import torch
def vector(parameters,gradient=False):
 return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).double() if gradient else p.detach().reshape(-1).double() for p in parameters])
@torch.no_grad()
def apply_vector(parameters,values):
 offset=0
 for p in parameters:p.copy_(values[offset:offset+p.numel()].reshape_as(p).to(p.dtype));offset+=p.numel()
 assert offset==values.numel()
def halfspace(displacement,reference):
 dot=displacement.dot(reference);norm=reference.dot(reference)
 return displacement-dot/norm*reference if dot<0 and norm>0 else displacement
@torch.no_grad()
def protect(parameters,before,reference,enabled):
 after=vector(parameters);delta=before-after;dot=delta.dot(reference);norm=reference.dot(reference);projected=halfspace(delta,reference)
 if enabled and dot<0 and norm>0:
  # Move just inside the feasible halfspace to cover float32 parameter rounding.
  # The bound is disclosed; closest-point equality is not claimed after rounding.
  margin=2*torch.finfo(torch.float32).eps*((before.abs()+projected.abs())*reference.abs()).sum()
  projected=projected+margin/norm*reference;apply_vector(parameters,before-projected)
 actual=before-vector(parameters);actual_dot=actual.dot(reference)
 if enabled:assert actual_dot>=-1e-10,('rounded actual step violates protection',float(actual_dot))
 return dict(proposed_dot_clean_gradient=float(dot),applied_dot_clean_gradient=float(actual_dot),proposed_step_norm=float(delta.norm()),applied_step_norm=float(actual.norm()),projected=bool(enabled and dot<0 and norm>0))
