"""Fixed channel-wise second difference, a causal trend-null diagnostic frontend.

Not adaptive patching, not a complete level-anomaly detector, not a novelty claim.
"""
import torch
from proposals.causal_contrast import CausalContrast
def second_difference(values):
 assert values.ndim==2 and values.dtype==torch.float64
 out=torch.zeros_like(values);out[2:]=values[2:]-2*values[1:-1]+values[:-2];return out
class TrendQuotientContrast(CausalContrast):
 def __init__(self):super().__init__();self.frontend_scalar_slots=0
 def forward(self,values,graph,mode):
  # Require raw float64 so a large trend is not rounded before cancellation.
  self.frontend_scalar_slots+=values.numel();return super().forward(second_difference(values).float(),graph,mode)
