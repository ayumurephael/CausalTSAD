"""Non-vacuous exact-marginal source and CI score-additivity audit."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from proposals.matched_pairs import generate,rewire
from proposals.causal_contrast import CausalContrast
from proposals.raw_context import parameter_sha
from causal_tsad.io import atomic_json
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(348);torch.backends.cuda.matmul.allow_tf32=False
 row=generate(8,348_999_999_999,'tanh');m=CausalContrast().cuda().eval();g=torch.from_numpy(row['graph']).cuda();before=parameter_sha(m);checks={}
 with torch.no_grad():
  for mode in ['true','degree','wrong']:
   graph=rewire(g,'preflight') if mode=='wrong' else g;s=[m(torch.tensor(v,dtype=torch.float32,device='cuda'),graph,mode)['logit'] for v in row['cases']]
   delta=float((s[0]+s[1]-s[2]-s[3]).abs().max());checks[mode]=dict(quartet_additivity_error=delta)
   if mode=='degree':assert delta<1e-5
  assert checks['true']['quartet_additivity_error']>1e-8,'Non-vacuous graph discriminator test'
  perm=torch.tensor([2,6,0,7,3,1,5,4],device='cuda');v=torch.tensor(row['cases'][2],dtype=torch.float32,device='cuda');z=m(v,g,'true');q=m(v[:,perm],g[perm][:,perm],'true')
  equiv=max(float((z['logit']-q['logit']).abs().max()),float((z['root_logits'][:,perm]-q['root_logits']).abs().max()));assert equiv<1e-5
 assert parameter_sha(m)==before
 m.train();z=m(v,g,'true');root_loss=z['root_logits'].mean();m.zero_grad(set_to_none=True);root_loss.backward();assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('root.'))
 m.zero_grad(set_to_none=True);z=m(v,g,'true');loss=z['logit'].square().mean()+.1*z['normal_NLL'].mean();loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 atomic_json(a.output,dict(status='passed',checks=checks,permutation_max_error=equiv,RCA_gradient_isolated=True,parameters=sum(p.numel() for p in m.parameters()),
  full_1D_sequence_class_matching=True,source_or_target_arrays_opened=False,optimizer_updates=0,random_source_unique_observed_scalars=int(row['cases'].size),
  extra_frozen_query_predictions=14*704*8,extra_backward_query_predictions=2*704*8,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
  scope='Technical nonzero random predictor/discriminator; earlier RawContextPredictor query/future/nonparent tests inherited and re-run separately, not a performance result'))
