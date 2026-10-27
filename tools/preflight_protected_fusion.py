"""Checkpoint-preserving streaming and exact zero-correction identity."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from dependencies.c04_v1_model import CausalContrast as Reference
from proposals.causal_contrast import CausalContrast
from proposals.raw_context import signed_log
from proposals.protected_fusion import ProtectedFusion,causal_features,parameter_sha
from causal_tsad.io import atomic_json,sha256
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(348);torch.backends.cuda.matmul.allow_tf32=False
 import json
 cfg=json.loads((Path(__file__).resolve().parents[1]/'experiments/protected_fusion_v2.json').read_text());assert all(k in cfg for k in ['data_seed','learning_rate','weight_decay','old_source_trust_weight'])
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m=CausalContrast().cuda().eval().requires_grad_(False);ref=Reference().cuda().eval().requires_grad_(False);m.load_state_dict(saved['model'],strict=True);ref.load_state_dict(saved['model'],strict=True);before=parameter_sha(m)
 x=torch.randn(1041,20,device='cuda');g=torch.zeros(20,20,device='cuda');g[torch.arange(19),torch.arange(1,20)]=1
 with torch.inference_mode():
  old=ref(x,g,saved['mode']);new=m(x,g,saved['mode']);full,_=causal_features(m,x,g,saved['mode'],chunk=10000);stream,roots=causal_features(m,x,g,saved['mode'],chunk=256)
  exact=float((old['logit']-new['logit']).abs().max());assert exact==0
  feature_error=float((full-stream).abs().max());root_error=float((old['root_logits']-roots).abs().max());assert feature_error<1e-5 and root_error<1e-5
 assert parameter_sha(m)==before
 fusion=ProtectedFusion([0]*7,[1]*7,2.).cuda();features=torch.randn(721,7,device='cuda');base=torch.randn(1041,device='cuda');out,delta=fusion.full_scores(base,features,'true',20);assert torch.equal(out,base) and delta.count_nonzero()==0
 with torch.no_grad():fusion.residual[-1].bias.fill_(1.)
 out,delta=fusion.full_scores(base,features,'true',20);assert torch.equal(out[:320],base[:320])
 c1,_=fusion.full_scores(base,features,'true',1);assert torch.equal(c1,base)
 changed=features.clone();changed[:,1:5]+=100;assert torch.equal(fusion(base[320:],features,'basecal')[0],fusion(base[320:],changed,'basecal')[0])
 fusion.zero_grad(set_to_none=True);loss=fusion(base[320:],features,'true')[0].square().mean();loss.backward();assert all(torch.isfinite(q.grad).all() for q in fusion.parameters() if q.grad is not None)
 atomic_json(a.output,dict(status='passed',old_full_new_full_logit_error=exact,streaming_feature_max_error=feature_error,streaming_root_logit_max_error=root_error,
  zero_correction_exact=True,C1_exact=True,warmup320_exact=True,basecal_causal_feature_invariance=True,fusion_parameters=sum(q.numel() for q in fusion.parameters()),
  teacher_parameter_SHA_unchanged=True,checkpoint_SHA256=sha256(a.checkpoint),optimizer_updates=0,target_arrays_opened=False,random_unique_input_scalars=x.numel(),
  random_teacher_forward_query_scalars=4*721*20,random_head_forward_time_points=7*721,random_head_backward_time_points=721,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
