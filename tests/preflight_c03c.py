"""Nonzero frozen weights: true-operator null, degree equality and query boundaries."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from causal_tsad.io import atomic_json
from dependencies.c02_model import ParentTiming
from proposals.c09.graph_views import ValueGraphTeacher,views,value_graph
from tools.cache_c09 import legal_views

def main(out):
 torch.set_num_threads(1);torch.manual_seed(348);torch.backends.cuda.matmul.allow_tf32=False
 original=ParentTiming().cuda().eval();model=ValueGraphTeacher().cuda().eval();model.load_state_dict(original.state_dict(),strict=True)
 x=torch.randn(1,192,5,device='cuda');g=torch.zeros(1,5,5,device='cuda');g[:,0,2]=1;g[:,1,2]=1;g[:,2,3]=1
 expected=legal_views(original,x,g);actual=views(model,x,g,'true','test')
 true_error=max(float((a-b).abs().max()) for a,b in zip(actual,expected));assert true_error==0
 checks={}
 for mode in ['degree_only','wrong_degree']:
  ctx,r,mu,lv=views(model,x,g,mode,'test');assert torch.equal(ctx[...,10],expected[0][...,10])
  changed=x.clone();changed[:,128,2]+=100;other=views(model,changed,g,mode,'test');own=float((other[0][:,128,2]-ctx[:,128,2]).abs().max());assert own==0
  changed=x.clone();changed[:,129:]+=100;other=views(model,changed,g,mode,'test');future=float((other[0][:,:129]-ctx[:,:129]).abs().max());assert future==0
  if mode=='degree_only':
   changed=x.clone();changed[:,:,0]+=100;other=views(model,changed,g,mode,'test');null=float((other[0][:,:,2]-ctx[:,:,2]).abs().max());assert null==0
  else:null=None
  control=value_graph(g[0].cpu().numpy(),mode,'test');checks[mode]=dict(own_query_error=own,future_error=future,no_parent_value_error=null,
   self_edges=int(np.trace(control)),incoming_degree_preserved=(np.array_equal(control.sum(0),g[0].cpu().numpy().sum(0)) if mode=='wrong_degree' else 'true degree retained separately'))
 atomic_json(out,dict(status='passed',original_teacher_true_operator_error=true_error,true_degree_feature_identical=True,checks=checks,
  source_arrays_opened=False,target_arrays_opened=False,optimizer_updates=0,random_observed_scalars=x.numel(),teacher_weights_updated=False,
  scope='Wrong values are an explicit noncausal predictive control, not a causal graph input claim'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
