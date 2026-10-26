"""Content-addressed source evidence. No label/root/SCM parameter enters the kernel."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from proposals.causal_syndromes import condition,evidence
from proposals.matched_pairs import rewire
from causal_tsad.io import atomic_json,sha256,array_sha
MODES=['true_signature','true_self','true_unit','wrong_signature','empty_signature']
def run(a):
 a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory);manifest=json.loads((a.source/'manifest.json').read_text());assert all(m in MODES for m in a.modes);tasks=[(mode,row) for mode in a.modes for row in manifest['files']];order=np.random.RandomState(348).permutation(len(tasks));records=[];query=slots=factors=conditioning=0;start=time.monotonic()
 for pos in range(a.worker,len(tasks),a.workers):
  assert time.monotonic()-start<3*3600 and time.time()<a.deadline;mode,row=tasks[int(order[pos])];path=a.source/row['path'];assert sha256(path)==row['SHA256']
  with np.load(path) as f:cases=f['cases'];graph=f['graph'];assert array_sha(cases)==row['input_cases_SHA256'] and array_sha(graph)==row['graph_SHA256']
  g=torch.tensor(graph,dtype=torch.float64,device='cuda');g=rewire(g,row['key']) if mode=='wrong_signature' else torch.zeros_like(g) if mode=='empty_signature' else g;prefix=torch.tensor(cases[0,:256],dtype=torch.float64,device='cuda');assert all(np.array_equal(v[:256],cases[0,:256]) for v in cases);state=condition(prefix,g);conditioning+=row['C']*3;results=[];nll=[]
  for view in range(row['model_views']):
   values=torch.tensor(cases[view],dtype=torch.float64,device='cuda');out=evidence(values,g,signature='self' if mode=='true_self' else 'unit' if mode=='true_unit' else 'full',state=state);results.append(out['log_factors'].cpu().numpy());nll.append(float(out['normal_NLL']));factors+=out['factorizations'];query+=704*row['C'];slots+=values.numel()
  outpath=a.output/mode/(row['key']+'.npz');outpath.parent.mkdir(exist_ok=True);assert not outpath.exists();np.savez_compressed(outpath,log_factors=np.stack(results),normal_NLL=np.array(nll));records.append(dict(mode=mode,key=row['key'],path=str(outpath.relative_to(a.output)),SHA256=sha256(outpath),input_file_SHA256=row['SHA256'],prefix_SHA256=row['prefix_SHA256'],used_graph_SHA256=array_sha(g.cpu().numpy()),split=row['split'],C=row['C'],family=row['family']));atomic_json(a.output/('worker'+str(a.worker)+'_progress.json'),dict(status='running',completed=len(records),query_scalar_forwards=query,full_input_scalar_slots=slots,Gaussian_covariance_factorizations=factors,context_ridge_factorizations=conditioning,optimizer_updates=0,benchmark_arrays_opened=False))
 atomic_json(a.output/('worker'+str(a.worker)+'_manifest.json'),dict(status='complete',records=records,query_scalar_forwards=query,full_input_scalar_slots=slots,Gaussian_covariance_factorizations=factors,context_ridge_factorizations=conditioning,source_manifest_SHA256=sha256(a.source/'manifest.json'),optimizer_updates=0,benchmark_arrays_opened=False));atomic_json(a.output/('worker'+str(a.worker)+'_progress.json'),dict(status='complete',completed=len(records),query_scalar_forwards=query,full_input_scalar_slots=slots,Gaussian_covariance_factorizations=factors,context_ridge_factorizations=conditioning,optimizer_updates=0,benchmark_arrays_opened=False,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['source','output']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--modes',nargs='+',choices=MODES,required=True);p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,required=True);p.add_argument('--deadline',type=float,default=float('inf'));a=p.parse_args()
 try:run(a)
 except Exception:atomic_json(a.output/('worker'+str(a.worker)+'_error.json'),dict(error=traceback.format_exc()));raise
