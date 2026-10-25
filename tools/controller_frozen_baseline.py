"""One node owns disjoint development systems; release GPUs before CPU metrics."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256

def main(a):
 r=a.root;code=Path(__file__).resolve().parents[1];plan=json.loads((r/'job_plan.json').read_text());start=time.monotonic();gpus=plan['gpus']
 with (r/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 assert json.loads((r/'preflight.json').read_text())['status']=='passed' and json.loads((r/'calibration.json').read_text())['status']=='complete'
 assert sha256(r/'calibration.json')==plan['calibration_SHA256'];progress=dict(status='running',stage='GPU_inference',pid=os.getpid(),jobs=len(plan['rows']),active=[])
 atomic_json(r/'controller_progress.json',progress)
 groups=[[] for _ in gpus];cost=[0]*len(gpus)
 # Prioritize the old100, then greedily balance dense attention work.
 for row in sorted(plan['rows'],key=lambda z:(not z['pilot100'],-z['length']*z['channels']**2)):
  i=min(range(len(gpus)),key=lambda j:cost[j]);groups[i].append(row);cost[i]+=row['length']*row['channels']**2
 children=[]
 for i,(gpu,rows) in enumerate(zip(gpus,groups)):
  jobs=r/'jobs'/f'gpu_{gpu}.json';atomic_json(jobs,rows);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTHONUNBUFFERED']='1'
  cmd=[sys.executable,str(code/'tools/frozen_inference.py'),'--root',str(r),'--benchmark',str(a.benchmark),'--original-code',str(a.original_code),'--weights',str(a.weights),'--jobs',str(jobs)]
  with (r/'logs'/f'gpu_{gpu}.log').open('x') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
  children.append(p);progress['active'].append(dict(gpu=gpu,pid=p.pid,systems=len(rows)))
 atomic_json(r/'controller_progress.json',progress)
 while any(p.poll() is None for p in children):
  if any(p.poll() not in [None,0] for p in children):raise RuntimeError('Frozen inference failed, preserved outputs; no metrics started')
  if time.monotonic()-start>6*3600:raise TimeoutError('GPU6h deadline')
  time.sleep(5)
 assert all(p.returncode==0 for p in children)
 progress.update(stage='CPU_metrics',active=[]);atomic_json(r/'controller_progress.json',progress)
 jobs=r/'jobs/all_node_rows.json';atomic_json(jobs,plan['rows'])
 cmd=[sys.executable,str(code/'tools/metric_frozen.py'),'--root',str(r),'--benchmark',str(a.benchmark),'--metric-code',str(a.metric_code),'--jobs',str(jobs),'--workers','8']
 with (r/'logs/metrics.log').open('x') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
 progress.update(metrics_pid=p.pid);atomic_json(r/'controller_progress.json',progress)
 while p.poll() is None:
  if time.monotonic()-start>12*3600:raise TimeoutError('Frozen total12h technical deadline')
  time.sleep(5)
 assert p.returncode==0;progress.update(status='complete',stage='complete');atomic_json(r/'controller_progress.json',progress)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['root','benchmark','original-code','weights','metric-code']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 try:main(a)
 except Exception:
  path=a.root/'controller_progress.json';z=json.loads(path.read_text()) if path.exists() else {};z.update(status='needs_attention',error=traceback.format_exc());atomic_json(path,z);raise
