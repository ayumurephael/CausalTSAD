"""One immutable source/dev cache per graph mode, all teacher weights frozen."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256

def main(a):
 root=a.root;code=Path(__file__).resolve().parents[1];cfg=code/'experiments/c03_specificity_v3.json';start=time.monotonic();refs=json.loads((a.base/'manifest.json').read_text());gpus=json.loads(a.gpus)
 with (root/'cache_preparation.lock').open('x') as f:f.write(str(os.getpid()))
 def launch(args,name,gpu):
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTHONUNBUFFERED']='1'
  with (root/'logs'/(name+'.log')).open('x') as f:return subprocess.Popen([sys.executable]+list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
 def wait(ps):
  while any(p.poll() is None for p in ps):
   if any(p.poll() not in [None,0] for p in ps):raise RuntimeError('Cache/G0 failed, no new project stages')
   if time.monotonic()-start>6*3600:raise TimeoutError('Fixed6h cache technical limit')
   time.sleep(3)
  assert all(p.returncode==0 for p in ps)
 wait([launch([code/'tests/preflight_c03c.py','--output',root/'graph_preflight.json'],'graph_preflight',gpus[0])]);assert json.loads((root/'graph_preflight.json').read_text())['status']=='passed'
 atomic_json(root/'cache_progress.json',dict(status='running',mode='start',pid=os.getpid(),optimizer_updates=0))
 for mode in ['degree_only','wrong_degree']:
  out=root/'caches'/mode;out.mkdir(parents=True,exist_ok=False);ps=[]
  for i,gpu in enumerate(gpus):ps.append(launch([code/'tools/cache_c03c.py','--base',a.base,'--source',a.source,'--benchmark',a.benchmark,'--teacher',a.teacher,'--config',cfg,'--output',out,'--mode',mode,'--worker',i,'--workers',len(gpus)],mode+'_'+str(i),gpu))
  atomic_json(root/'cache_progress.json',dict(status='running',mode=mode,pid=os.getpid(),workers=[p.pid for p in ps],optimizer_updates=0));wait(ps)
  rows=[json.loads((out/'records'/(r['name']+'.json')).read_text()) for r in refs['files']]
  audits=[json.loads((out/f'worker_{i}.json').read_text()) for i in range(len(gpus))]
  assert len(rows)==535 and all(z['status']=='complete' and z['teacher_parameter_sha_before']==z['teacher_parameter_sha_after'] for z in audits)
  for row in rows:assert sha256(out/row['path'])==row['SHA256']
  atomic_json(out/'manifest.json',dict(status='complete',mode=mode,files=rows,workers=audits,source_manifest_sha256=sha256(a.source/'manifest.json'),teacher_checkpoint_sha256=sha256(a.teacher),
   true_input_cache_manifest_sha256=sha256(a.base/'manifest.json'),teacher_query_scalars=sum(z['teacher_query_scalars'] for z in audits),
   teacher_Linear_forward_FLOPs=sum(z['teacher_Linear_forward_FLOPs'] for z in audits),optimizer_updates=0,encoder_forward_scalars=0,target_validation_or_locked_arrays_opened=False))
 atomic_json(root/'cache_progress.json',dict(status='complete',pid=os.getpid(),modes=['degree_only','wrong_degree'],teacher_query_scalars=39555520,optimizer_updates=0))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['root','base','source','benchmark','teacher']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--gpus',required=True);a=p.parse_args()
 try:main(a)
 except Exception:atomic_json(a.root/'cache_progress.json',dict(status='needs_attention',error=traceback.format_exc(),optimizer_updates=0));raise
