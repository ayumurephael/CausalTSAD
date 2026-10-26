"""Adopt healthy original worker and finish only a cache-identical pending run."""
import os,sys,json,time,subprocess,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256

def main(a):
    plan=json.loads(a.plan.read_text());r=a.root
    with (r/'recovery/controller.lock').open('x') as f:f.write(str(os.getpid()))
    procs={};logs={}
    for ident,gpu in plan['recovery_gpus'].items():
        i=next(i for i,t in enumerate(plan['tasks']) if t['id']==ident)
        assert not (r/'records'/(ident+'.json')).exists()
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        log=(r/'recovery'/f'{ident}.log').open('x')
        p=subprocess.Popen([sys.executable,plan['code']+'/tools/run_i09_mixture.py','--root',str(r),'--plan',str(a.plan),'--task',str(i)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
        procs[ident]=p;logs[ident]=log
    while True:
        done=[t['id'] for t in plan['tasks'] if (r/'records'/(t['id']+'.json')).exists()]
        errors=list((r/'errors').glob('*.json'));deadline=time.time()>plan['deadline_unix']
        if errors or deadline:
            for p in procs.values():
                if p.poll() is None:p.terminate()
            atomic_json(r/'progress.json',dict(status='needs_attention',completed=len(done),total=len(plan['tasks']),errors=[p.name for p in errors],deadline_passed=deadline,controller='cache_reuse'));return
        for ident,p in procs.items():
            if p.poll() not in [None,0] and ident not in done:atomic_json(r/'errors'/f'recovery_{ident}.json',dict(returncode=p.returncode))
        for ident,pid in plan['adopted_workers'].items():
            if ident in done:continue
            p=Path('/proc')/str(pid)/'cmdline'
            if not p.exists() or str(r) not in p.read_bytes().decode(errors='replace'):
                atomic_json(r/'errors'/f'adopted_{ident}.json',dict(error='Original worker exited without final record'))
        if len(done)==len(plan['tasks']):break
        atomic_json(r/'progress.json',dict(status='running',completed=len(done),total=len(plan['tasks']),controller='cache_reuse',
            active={k:dict(pid=v.pid) for k,v in procs.items() if k not in done},adopted={k:v for k,v in plan['adopted_workers'].items() if k not in done},pending=0))
        time.sleep(5)
    for p in procs.values():p.wait(timeout=30)
    for f in logs.values():f.close()
    files=[dict(path=str(p.relative_to(r)),SHA256=sha256(p),bytes=p.stat().st_size) for p in sorted((r/'records').iterdir())]
    atomic_json(r/'manifest.json',dict(status='complete',tasks=len(done),files=files,plan_SHA256=sha256(r/'job_plan.json'),
        recovery_plan_SHA256=sha256(a.plan),reused_frozen_cache=True))
    atomic_json(r/'progress.json',dict(status='complete',completed=len(done),total=len(plan['tasks']),controller='cache_reuse'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);main(p.parse_args())
