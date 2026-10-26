"""Bounded GPU queue. Stop this project's queue on any error, never other jobs."""
import sys,os,json,time,subprocess,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256
def main(a):
    plan=json.loads(a.plan.read_text());lock=a.root/'controller.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    todo=list(range(len(plan['tasks'])));active={};done=[]
    while todo or active:
        failed=list((a.root/'errors').glob('*.json'))
        if failed or time.time()>plan['deadline_unix']:
            # Only children launched by this controller are terminated.
            for p,f,i in active.values():p.terminate();f.close()
            atomic_json(a.root/'progress.json',dict(status='needs_attention',completed=len(done),total=len(plan['tasks']),errors=[p.name for p in failed],deadline_passed=time.time()>plan['deadline_unix']));return
        for gpu in list(active):
            p,f,i=active[gpu]
            if p.poll() is not None:
                f.close();del active[gpu]
                if p.returncode:atomic_json(a.root/'errors'/f'exit{i}.json',dict(returncode=p.returncode,task=i));continue
                done.append(i)
        free={int(l.split(',')[0]):float(l.split(',')[1]) for l in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode().strip().splitlines()}
        for gpu in plan['gpus']:
            if gpu not in active and todo and free.get(gpu,0)>=10000:
                i=todo.pop(0);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
                f=(a.root/'logs'/f'task{i}.log').open('x')
                p=subprocess.Popen([sys.executable,plan['code']+'/tools/run_improvement_tier1.py','--root',str(a.root),'--plan',str(a.plan),'--task',str(i)],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                active[gpu]=(p,f,i)
        atomic_json(a.root/'progress.json',dict(status='running',completed=len(done),total=len(plan['tasks']),active={str(g):dict(pid=p.pid,task=i) for g,(p,f,i) in active.items()},pending=len(todo)))
        time.sleep(5)
    files=[]
    for p in sorted((a.root/'records').iterdir()):files.append(dict(path=str(p.relative_to(a.root)),SHA256=sha256(p),bytes=p.stat().st_size))
    atomic_json(a.root/'manifest.json',dict(status='complete',tasks=len(done),files=files,plan_SHA256=sha256(a.plan)))
    atomic_json(a.root/'progress.json',dict(status='metrics_running',completed=len(done),total=len(plan['tasks'])))
    with (a.root/'logs/metrics_controller.log').open('x') as f:
        p=subprocess.run([sys.executable,plan['code']+'/tools/metrics_improvement_tier1.py','--root',str(a.root),'--plan',str(a.plan)],stdout=f,stderr=subprocess.STDOUT,timeout=28800)
    atomic_json(a.root/'progress.json',dict(status='complete' if p.returncode==0 else 'needs_attention',completed=len(done),total=len(plan['tasks']),metrics_returncode=p.returncode))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);main(p.parse_args())
