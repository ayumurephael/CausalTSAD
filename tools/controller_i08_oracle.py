"""Bounded GPU queue. Stop this project's queue on any error, never other jobs."""
import sys,os,json,time,subprocess,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256
def main(a):
    plan=json.loads(a.plan.read_text());lock=a.root/'controller.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    done=[]
    for i,task in enumerate(plan['tasks']):
        p=a.root/'records'/(task['id']+'.json')
        if p.exists():
            old=json.loads(p.read_text());assert old['task']==task
            assert sha256(p.with_suffix('.npz'))==old['scores_SHA256']
            assert sha256(a.root/'node_cache'/(task['id']+'.npz'))==old['node_cache_SHA256']
            done.append(i)
    reused=len(done);todo=[i for i in range(len(plan['tasks'])) if i not in done];active={}
    while todo or active:
        failed=list((a.root/'errors').glob('*.json'))
        if failed or (a.root/'STOP_NEW').exists() or time.time()>plan['deadline_unix']:
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
            fits=[i for i in todo if free.get(gpu,0)>=plan['required_free_MiB']]
            if gpu not in active and fits:
                i=fits[0];todo.remove(i);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
                f=(a.root/'logs'/f'task{i}.log').open('x')
                p=subprocess.Popen([sys.executable,plan['code']+'/tools/run_i08_oracle.py','--root',str(a.root),'--plan',str(a.plan),'--task',str(i)],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                active[gpu]=(p,f,i)
        atomic_json(a.root/'progress.json',dict(status='running',completed=len(done),reused_verified=reused,total=len(plan['tasks']),active={str(g):dict(pid=p.pid,task=i) for g,(p,f,i) in active.items()},pending=len(todo)))
        time.sleep(5)
    files=[]
    for p in sorted((a.root/'records').iterdir()):files.append(dict(path=str(p.relative_to(a.root)),SHA256=sha256(p),bytes=p.stat().st_size))
    atomic_json(a.root/'manifest.json',dict(status='complete',tasks=len(done),reused_verified=reused,files=files,plan_SHA256=sha256(a.plan)))
    atomic_json(a.root/'progress.json',dict(status='complete',completed=len(done),total=len(plan['tasks'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);main(p.parse_args())
