"""Finite dependency-aware queue; stop only this controller's own children.

Cache/frozen jobs use core snapshot; learned jobs use their proposal's immutable
branch snapshot. No polling of, or intervention in, other users' jobs.
"""
import sys,os,json,time,argparse,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256

def main(a):
    p=json.loads(a.plan.read_text());(a.root/'controller.lock').open('x').write(str(os.getpid()))
    active={};done=set();jobs=[]
    for i,t in enumerate(p['cache_tasks']):jobs.append(dict(kind='cache',index=i,id=t['id'],profile=t['profile']))
    for i,t in enumerate(p['train_tasks']):jobs.append(dict(kind='train',index=i,id=t['id'],profile=t['profile']))
    for profile in p['profiles']:jobs.append(dict(kind='frozen',id='frozen_'+profile,profile=profile))
    # New root only; never silently resume an incompatible/partial stage.
    for j in jobs:assert not (a.root/'records'/(j['id']+'.json')).exists()
    while len(done)<len(jobs):
        if list((a.root/'errors').glob('*.json')) or (a.root/'STOP_NEW').exists() or time.time()>p['deadline_unix']:
            for child,log,j in active.values():child.terminate();log.close()
            atomic_json(a.root/'progress.json',dict(status='needs_attention',completed=len(done),total=len(jobs),deadline_passed=time.time()>p['deadline_unix']));return
        for gpu in list(active):
            child,log,j=active[gpu]
            if child.poll() is not None:
                log.close();del active[gpu]
                if child.returncode:
                    atomic_json(a.root/'errors'/('exit_'+j['id']+'.json'),dict(returncode=child.returncode,job=j))
                else:
                    r=json.loads((a.root/'records'/(j['id']+'.json')).read_text());assert r['status']=='complete';done.add(j['id'])
        if list((a.root/'errors').glob('*.json')):continue
        free={int(x.split(',')[0]):float(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode().strip().splitlines()}
        ready_profiles={profile for profile in p['profiles'] if all(t['id'] in done for t in p['cache_tasks'] if t['profile']==profile)}
        running={j['id'] for _,_,j in active.values()}
        available=[j for j in jobs if j['id'] not in done|running and (j['kind']=='cache' or j['profile'] in ready_profiles)]
        # Once its inputs are complete, prefer a scientific run over remaining
        # secondary-profile caches; still occupy all admissible GPU slots.
        available.sort(key=lambda j:(j['kind']=='cache',j['kind']=='train'))
        for gpu in p['gpus']:
            if gpu in active or free.get(gpu,0)<p['required_free_MiB'] or not available:continue
            j=available.pop(0);code=p['code'] if j['kind']!='train' else p['train_tasks'][j['index']]['code']
            cmd=[sys.executable,code+'/tools/run_improvement3.py',j['kind'],'--root',str(a.root),'--plan',str(a.plan)]
            cmd+=['--profile',j['profile']] if j['kind']=='frozen' else ['--task',str(j['index'])]
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
            log=(a.root/'logs'/(j['id']+'.log')).open('x')
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            active[gpu]=(child,log,j)
        atomic_json(a.root/'progress.json',dict(status='running',completed=len(done),total=len(jobs),
          active={str(g):dict(pid=c.pid,**j) for g,(c,_,j) in active.items()},ready_profiles=sorted(ready_profiles),
          cache_completed=sum(j['kind']=='cache' and j['id'] in done for j in jobs),
          train_completed=sum(j['kind']=='train' and j['id'] in done for j in jobs)))
        time.sleep(5)
    files=[dict(path=str(f.relative_to(a.root)),SHA256=sha256(f),bytes=f.stat().st_size) for f in sorted((a.root/'records').iterdir())]
    atomic_json(a.root/'manifest.json',dict(status='complete',files=files,plan_SHA256=sha256(a.plan)))
    atomic_json(a.root/'progress.json',dict(status='complete',completed=len(done),total=len(jobs)))

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--root',type=Path,required=True);q.add_argument('--plan',type=Path,required=True);main(q.parse_args())
