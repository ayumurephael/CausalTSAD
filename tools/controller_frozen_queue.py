"""Finite immutable GPU queue; only owns its own Popen children."""
import os,sys,json,time,argparse,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256

def main(a):
    r=a.root;p=json.loads(a.plan.read_text());plan_sha=sha256(a.plan)
    (r/'controller.lock').open('x').close();active={};done=[];pending=list(p['jobs']);start=time.time()
    def write(status):
        atomic_json(r/'progress.json',dict(status=status,completed=len(done),total=len(p['jobs']),pending=len(pending),
          active=[dict(id=x['job']['id'],gpu=g,pid=x['proc'].pid) for g,x in active.items()],seconds=time.time()-start,plan_SHA256=plan_sha))
    try:
        while pending or active:
            if time.time()>p['deadline_unix']:raise TimeoutError('finite deadline reached')
            if (r/'STOP_NEW').exists():raise RuntimeError('STOP_NEW requested for this queue')
            for gpu,x in list(active.items()):
                rc=x['proc'].poll()
                if rc is None:continue
                x['log'].close();job=x['job'];output=Path(job['output'])
                if rc or not output.exists():raise RuntimeError(f"{job['id']} failed rc={rc}; inspect its log")
                rec=json.loads(output.read_text());assert rec['status']=='complete'
                done.append(dict(id=job['id'],path=str(output),SHA256=sha256(output)));del active[gpu]
                atomic_json(r/'completed.json',done)
            free={int(s.split(',')[0]):float(s.split(',')[1]) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode().strip().splitlines()}
            for gpu in p['gpus']:
                if not pending:break
                if gpu in active or free.get(gpu,0)<p['required_free_MiB']:continue
                job=pending.pop(0);assert not Path(job['output']).exists()
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);log=(r/'logs'/(job['id']+'.log')).open('x')
                proc=subprocess.Popen(job['argv'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
                active[gpu]=dict(proc=proc,job=job,log=log)
            write('running');time.sleep(3)
        atomic_json(r/'inference_manifest.json',dict(status='complete',rows=done,plan_SHA256=plan_sha))
        if p.get('postprocess'):
            write('metrics_running')
            with (r/'logs/metrics.log').open('x') as f:
                q=subprocess.Popen(p['postprocess'],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                post_start=time.time()
                while q.poll() is None:
                    if time.time()-post_start>p['metrics_seconds']:q.terminate();raise TimeoutError('finite metric deadline')
                    time.sleep(5)
                assert q.returncode==0,'Original four metric worker failed'
        write('complete');atomic_json(r/'manifest.json',dict(status='complete',rows=done,plan_SHA256=plan_sha))
    except Exception:
        for x in active.values():
            if x['proc'].poll() is None:x['proc'].terminate()
        atomic_json(r/'failure.json',dict(status='needs_attention',error=traceback.format_exc()));write('needs_attention');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);main(p.parse_args())
