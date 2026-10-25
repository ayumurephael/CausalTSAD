"""Finite cache -> audit -> train/evaluate queue. Only owns its own children."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256
from tools.cache_c01 import make_tasks


def main(a):
    root=a.root;root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('x') as f:f.write(str(os.getpid()))
    plan=json.loads(a.plan.read_text());code=Path(__file__).resolve().parents[1]
    config=code/'experiments/c01_pilot_v1.json';cfg=json.loads(config.read_text());started=time.monotonic()
    cache=root/'cache';cache.mkdir(exist_ok=False);logs=root/'logs';logs.mkdir(exist_ok=True)
    status=dict(status='running',stage='preflight',pid=os.getpid(),active=[],completed_runs=[],pending=12)
    atomic_json(root/'controller_progress.json',status)
    gpus=plan['gpus'];all_children=[]
    def launch(cmd,name,gpu):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTHONUNBUFFERED']='1'
        with (logs/(name+'.log')).open('x') as f:
            process=subprocess.Popen([sys.executable]+list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        all_children.append(process);return process
    def wait_stage(active,limit):
        beginning=time.monotonic()
        while any(p.poll() is None for p in active):
            if any(p.poll() not in [None,0] for p in active):raise RuntimeError('One project worker failed; no new stage will start')
            if time.monotonic()-beginning>limit*3600:raise TimeoutError('Fixed technical stage deadline')
            if time.monotonic()-started>cfg['max_campaign_hours']*3600:raise TimeoutError('Fixed campaign deadline')
            time.sleep(5)
        assert all(p.returncode==0 for p in active),'Failed worker; preserve logs and completed work'
    common=['--source',a.source,'--benchmark',a.benchmark,'--original-code',a.original_code,'--checkpoint',a.checkpoint,'--config',config]
    pre=launch([code/'tests/preflight_c01.py']+common+['--output',root/'preflight.json'],'preflight',gpus[0])
    wait_stage([pre],1);assert json.loads((root/'preflight.json').read_text())['status']=='passed'
    tasks=make_tasks(a.source,a.benchmark);atomic_json(root/'cache_tasks.json',tasks)
    status.update(stage='cache',cache_tasks=len(tasks));atomic_json(root/'controller_progress.json',status)
    workers=[]
    for i,gpu in enumerate(gpus):
        workers.append(launch([code/'tools/cache_c01.py']+common+['--output',cache,'--tasks',root/'cache_tasks.json','--worker',i,'--workers',len(gpus)],'cache_'+str(i),gpu))
    wait_stage(workers,cfg['max_stage_hours'])
    rows=[];totals={}
    for task in tasks:
        r=json.loads((cache/'records'/(task['name']+'.json')).read_text());assert sha256(cache/r['path'])==r['SHA256']
        rows.append(r)
        for view,count in r['counts'].items():
            for k,v in count.items():
                key=r['kind']+'/'+view+'/'+k;totals[key]=totals.get(key,0)+v
    worker_reports=[json.loads((cache/f'worker_{i}.json').read_text()) for i in range(len(gpus))]
    assert all(r['status']=='complete' and r['frozen_parameter_digests_before']==r['frozen_parameter_digests_after'] for r in worker_reports)
    source=json.loads((a.source/'manifest.json').read_text())
    assert sum(r['scalars'] for r in rows if r['kind']=='train')==source['training_original_observed_scalars']
    assert sum(r['scalars'] for r in rows if r['kind']=='source_validation')==source['source_validation_original_observed_scalars']
    assert sum(r['kind']=='development' for r in rows)==100
    atomic_json(cache/'manifest.json',dict(status='complete',files=rows,counts=totals,workers=worker_reports,
        source_manifest_sha256=sha256(a.source/'manifest.json'),config_sha256=sha256(config),
        feature_dim=json.loads((root/'preflight.json').read_text())['feature_dim'],
        original_checkpoint_sha256=sha256(a.checkpoint),code_local_commit=plan['local_commit'],code_github_commit=plan['github_commit'],
        target_validation_or_locked_arrays_opened=False,optimizer_updates=0,
        cache_compute_note='Encoder history, overlap, normal twins and original detector counted separately; shared once across seeds/heads. Raw features have lower native encoder cost; not a strict end-to-end FLOPs match.'))
    pending=[dict(arm=arm,seed=seed,name=f'{arm}_s{seed}') for seed in cfg['seeds'] for arm in cfg['arms']]
    active={};complete=[]
    def start_job(gpu,job,stage):
        dest=root/'runs'/job['name']
        if stage=='train':
            command=[code/'tools/train_c01.py','--cache',cache,'--output',dest,'--config',config,'--arm',job['arm'],'--seed',job['seed'],
                '--local-commit',plan['local_commit'],'--github-commit',plan['github_commit']]
        else:command=[code/'tools/evaluate_c01.py','--cache',cache,'--run',dest,'--benchmark',a.benchmark,'--metric-code',a.metric_code]
        process=launch(command,job['name']+'_'+stage,gpu)
        active[gpu]=dict(job=job,stage=stage,process=process,started=time.monotonic())
    while pending or active:
        if time.monotonic()-started>cfg['max_campaign_hours']*3600:raise TimeoutError('Fixed campaign deadline')
        for gpu in gpus:
            if gpu not in active and pending:start_job(gpu,pending.pop(0),'train')
        status.update(stage='train_and_evaluate',active=[dict(gpu=g,pid=e['process'].pid,stage=e['stage'],run=e['job']['name']) for g,e in active.items()],pending=len(pending),completed_runs=complete)
        atomic_json(root/'controller_progress.json',status);time.sleep(5)
        for gpu,e in list(active.items()):
            result=e['process'].poll()
            if result is None:
                if time.monotonic()-e['started']>cfg['max_stage_hours']*3600:raise TimeoutError('Worker stage deadline')
                continue
            if result!=0:raise RuntimeError(f"{e['job']['name']} {e['stage']} failed: {result}; new jobs stopped")
            if e['stage']=='train':start_job(gpu,e['job'],'evaluate')
            else:complete.append(e['job']);del active[gpu]
    status.update(status='complete',stage='complete',active=[],pending=0,completed_runs=complete,seconds=time.monotonic()-started)
    atomic_json(root/'controller_progress.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--original-code',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--metric-code',type=Path,required=True);a=p.parse_args()
    try:main(a)
    except Exception:
        prior=a.root/'controller_progress.json';s=json.loads(prior.read_text()) if prior.exists() else {}
        s.update(status='needs_attention',error=traceback.format_exc(),new_jobs_stopped=True)
        # Already running children can finish; no other user's task is touched.
        atomic_json(prior,s);raise
