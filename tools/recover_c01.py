"""Resume only missing C01 work, preserving old attempts and numerical code."""
import argparse,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from causal_tsad.io import atomic_json,sha256


def main(a):
    parent=a.parent;root=a.output;code=Path(__file__).resolve().parents[1]
    assert not root.exists(),'Recovery attempts are immutable'
    assert json.loads((parent/'controller_progress.json').read_text())['status']=='needs_attention'
    config=code/'experiments/c01_pilot_v1.json';cfg=json.loads(config.read_text())
    plan=json.loads(a.plan.read_text());old_code=Path(plan['old_code'])
    names=['proposals/c01/model.py','proposals/c01/data.py','tools/train_c01.py','tools/evaluate_c01.py','tools/cache_c01.py','experiments/c01_pilot_v1.json']
    unchanged={n:sha256(code/n) for n in names}
    assert unchanged=={n:sha256(old_code/n) for n in names},'Recovery must not change algorithm or budget'
    root.mkdir();(root/'logs').mkdir();(root/'runs').mkdir()
    with (root/'controller.lock').open('x') as f:f.write(str(os.getpid()))
    cache=parent/'cache';manifest=json.loads((cache/'manifest.json').read_text())
    assert manifest['status']=='complete' and len(manifest['files'])==1901
    assert manifest['source_manifest_sha256']==cfg['source_manifest_sha256']
    for r in manifest['files']:assert sha256(cache/r['path'])==r['SHA256']
    assert all(w['frozen_parameter_digests_before']==w['frozen_parameter_digests_after'] for w in manifest['workers'])
    jobs=[];reused=[];failed=[]
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            name=f'{arm}_s{seed}';old=parent/'runs'/name;dest=root/'runs'/name
            status=json.loads((old/'progress.json').read_text()) if (old/'progress.json').exists() else None
            if status and status['status']=='training_complete':
                assert status['updates']==status['total_updates']==434
                assert status['optimized_original_observed_scalars']==4976640
                assert sha256(old/'final.pth')==status['checkpoint_sha256']
                dest.mkdir()
                for filename in ['final.pth','progress.json','initial_source_validation.json','final_source_validation.json','learning_curve.jsonl']:
                    shutil.copy2(old/filename,dest/filename)
                    assert sha256(old/filename)==sha256(dest/filename)
                atomic_json(dest/'recovery_provenance.json',dict(reused_from=str(old),optimizer_repeated=False,
                    checkpoint_sha256=status['checkpoint_sha256'],original_training_commit=status['local_commit']))
                reused.append(name);stage='evaluate'
            else:
                if status:
                    assert status['updates']==0 and status['optimized_original_observed_scalars']==0,'Partial optimization needs checkpoint-aware recovery'
                    failed.append(dict(name=name,optimizer_steps=0,source_validation_partial_forward_exact_count=None,
                        source_validation_query_scalar_upper_bound=2*1968128,attempt_preserved=str(old)))
                stage='train'
            jobs.append(dict(name=name,arm=arm,seed=seed,stage=stage))
    assert len(reused)==4 and sum(j['stage']=='train' for j in jobs)==8
    atomic_json(root/'recovery_audit.json',dict(status='passed',unchanged_algorithm_file_sha256=unchanged,
        reused_training_runs=reused,new_training_runs=8,failed_zero_step_attempts=failed,
        cache_files_verified=1901,cache_manifest_sha256=sha256(cache/'manifest.json'),
        new_encoder_or_cache_computation=0,optimization_budget_not_increased=True,
        planned_new_optimized_original_scalars=8*4976640,planned_total_original_scalars_including_reused=12*4976640,
        original_preflight=json.loads((parent/'preflight.json').read_text()),plan=plan))
    pending=list(jobs);active={};complete=[];started=time.monotonic()
    status=dict(status='running',stage='recover_train_and_evaluate',pid=os.getpid(),reused_training_runs=reused,
        pending=len(pending),active=[],completed_runs=[])
    def launch(gpu,job,stage):
        dest=root/'runs'/job['name'];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTHONUNBUFFERED']='1'
        if stage=='train':
            cmd=[code/'tools/train_c01.py','--cache',cache,'--output',dest,'--config',config,'--arm',job['arm'],'--seed',job['seed'],
                '--local-commit',plan['local_commit'],'--github-commit',plan['github_commit']]
        else:cmd=[code/'tools/evaluate_c01.py','--cache',cache,'--run',dest,'--benchmark',a.benchmark,'--metric-code',a.metric_code]
        with (root/'logs'/(job['name']+'_'+stage+'.log')).open('x') as f:
            p=subprocess.Popen([sys.executable]+list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        active[gpu]=dict(job=job,stage=stage,process=p,started=time.monotonic())
    while pending or active:
        if time.monotonic()-started>cfg['max_campaign_hours']*3600:raise TimeoutError('Bounded technical recovery deadline')
        free_raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode()
        free={int(s.split(',')[0]):float(s.split(',')[1]) for s in free_raw.strip().splitlines()}
        # Select the pending job once; no stale stage may be inherited.
        for gpu in plan['gpus']:
            if gpu not in active and pending and free.get(gpu,0)>=12288:
                job=pending.pop(0);launch(gpu,job,job['stage'])
        status.update(active=[dict(gpu=g,run=e['job']['name'],stage=e['stage'],pid=e['process'].pid) for g,e in active.items()],
            pending=len(pending),completed_runs=complete,elapsed_seconds=time.monotonic()-started)
        atomic_json(root/'controller_progress.json',status);time.sleep(5)
        for gpu,e in list(active.items()):
            result=e['process'].poll()
            if result is None:
                if time.monotonic()-e['started']>cfg['max_stage_hours']*3600:raise TimeoutError('Recovery worker deadline')
                continue
            if result!=0:raise RuntimeError(f"{e['job']['name']} {e['stage']} failed ({result}), new jobs stopped")
            if e['stage']=='train':launch(gpu,e['job'],'evaluate')
            else:complete.append(e['job']['name']);del active[gpu]
    status.update(status='complete',stage='complete',active=[],pending=0,completed_runs=complete)
    atomic_json(root/'controller_progress.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--metric-code',type=Path,required=True)
    a=p.parse_args()
    try:main(a)
    except Exception:
        dest=a.output/'controller_progress.json';s=json.loads(dest.read_text()) if dest.exists() else {}
        s.update(status='needs_attention',error=traceback.format_exc(),new_jobs_stopped=True);atomic_json(dest,s);raise
