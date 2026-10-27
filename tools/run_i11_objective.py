"""Fixed 512 source-pair updates; no basis work or target/global adaptation."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,argparse,traceback,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from causal_tsad.io import sha256,atomic_json
from causal_tsad.frozen_detector import parameter_sha
from proposals.context_mixture import ContextSelector
from proposals.objective_alignment import predict,objective
from proposals.mixture_evidence_audit import normal_target_nll
from tools.run_i06_controls import metrics

def setup(plan,task):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(plan['worker_GiB']*2**30/torch.cuda.get_device_properties(0).total_memory)
    prior=Path(plan['prior']);donor=task['donor'];rp=prior/'records'/(donor+'.json')
    assert sha256(rp)==task['donor_record_SHA256'];rec=json.loads(rp.read_text());assert rec['status']=='complete'
    cm=prior/'cache'/donor/'manifest.json';assert sha256(cm)==rec['cache_manifest_SHA256']
    cache=json.loads(cm.read_text());assert len(cache['files'])==1088 and cache['basis_SHA256']==rec['basis_parameter_SHA256']
    wp=Path(plan['weight']);assert sha256(wp)==plan['weight_SHA256'];state=torch.load(wp,map_location='cpu',weights_only=False)['model']
    temp=float(state['log_temperature'].exp().clamp(.1,10));bias=float(state['bias'])
    assert sha256(Path(plan['source_eval'])/'manifest.json')==plan['eval_manifest_SHA256']
    assert sha256(Path(plan['source_train'])/'manifest.json')==plan['train_manifest_SHA256']
    selector=ContextSelector(task['seed']).cuda().double()
    return selector,cache,rec,temp,bias

def load(item):
    p=Path(item['path']);assert sha256(p)==item['SHA256']
    with np.load(p) as f:
        # Explicit whitelist. Supervision is removed before predict() is called.
        z={k:torch.as_tensor(f[k],device='cuda') for k in ['response','mu','variance','features','valid']}
        target=torch.as_tensor(f['training_response'],device='cuda') if 'training_response' in f else None
    assert all(not v.requires_grad for v in z.values())
    return z,target

def pairs_and_eval(cache):
    train={};ev=[]
    for r in cache['files']:
        if 'split' not in r['row']:train.setdefault(r['key'],{})[r['view']]=r
        else:ev.append(r)
    pairs=[]
    for key,views in sorted(train.items()):
        assert len(views)==8
        for normal,fault in [(0,2),(1,3),(4,6),(5,7)]:pairs.append((views[normal],views[fault]))
    assert len(pairs)==256 and len(ev)==576
    return pairs,ev

def preflight(plan,output):
    task=plan['tasks'][0];sel,cache,rec,temp,bias=setup(plan,task);pairs,ev=pairs_and_eval(cache)
    n,nt=load(pairs[0][0]);f,ft=load(pairs[0][1]);row=pairs[0][0]['row'];before=parameter_sha(sel)
    with torch.no_grad():
        p=predict(sel,f,temp,bias)
        assert torch.allclose(p['lw'].exp(),torch.full_like(p['lw'],.2),atol=1e-14,rtol=0)
        uniform_error=float((p['mean']-((f['response'][...,None]-f['mu']).square()/(3*f['variance'])-.5*math.log(3)).mean(-1)).abs().max())
        perm=torch.tensor([4,2,0,3,1],device='cuda');ff=dict(f)
        for key in ['mu','variance','valid']:ff[key]=f[key][...,perm]
        ff['features']=f['features'][...,perm,:];pp=predict(sel,ff,temp,bias)
        permutation_error=max(float((p[k]-pp[k]).abs().max()) for k in ['mean','density'])
    grads={}
    for kind in ['all_nll','root_nll','relation_rank']:
        sel.zero_grad();loss,_=objective(sel,n,f,nt,ft,row['root'],row['fork'],kind,temp,bias);loss.backward()
        grads[kind]=sum(float(v.grad.abs().sum()) for v in sel.parameters() if v.grad is not None)
        assert torch.isfinite(loss) and grads[kind]>0
    assert parameter_sha(sel)==before and uniform_error<1e-8 and permutation_error<1e-8
    # Root/onset/normal targets never enter predict, not merely detached.
    assert set(f)=={'response','mu','variance','features','valid'}
    atomic_json(output,dict(status='passed',uniform_evidence_error=uniform_error,candidate_permutation_error=permutation_error,
       selector_parameters=sum(p.numel() for p in sel.parameters()),gradient_L1=grads,initial_SHA256=before,
       optimizer_updates=0,technical_backward_calls=3,unique_cache_reads=2,new_basis_forwards=0,new_context_factorizations=0))

def run(a):
    start=time.time();plan=json.loads(a.plan.read_text());task=plan['tasks'][a.task];ident=task['id']
    sel,cache,rec,temp,bias=setup(plan,task);pairs,ev=pairs_and_eval(cache);initial=parameter_sha(sel)
    curves=[];exposure=target_exposure=0;factor_sha=rec['basis_parameter_SHA256']
    def progress(stage,**kw):atomic_json(a.root/'task_progress'/(ident+'.json'),dict(stage=stage,seconds=time.time()-start,**kw))
    def evaluate(step):
        arrays={};rca={};diag=[]
        with torch.no_grad():
            for item in ev:
                if time.time()>plan['deadline_unix']:raise TimeoutError('I11 fixed deadline')
                z,target=load(item);assert target is None;p=predict(sel,z,temp,bias)
                for method in ['mean','density','full']:
                    key=method+'__'+item['key'];arrays.setdefault(key,{})[item['view']]=p['score_'+method].cpu().numpy()
                    if item['row']['split']=='source_validation' and item['view']%8>=2:
                        v=item['view']%8;lo,hi=(0,704) if v in [2,3] else (192,193) if v in [4,5] else (192,320)
                        rank=int(torch.where(torch.argsort(torch.softmax(p[method][lo:hi],-1).mean(0),descending=True,stable=True)==item['row']['root'])[0])+1
                        rca.setdefault(key,[]).append(dict(view=item['view'],Top1=int(rank==1),Top3=int(rank<=3),MRR=1/rank))
                diag.append(dict(key=item['key'],view=item['view'],weights=p['lw'].exp().mean((0,1)).cpu().tolist()))
        arrays={k:np.stack([v[i] for i in sorted(v)]) for k,v in arrays.items()};met={}
        for row in {v['key']:v['row'] for v in ev}.values():
            src=Path(plan['source_eval'])/row['path'];assert sha256(src)==row['SHA256']
            with np.load(src) as f:labels=f['labels']
            for method in ['mean','density','full']:
                key=method+'__'+row['key'];met[key]=metrics(arrays[key],labels,row)
        prefix=a.root/'records'/f'{ident}_{step}';np.savez_compressed(str(prefix)+'.npz',**arrays)
        atomic_json(str(prefix)+'.json',dict(task=task,step=step,metrics=met,conditional_RCA=rca,selector_diagnostics=diag,
             scores_SHA256=sha256(str(prefix)+'.npz'),optimizer_exposure=exposure,normal_target_exposure=target_exposure,
             basis_parameter_SHA256=factor_sha))
    evaluate(0);opt=torch.optim.Adam(sel.parameters(),lr=.001)
    rng=np.random.default_rng(task['seed']);order=np.concatenate([rng.permutation(256) for _ in range(2)])
    for step,index in enumerate(order,1):
        if time.time()>plan['deadline_unix']:raise TimeoutError('I11 finite 4h budget, no extension')
        ni,fi=pairs[int(index)];n,nt=load(ni);f,ft=load(fi);row=ni['row']
        loss,diag=objective(sel,n,f,nt,ft,row['root'],row['fork'],task['objective'],temp,bias)
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(sel.parameters(),1.);opt.step()
        assert all(v.grad is None for v in list(n.values())+list(f.values()))
        exposure+=2*1024*row['C'];target_exposure+=2*1024*row['C']
        curves.append(dict(step=step,loss=float(loss.detach()),**diag,gradient_norm=float(norm),key=row['key'],normal_view=ni['view'],
                           fault_view=fi['view'],observed_exposure=exposure))
        if step%32==0:progress('training',step=step,observed_exposure=exposure)
    assert exposure==target_exposure==14680064
    evaluate(512)
    # Fixed-pool audit after training, independent of minibatch order.
    nll_records=[]
    with torch.no_grad():
        for item in [r for r in cache['files'] if 'split' not in r['row']]:
            if time.time()>plan['deadline_unix']:raise TimeoutError('I11 fixed deadline')
            z,target=load(item);audit=dict(z,training_response=target)
            nll_records.append(dict(key=item['key'],view=item['view'],C=item['C'],row=item['row'],**normal_target_nll(audit,sel)))
    atomic_json(a.root/'records'/(ident+'_fixed_nll.json'),nll_records)
    checkpoint=a.root/'records'/(ident+'_checkpoint.pth')
    torch.save(dict(selector=sel.state_dict(),task=task,steps=512,observed_exposure=exposure,normal_target_exposure=target_exposure,basis_SHA256=factor_sha),checkpoint)
    atomic_json(a.root/'records'/(ident+'_curves.json'),curves)
    assert sha256(Path(plan['weight']))==plan['weight_SHA256']
    atomic_json(a.root/'records'/(ident+'.json'),dict(status='complete',task=task,updates=512,observed_exposure=exposure,normal_target_exposure=target_exposure,
        selector_initial_SHA256=initial,selector_final_SHA256=parameter_sha(sel),checkpoint_SHA256=sha256(checkpoint),selector_parameters=sum(p.numel() for p in sel.parameters()),
        basis_parameter_SHA256=factor_sha,basis_unchanged=True,weight_SHA256=plan['weight_SHA256'],donor_record_SHA256=task['donor_record_SHA256'],
        cache_manifest_SHA256=rec['cache_manifest_SHA256'],source_train_manifest_SHA256=plan['train_manifest_SHA256'],source_eval_manifest_SHA256=plan['eval_manifest_SHA256'],
        order_SHA256=sha256(a.plan),paired_schedule_SHA256=__import__('hashlib').sha256(order.tobytes()).hexdigest(),
        budget=dict(training_cache_reads=1024,evaluation_cache_reads=1152,fixed_NLL_cache_reads=512,evaluation_cached_input_scalars=16515072,
            fixed_NLL_cached_input_scalars=7340032,fixed_NLL_target_query_scalars=5046272,
            train_query_node_points=10092544,eval_query_node_points=11354112,component_evidence_outputs_per_readout=5,
            new_basis_forwards=0,new_context_factorizations=0),
        benchmark_arrays_opened=False,normal_twin_in_selector_input=False,root_in_selector_input=False,TimeRCD_RCA=False,
        final_step_selected=512,seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),code=plan['code']))
    progress('complete',step=512)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--plan',type=Path,required=True);p.add_argument('--task',type=int);p.add_argument('--preflight',type=Path);a=p.parse_args()
    if a.preflight:preflight(json.loads(a.plan.read_text()),a.preflight)
    else:
        try:run(a)
        except Exception:atomic_json(a.root/'errors'/f'task{a.task}.json',dict(error=traceback.format_exc()));raise
