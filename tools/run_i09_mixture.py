"""Finite source-only selector learning. Components/basis never receive gradients."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,argparse,traceback,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from causal_tsad.io import atomic_json,sha256,array_sha
from causal_tsad.frozen_detector import parameter_sha
from proposals.causal_meta_basis import CausalMetaBasis
from proposals.context_mixture import components,ContextSelector,mixture_logpdf,node_evidence,aggregate
from tools.train_meta_basis import graph_for
from tools.run_i06_controls import metrics

def load_model(plan):
    p=Path(plan['weight']);assert sha256(p)==plan['weight_SHA256']
    m=CausalMetaBasis(348).cuda().eval().requires_grad_(False)
    m.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']);return m

def tensor_cache(path):
    with np.load(path) as z:return {k:torch.as_tensor(z[k],device='cuda') for k in z.files}

def run(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    plan=json.loads(a.plan.read_text());task=plan['tasks'][a.task];ident=task['id'];start=time.time()
    torch.cuda.set_per_process_memory_fraction(plan['worker_GiB']*2**30/torch.cuda.get_device_properties(0).total_memory)
    model=load_model(plan);initial=parameter_sha(model);temp=float(model.log_temperature.exp().clamp(.1,10));bias=float(model.bias)
    selector=ContextSelector(task['seed'],task['arm']=='true_hidden_parent').cuda().double();init=parameter_sha(selector)
    opt=torch.optim.Adam(selector.parameters(),lr=.001)
    cache=a.root/'cache'/ident;cache.mkdir(parents=True,exist_ok=False)
    train_manifest=Path(plan['source_train'])/'manifest.json';eval_manifest=Path(plan['source_eval'])/'manifest.json'
    assert sha256(train_manifest)==plan['train_manifest_SHA256'] and sha256(eval_manifest)==plan['eval_manifest_SHA256']
    train=json.loads(train_manifest.read_text())['rows'];ev=json.loads(eval_manifest.read_text())['rows'];budget={}
    files=[];train_cases=[];eval_cases=[]
    def progress(stage,**kw):atomic_json(a.root/'task_progress'/(ident+'.json'),dict(stage=stage,seconds=time.time()-start,**kw))
    for split,rows,source in [('train',train,plan['source_train']),('eval',ev,plan['source_eval'])]:
        for row in rows:
            p=Path(source)/row['path'];assert sha256(p)==row['SHA256']
            with np.load(p) as z:values=z['values'];graph=z['graph'];target=z['normal_response'] if split=='train' else None
            assert array_sha(values)==row['input_SHA256']
            graph=torch.as_tensor(graph_for(graph,row['key'],'wrong' if task['arm']=='wrong_all' else 'true'),device='cuda',dtype=torch.float64)
            for view,x in enumerate(values):
                if time.time()>plan['deadline_unix']:raise TimeoutError('Fixed campaign time limit; no budget extension')
                out=components(model,torch.as_tensor(x,device='cuda',dtype=torch.float64),graph)
                for k,v in out.pop('audit').items():budget[k]=budget.get(k,0)+v
                if split=='train':out['training_response']=(torch.as_tensor(target[view,320:],device='cuda')-out['center'])/out['scale']
                p=cache/f'{row["key"]}_{view}.npz'
                np.savez_compressed(p,**{k:v.cpu().numpy() for k,v in out.items()})
                record=dict(path=str(p),SHA256=sha256(p),key=row['key'],view=view,C=row['C'],row=row)
                files.append(record);(train_cases if split=='train' else eval_cases).append(record)
            progress('cache_'+split,completed_cases=len(files),budget=budget)
    assert len(train_cases)==512 and len(eval_cases)==576
    atomic_json(cache/'manifest.json',dict(files=files,budget=budget,basis_SHA256=initial,graph=task['arm'],input_manifest_SHA256=plan['train_manifest_SHA256']))
    curves=[];exposure=0;target_exposure=0
    def evaluate(items,methods,save_name=None):
        all_scores={};all_rca={};all_nll={};diagnostics=[]
        with torch.no_grad():
            for item in items:
                z=tensor_cache(item['path']);valid=z['valid'];lw=selector(z['features'],valid)
                uniform=valid.to(torch.float64)/valid.sum(-1,keepdim=True);full=torch.full_like(lw,-torch.inf);full[...,0]=0
                weights={'learned':lw,'uniform':uniform.log(),'full':full}
                for method in methods:
                    e=node_evidence(z['response'],z['mu'],z['variance'],weights[method]);s=aggregate(e,temp,bias)
                    k=method+'__'+item['key'];all_scores.setdefault(k,{})[item['view']]=s.cpu().numpy()
                    if item['row']['split']=='source_validation':
                        vi=item['view']%8
                        if vi in [2,3,4,5,6,7]:
                            lo,hi=(0,704) if vi in [2,3] else (192,193) if vi in [4,5] else (192,320)
                            rank=int(torch.where(torch.argsort(torch.softmax(e[lo:hi],-1).mean(0),descending=True,stable=True)==item['row']['root'])[0])+1
                            all_rca.setdefault(k,[]).append(dict(view=item['view'],Top1=int(rank==1),Top3=int(rank<=3),MRR=1/rank))
                diagnostics.append(dict(key=item['key'],view=item['view'],mean_weights=lw.exp().mean((0,1)).cpu().tolist(),entropy=float(-(lw.exp()*lw.nan_to_num()).sum(-1).mean())))
        arrays={k:np.stack([d[i] for i in sorted(d)]) for k,d in all_scores.items()}
        if save_name:
            p=a.root/'records'/save_name;np.savez_compressed(p.with_suffix('.npz'),**arrays)
            rows={r['key']:r for r in ev};met={}
            for key,s in arrays.items():
                method,system=key.split('__');row=rows[system]
                with np.load(Path(plan['source_eval'])/row['path']) as z:labels=z['labels']
                met[key]=metrics(s,labels,row)
            atomic_json(p.with_suffix('.json'),dict(task=task,metrics=met,conditional_RCA=all_rca,selector_diagnostics=diagnostics,scores_SHA256=sha256(p.with_suffix('.npz')),
                step=int(save_name.rsplit('_',1)[-1]),basis_parameter_SHA256=initial,optimizer_exposure=exposure,normal_target_exposure=target_exposure))
        return arrays
    # Full initial and final evaluation; intermediate curve uses the same
    # predeclared 2 systems per C (index0/1), never for checkpoint selection.
    evaluate(eval_cases,['learned','uniform','full'],ident+'_0')
    order=np.random.default_rng(task['seed']).permutation(len(train_cases))
    for step,idx in enumerate(order,1):
        if time.time()>plan['deadline_unix']:raise TimeoutError('Fixed campaign time limit')
        item=train_cases[int(idx)];assert sha256(Path(item['path']))==item['SHA256'];z=tensor_cache(item['path'])
        lw=selector(z['features'],z['valid']);loss=-mixture_logpdf(z['training_response'],z['mu'],z['variance'],lw).mean()
        assert torch.isfinite(loss);opt.zero_grad();loss.backward()
        assert all(p.grad is None for p in model.parameters())
        norm=torch.nn.utils.clip_grad_norm_(selector.parameters(),1.);opt.step()
        exposure+=1024*item['C'];target_exposure+=1024*item['C']
        curves.append(dict(step=step,NLL=float(loss.detach()),gradient_norm=float(norm),key=item['key'],view=item['view'],observed_exposure=exposure))
        if step in [128,256]:
            subset=[r for r in eval_cases if r['row']['split']=='source_validation' and int(r['key'])%10000<2]
            evaluate(subset,['learned'],ident+'_'+str(step))
        if step%16==0:progress('training',step=step,observed_exposure=exposure)
    assert exposure==7340032 and target_exposure==7340032 and parameter_sha(model)==initial
    evaluate(eval_cases,['learned'],ident+'_512')
    p=a.root/'records'/(ident+'_checkpoint.pth');torch.save(dict(selector=selector.state_dict(),task=task,steps=512,observed_exposure=exposure,basis_SHA256=initial),p)
    atomic_json(a.root/'records'/(ident+'_curves.json'),curves)
    atomic_json(a.root/'records'/(ident+'.json'),dict(task=task,status='complete',basis_parameter_SHA256=initial,basis_unchanged=True,
        selector_initial_SHA256=init,selector_final_SHA256=parameter_sha(selector),checkpoint_SHA256=sha256(p),selector_parameters=sum(p.numel() for p in selector.parameters()),
        source_train_manifest_SHA256=plan['train_manifest_SHA256'],source_eval_manifest_SHA256=plan['eval_manifest_SHA256'],weight_SHA256=plan['weight_SHA256'],
        updates=512,observed_exposure=exposure,normal_target_exposure=target_exposure,budget=budget,seconds=time.time()-start,
        cache_manifest_SHA256=sha256(cache/'manifest.json'),benchmark_arrays_opened=False,normal_twin_in_selector_input=False,TimeRCD_RCA=False,
        extra_selector_eval_passes=2*576+2*128,peak_cuda_bytes=torch.cuda.max_memory_allocated(),code=plan['code']))
    progress('complete',step=512)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--task',type=int,required=True);a=p.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors'/f'task{a.task}.json',dict(error=traceback.format_exc()));raise
