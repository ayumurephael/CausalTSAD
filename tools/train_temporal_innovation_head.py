"""Fixed64 source updates; matched views, seeds and nominal head capacity."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,argparse,time,traceback,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score,roc_auc_score
from proposals.temporal_innovation_head import TemporalInnovationHead,objective
from causal_tsad.io import atomic_json,sha256,array_sha
from tools.train_meta_basis import psha
def load(root,index,seed,key,bg):
    row=index[str(seed)+'/'+key+'/'+bg];p=root/row['path'];assert sha256(p)==row['SHA256']
    with np.load(p) as f:return f['features'].copy(),f['energy'].copy(),f['labels'].copy(),dict(row,temperature=float(f['temperature']),bias=float(f['bias']))
def evaluate(m,rows,index,a,runroot,seed,stage):
    records=[];slots=0;fixedslots=0
    for row in rows:
        for bg in ['plain','smooth']:
            assert time.time()<a.deadline;ff,energy,labels,cache=load(a.root,index,seed,row['key'],bg)
            with torch.no_grad():o=m(torch.tensor(ff,device='cuda'));score=o['score'].cpu().numpy();root=o['root'].cpu().numpy()
            slots+=ff.shape[0]*1024*row['C'];metrics=[]
            for v in range(len(score)):
                meta=row['case_metadata'][v];y=labels[v]>0;valid=bool(y.any() and not y.all());item=dict(view=v,kind=meta['kind'],duration=meta['duration'],AP=float(average_precision_score(y,score[v])) if valid else None,ROC_AUC=float(roc_auc_score(y,score[v])) if valid else None,Top1=None,Top3=None,MRR=None)
                if meta['root'] is not None and y.any():
                    r=root[v,y].mean(0);order=np.argsort(-r,kind='stable');rank=int(np.flatnonzero(order==meta['root'])[0])+1;item.update(Top1=int(rank==1),Top3=int(rank<=min(3,row['C'])),MRR=1/rank)
                metrics.append(item)
            path=runroot/'evaluation'/stage/(row['key']+'_'+bg);path.parent.mkdir(parents=True,exist_ok=True);assert not path.with_suffix('.npz').exists()
            # Same compressed point initialization baseline, plus old Gaussian
            # energy point/W16, are cached for source-only comparisons.
            fixed={};fixedmetrics={}
            for name,e in [('original_point',energy),('original16',F.avg_pool1d(F.pad(torch.tensor(energy).transpose(1,2),(15,0),mode='replicate'),16,stride=1).transpose(1,2).numpy()),('compressed_point',ff[...,2]-math.log(3))]:
                temp=1. if name=='compressed_point' else cache['temperature'];bias=0. if name=='compressed_point' else cache['bias'];normalized=e/temp;maximum=normalized.max(-1);fixed[name]=temp*(maximum+np.log(np.exp(normalized-maximum[...,None]).sum(-1))-math.log(row['C']))+bias;fixedslots+=ff.shape[0]*1024*row['C']
                fixedmetrics[name]=[]
                eroot=np.exp(e-e.max(-1,keepdims=True));eroot/=eroot.sum(-1,keepdims=True)
                for v in range(len(score)):
                    meta=row['case_metadata'][v];y=labels[v]>0
                    if not y.any():continue
                    order=np.argsort(-eroot[v,y].mean(0),kind='stable');rank=int(np.flatnonzero(order==meta['root'])[0])+1
                    fixedmetrics[name].append(dict(view=v,kind=meta['kind'],duration=meta['duration'],AP=float(average_precision_score(y,fixed[name][v])),Top1=int(rank==1),Top3=int(rank<=min(3,row['C'])),MRR=1/rank))
            np.savez_compressed(path.with_suffix('.npz'),scores=score,**fixed)
            record=dict(key=row['key'],split=row['split'],background=bg,C=row['C'],family=row['family'],metrics=metrics,fixed_metrics=fixedmetrics,feature_cache_SHA256=cache['SHA256'],feature_SHA256=cache['feature_SHA256'],input_SHA256=cache['input_SHA256'],scores_SHA256=sha256(path.with_suffix('.npz')))
            atomic_json(path.with_suffix('.json'),record);records.append(dict(path=str(path.with_suffix('.json').relative_to(runroot)),SHA256=sha256(path.with_suffix('.json')),scores_SHA256=record['scores_SHA256']))
    return records,slots,fixedslots
def run(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    plan=json.loads(a.plan.read_text());a.deadline=plan['deadline_unix'];index=json.loads((a.root/'feature_index.json').read_text());source=json.loads((Path(plan['source'])/'manifest.json').read_text());rows={v['key']:v for v in source['files']};runroot=a.root/'runs'/(a.mode+'_s'+str(a.seed));runroot.mkdir(parents=True,exist_ok=False)
    m=TemporalInnovationHead(a.mode,a.seed).cuda();initial=psha(m);torch.save(dict(model=m.state_dict()),runroot/'initial.pth');opt=torch.optim.Adam(m.parameters(),lr=.001);curves=[];exposure=0;headquery=0
    for step,job in enumerate(plan['training_plan']):
        assert time.time()<a.deadline and rows[job['key']]['split']=='train';row=rows[job['key']];ff,_,labels,cache=load(a.root,index,a.seed,job['key'],job['background']);ids=job['views'];x=torch.tensor(ff[ids],device='cuda');y=torch.tensor(labels[ids],device='cuda');roots=torch.tensor([row['case_metadata'][i]['root'] or 0 for i in ids],device='cuda')
        opt.zero_grad(set_to_none=True);o=m(x);loss,det,rca=objective(o,y,roots);assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),5));opt.step()
        exposure+=len(ids)*1024*row['C'];headquery+=len(ids)*704*row['C'];curves.append(dict(step=step+1,key=row['key'],background=job['background'],views=ids,cache_SHA256=cache['SHA256'],selected_feature_SHA256=array_sha(ff[ids]),loss=float(loss.detach()),detection_BCE=float(det.detach()),conditional_root_CE=float(rca.detach()),gradient_norm=gn,raw_input_exposure=exposure))
        atomic_json(runroot/'progress.json',dict(status='training',updates=step+1,raw_input_exposure=exposure))
    assert len(curves)==64 and exposure==7340032
    final=runroot/'final.pth';torch.save(dict(model=m.state_dict(),seed=a.seed,mode=a.mode,updates=64,raw_input_exposure=exposure,parameters=1217,feature_index_SHA256=sha256(a.root/'feature_index.json')),final)
    m.eval().requires_grad_(False);before=psha(m);records=[];evalslots=fixedslots=0
    for stage in ['calibration','source_validation']:
        new,slots,fs=evaluate(m,[v for v in source['files'] if v['split']==stage],index,a,runroot,a.seed,stage);records+=new;evalslots+=slots;fixedslots+=fs
    assert psha(m)==before;atomic_json(runroot/'curves.json',curves)
    atomic_json(runroot/'manifest.json',dict(status='complete',seed=a.seed,mode=a.mode,updates=64,raw_input_exposure=exposure,head_query_slots=headquery,
        evaluation_raw_slot_applications=evalslots,fixed_baseline_slot_applications=fixedslots,parameters=sum(p.numel() for p in m.parameters()),dense_multiply_adds_per_node_time=1168,
        initial_parameter_SHA256=initial,final_parameter_SHA256=psha(m),initial_checkpoint_SHA256=sha256(runroot/'initial.pth'),final_checkpoint_SHA256=sha256(final),records=records,
        plan_SHA256=sha256(a.plan),feature_index_SHA256=sha256(a.root/'feature_index.json'),target_arrays_opened=False,target_updates=0,backbone_optimizer_updates=0))
    atomic_json(runroot/'progress.json',dict(status='complete',updates=64,raw_input_exposure=exposure))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--mode',required=True);a=p.parse_args()
    try:run(a)
    except Exception:atomic_json(a.root/'errors'/(a.mode+'_s'+str(a.seed)+'.json'),dict(error=traceback.format_exc()));raise
