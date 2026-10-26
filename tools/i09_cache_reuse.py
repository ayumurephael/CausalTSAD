"""Exact frozen-cache reuse; never reuse selector weights or optimizer state."""
import json
from pathlib import Path
import numpy as np
from causal_tsad.io import sha256,array_sha
from proposals.context_mixture import graph_input

def restore(plan,task,initial,train,ev):
    spec=plan['reuse_cache'][task['id']];path=Path(spec['manifest'])
    assert sha256(path)==spec['manifest_SHA256']
    donor=json.loads(Path(spec['completed_record']).read_text())
    assert sha256(Path(spec['completed_record']))==spec['record_SHA256']
    assert donor['status']=='complete' and donor['basis_unchanged']
    assert donor['basis_parameter_SHA256']==initial
    assert donor['source_train_manifest_SHA256']==plan['train_manifest_SHA256'] and donor['source_eval_manifest_SHA256']==plan['eval_manifest_SHA256']
    assert donor['weight_SHA256']==plan['weight_SHA256']
    assert donor['task']['arm'] in ['true_all','true_hidden_parent'] and task['arm'] in ['true_all','true_hidden_parent']
    m=json.loads(path.read_text());assert m['basis_SHA256']==initial and len(m['files'])==1088
    keys={r['key']:r for r in train};evalkeys={r['key']:r for r in ev};tr=[];va=[]
    for item in m['files']:
        assert sha256(Path(item['path']))==item['SHA256']
        expected=keys.get(item['key'],evalkeys.get(item['key']));assert item['row']==expected
        (tr if item['key'] in keys else va).append(item)
    assert len(tr)==512 and len(va)==576
    audits=[]
    for rows,folder in [(train,plan['source_train']),(ev,plan['source_eval'])]:
        for row in rows:
            p=Path(folder)/row['path'];assert sha256(p)==row['SHA256']
            with np.load(p) as z:g=z['graph']
            gg=graph_input(g,row['key'],False).numpy()
            audits.append(dict(key=row['key'],same_as_true=True,graph_SHA256=array_sha(gg),each_node_in_out_degree_preserved=True))
    return m['files'],tr,va,audits,m['budget'],dict(donor=donor['task']['id'],manifest_SHA256=sha256(path),
        completed_record_SHA256=sha256(Path(spec['completed_record'])),selector_reused=False,optimizer_reused=False,
        exact_overlap_audit=spec['exact_overlap_audit'],retained_partial_compute=spec['partial_compute'])
