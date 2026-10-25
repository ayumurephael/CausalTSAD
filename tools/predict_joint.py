"""Read only values and normal adjacency; labels/roots/twins are never input."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from causal_tsad.joint_baseline import JointTSADRCA
from causal_tsad.io import atomic_json

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['input','output','original-code','base-checkpoint','teacher-checkpoint','head-checkpoint']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();assert not a.output.exists(),'Preserve old outputs; choose a new path'
 with np.load(a.input) as f:values=np.array(f['values']);graph=np.array(f['adjacency'])
 model=JointTSADRCA(a.original_code,a.base_checkpoint,a.teacher_checkpoint,a.head_checkpoint);result=model.predict(values,graph);a.output.mkdir(parents=True)
 np.savez_compressed(a.output/'predictions.npz',**result)
 atomic_json(a.output/'events.json',dict(events=model.alarm_events(result),threshold=float(model.threshold),provenance=model.provenance,
  scope='Automatic threshold runs, no true windows input; free-running event precision/RCA not yet measured by the published conditional Top1'))
