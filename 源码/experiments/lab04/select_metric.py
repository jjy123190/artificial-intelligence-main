"""Pre-test second selection stage using only base-class covariance and validation tasks.

Depends on completed multiview selection. The original locked candidate stays
as a control. No new test result is accessed to choose these settings.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import argparse,json
import numpy as np
from .data import digest_file
from .experiment import write_json,log
from .refine import tasks_for,validation_partition
from .multiview import combine_views
from .graph_calibration import GraphConfig,apply_transform
from .supervised_whitening import fit
from .study_multiview import evaluate


def run(args):
    source,out=Path(args.study),Path(args.output);out.mkdir(parents=True,exist_ok=True)
    previous=json.loads((source/'locked.json').read_text())
    if previous['status']!='locked':raise ValueError('First stage unfinished')
    plan={'seed':args.seed,'source_lock_sha256':digest_file(source/'locked.json'),
          'exponents':[.25,.5,.75,1.],'residual_mix':[0.,.5,1.],
          'search_episodes':120,'confirmation_episodes':300,'test_accessed':False}
    plan_path=out/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Changed metric plan')
    write_json(plan_path,plan)
    with np.load(source/'train_views.npz',allow_pickle=False) as data:train={k:data[k] for k in data.files}
    with np.load(source/'val_views.npz',allow_pickle=False) as data:val={k:data[k] for k in data.files}
    search,confirm=validation_partition(val['labels'],args.seed)
    result={'status':'selecting','test_accessed':False,'plan':plan,'shots':{}}
    for shot in (1,5):
        chosen=previous['selected'][str(shot)];cfg=GraphConfig(**chosen['config'])
        training=combine_views(train['original'],train['flipped'],chosen['views'])
        validation=combine_views(val['original'],val['flipped'],chosen['views'])
        tasks=tasks_for(val['labels'][search],args.seed+10+shot,shot,120)
        confirms=tasks_for(val['labels'][confirm],args.seed+20+shot,shot,300)
        rows=[]
        for exponent in plan['exponents']:
            for residual_mix in plan['residual_mix']:
                space={'strength':1.,'exponent':exponent,'residual_mix':residual_mix}
                transform=fit(training,train['labels'],**space)
                features=apply_transform(validation[search],transform)
                score=evaluate(features,tasks,cfg,chosen['assignment'])
                rows.append({'space':space,'views':chosen['views'],'assignment':chosen['assignment'],'config':chosen['config'],**score})
        shortlist=sorted(rows,key=lambda r:-r['accuracy'])[:5]
        previous_row=next(r for r in rows if r['space']['residual_mix']==0 and r['space']['exponent']==chosen['space']['exponent'])
        for row in shortlist+([] if previous_row in shortlist else [previous_row]):
            transform=fit(training,train['labels'],**row['space']);features=apply_transform(validation[confirm],transform)
            row['confirmation']=evaluate(features,confirms,cfg,chosen['assignment'])
        winner=max([*shortlist,previous_row],key=lambda r:r['confirmation']['accuracy'])
        result['shots'][str(shot)]={'candidates':rows,'shortlist':shortlist,'selected':winner,'previous':previous_row}
        write_json(out/'validation.json',result);log('metric_validation',shots=shot,selected=winner,previous=previous_row['confirmation'])
    result['status']='complete';result['completed_utc']=datetime.now(timezone.utc).isoformat();write_json(out/'validation.json',result)
    write_json(out/'locked.json',{'status':'locked','locked_utc':datetime.now(timezone.utc).isoformat(),'test_accessed':False,
      'checkpoint_sha256':previous['checkpoint_sha256'],'selection_sha256':digest_file(out/'validation.json'),
      'previous_selection_sha256':previous['selection_sha256'],
      'selected':{str(n):{k:result['shots'][str(n)]['selected'][k] for k in ('views','space','assignment','config')} for n in (1,5)}})
    log('metric_selection_complete')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',default='.local/lab04-multiview')
    p.add_argument('--output',default='.local/lab04-metric');p.add_argument('--seed',type=int,default=20261126)
    run(p.parse_args())

if __name__=='__main__':main()
