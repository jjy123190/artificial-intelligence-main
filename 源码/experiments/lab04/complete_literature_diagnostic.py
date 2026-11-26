"""Complete the capacity diagnostic after an explicit numeric-budget amendment.

The original teacher PT-MAP call hit the strict mass-residual guard at 300
Sinkhorn iterations. That failure is kept, not relabeled as an accuracy result.
This separate preregistration uses the upstream 1000-iteration cap for BOTH
student and teacher. No temperature/centroid/power/seed/task selection changes.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .data import digest_file
from .graph_calibration import GraphConfig
from .literature_methods import MAPConfig,map_predict_batch,prepare_episode_batch
from .multiview import combine_views
from .calibration import unit_features
from .study_literature import (inputs,make_tasks,spaces,aligned_indices,read,write,digest_json,
    score,summary,paired,validate_scores,DIAGNOSTIC_IDS)


def audit(out,tasks,plan_sha):
    plan=read(out/'plan.json');result=read(out/'diagnostic.json');scores={};failures={}
    for p in (out/'scores').glob('*.json'):
        row=read(p);key=(row['group'],row['candidate'])
        if key in scores or key in failures:raise ValueError('Duplicate diagnostic')
        if row.get('status')=='failed_numeric':
            if row['plan_sha256']!=plan_sha or row['test_accessed'] is not False:raise ValueError('Invalid failure record')
            failures[key]=row
        else:scores[key]=validate_scores(row,*key,tasks['groups'][row['group']],plan_sha)
    for shot in (1,5):
        row=result['shots'][str(shot)]
        expected={}
        for name in DIAGNOSTIC_IDS:
            key=(f'diagnostic/{shot}',name)
            if key in scores:expected[name]=summary(scores[key])
            elif key in failures:expected[name]={'status':'failed_numeric','accuracy':None,'episodes_scored':0}
            else:raise ValueError('Missing method; do not silently omit failure')
        if row['methods']!=expected:raise ValueError('Diagnostic values do not match predictions')
        pairs={}
        for h in ('centered_proto','centered_graph','ptmap'):
            a,b=(f'diagnostic/{shot}',f'teacher_{h}'),(f'diagnostic/{shot}',f'student_{h}')
            if a in scores and b in scores:pairs[h]=paired(scores[a],scores[b])
        if row['matched_head_pairs']!=pairs:raise ValueError('Diagnostic pair mismatch')
        if row['teacher_vs_published']!=paired(scores[(f'diagnostic/{shot}','teacher_centered_graph')],scores[(f'diagnostic/{shot}','continued_published')]):
            raise ValueError('Reference pair mismatch')
    check={'status':'passed','test_accessed':False,'plan_sha256':plan_sha,
           'score_files':len(scores),'numeric_failures':len(failures),'episode_predictions':sum(len(v) for v in scores.values()),
           'summary_and_pairs_recomputed':True,
           'files':{str(p.relative_to(out)):digest_file(p) for p in out.rglob('*.json') if p.name!='audit.json'}}
    write(out/'audit.json',check);return check


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'scores').mkdir(exist_ok=True)
    original=Path(args.original);pools,external,hashes,provenance=inputs(args);tasks=make_tasks(pools,external)
    old_plan=read(original/'plan.json')
    if old_plan['inputs']!=hashes or old_plan['tasks_content_sha256']!=digest_json(tasks):raise ValueError('Original inputs/tasks changed')
    for name,value in old_plan['source_hashes'].items():
        if digest_file(Path(__file__).with_name(name))!=value:raise ValueError('Original source changed')
    reusable={}
    for p in (original/'scores').glob('diagnostic*.json'):
        row=read(p)
        if row['candidate'].endswith('ptmap'):continue
        validate_scores(row,row['group'],row['candidate'],tasks['groups'][row['group']],digest_file(original/'plan.json'))
        reusable[str(p)]=digest_file(p)
    plan={'schema_version':1,'kind':'validation_capacity_diagnostic_numeric_amendment','test_accessed':False,
          'original_plan_sha256':digest_file(original/'plan.json'),'original_failure_sha256':digest_file(original/'diagnostic_failure.json'),
          'source_sha256':digest_file(__file__),'original_sources':old_plan['source_hashes'],'inputs':hashes,
          'tasks_content_sha256':digest_json(tasks),'reused_scores':reusable,'external_reference':provenance,
          'only_change':'Sinkhorn max iterations 300 -> 1000 for both PT-MAP heads, matching upstream maxiters; strict residual guard retained.',
          'map_config':MAPConfig(sinkhorn_steps=1000).to_dict(),
          'on_failure':'Record numeric failure with no accuracy; never drop the failing task or loosen the residual check.',
          'limitations':old_plan['limitations']}
    if (out/'plan.json').exists():
        old=read(out/'plan.json')
        if {k:v for k,v in old.items() if k!='registered_utc'}!=plan:raise ValueError('Diagnostic supplement changed')
    else:write(out/'plan.json',{**plan,'registered_utc':datetime.now(timezone.utc).isoformat()})
    plan_sha=digest_file(out/'plan.json');write(out/'tasks.json',tasks)
    print(json.dumps({'registered':True,'test_accessed':False,'amendment':plan['only_change']}),flush=True)
    if args.audit_only:
        print(json.dumps(audit(out,tasks,plan_sha)));return
    for path in reusable:
        record=read(path);target=out/'scores'/Path(path).name
        converted={**record,'plan_sha256':plan_sha,'reused_original_score_sha256':reusable[path]}
        if target.exists():
            if read(target)!=converted:raise ValueError('Inherited score modified')
        else:write(target,converted)
    fs,cfg=spaces(args,pools)
    vi=aligned_indices(pools['val'],external['val']);ti=aligned_indices(pools['train'],external['train'])
    st=combine_views(pools['train']['original'][ti],pools['train']['flipped'][ti],'mean')
    sv=fs[1]['raw'][vi];tt=unit_features(external['train']['features']);tv=unit_features(external['val']['features'])
    vec={'student_raw':sv,'teacher_raw':tv,'student_centered':unit_features(sv-st.mean(0)),
         'teacher_centered':unit_features(tv-tt.mean(0))}
    result={'status':'complete','test_accessed':False,'plan_sha256':plan_sha,'shots':{},
            'student_parameters':509856,'external_encoder_parameters':12424320,'external_checkpoint':provenance,
            'paired_val_images':3200,'paired_train_center_images':6400,'scope':'validation only; NOT a published local-model result',
            'amendment':plan['only_change'],'limitations':plan['limitations']}
    for shot in (1,5):
        group=f'diagnostic/{shot}';batch_tasks=tasks['groups'][group];graph_cfg=GraphConfig(**cfg[str(shot)]['config'])
        values={};failures=[]
        for name in DIAGNOSTIC_IDS:
            if not name.endswith('ptmap'):
                if name=='continued_published':spec={'id':name,'algorithm':'graph'};features=fs[shot]['whitened'][vi]
                else:spec={'id':name,'algorithm':name.split('_')[-1]};features=vec[name.split('_')[0]+'_centered']
                values[name]=score(args,plan_sha,group,spec,features,batch_tasks,graph_cfg)
                continue
            path=out/'scores'/f'{group.replace("/","_")}_{name}.json'
            if path.exists():
                record=read(path)
                if record.get('status')=='failed_numeric':failures.append(name)
                else:values[name]=validate_scores(record,group,name,batch_tasks,plan_sha)
                continue
            features=vec[name.split('_')[0]+'_raw'];predictions=[];residuals=[];start=time.perf_counter()
            try:
                for offset in range(0,len(batch_tasks),40):
                    batch=batch_tasks[offset:offset+40]
                    si=np.asarray([t['support_indices'] for t in batch]);qi=np.asarray([t['query_indices'] for t in batch]);ys=np.asarray([t['support_labels'] for t in batch])
                    support,query=prepare_episode_batch(features[si],features[qi],signed_power=.5,task_center=True)
                    observed=map_predict_batch(support,ys,query,MAPConfig(sinkhorn_steps=1000))
                    predictions.extend(observed['predictions'].tolist());residuals.append(observed['max_transport_residual'])
            except ValueError as error:
                if 'Transport failed to converge' not in str(error):raise
                record={'status':'failed_numeric','test_accessed':False,'plan_sha256':plan_sha,'group':group,'candidate':name,
                        'error':str(error),'first_failed_batch_start':offset,'accuracy':None,'episodes_scored':0,
                        'reason':'Reject entire method; no failed-task filtering; no query truth scored.'}
                write(path,record);failures.append(name);print(json.dumps(record),flush=True);continue
            truth=np.asarray([t['query_labels'] for t in batch_tasks]);accuracy=(np.asarray(predictions)==truth).mean(1).tolist()
            record={'test_accessed':False,'plan_sha256':plan_sha,'group':group,'candidate':name,'predictions':predictions,
                    'accuracies':accuracy,'max_transport_residual':max(residuals),'cached_head_seconds':time.perf_counter()-start}
            values[name]=validate_scores(record,group,name,batch_tasks,plan_sha);write(path,record)
            print(json.dumps({'group':group,'candidate':name,**summary(accuracy),'seconds':record['cached_head_seconds']}),flush=True)
        result['shots'][str(shot)]={'methods':{**{n:summary(v) for n,v in values.items()},
                         **{n:{'status':'failed_numeric','accuracy':None,'episodes_scored':0} for n in failures}},
              'matched_head_pairs':{h:paired(values[f'teacher_{h}'],values[f'student_{h}']) for h in ('centered_proto','centered_graph','ptmap')
                                    if f'teacher_{h}' in values and f'student_{h}' in values},
              'teacher_vs_published':paired(values['teacher_centered_graph'],values['continued_published'])}
    write(out/'diagnostic.json',result);print(json.dumps(audit(out,tasks,plan_sha)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',default='.local/lab04-continuation-validation-20261125')
    p.add_argument('--external',default='.local/lab04-reference-feat')
    p.add_argument('--artifacts',default='outputs/lab04')
    p.add_argument('--original',default='outputs/lab04/next_study/literature_inference_20261125')
    p.add_argument('--output',default='outputs/lab04/next_study/literature_diagnostic_20261125')
    p.add_argument('--audit-only',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):run(args)


if __name__=='__main__':main()
