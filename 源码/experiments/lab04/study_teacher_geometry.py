"""Separate preregistered validation-only, external-teacher metric-transfer probe.

This is ridge geometry transfer inspired by representation distillation, not RFS
or a trained small neural student. No test path, teacher query, or publication.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .data import digest_file
from .graph_calibration import GraphConfig
from .multiview import combine_views
from .study_literature import (inputs, make_tasks, spaces, aligned_indices, read, write, digest_json,
                               score, summary, paired, validate_scores)
from .teacher_geometry import fit_teacher_geometry, apply_teacher_geometry


CANDIDATES = tuple({"id": f"ridge{r:g}_mix{m:g}", "ridge_ratio": r, "mix": m}
                   for r in (.01, .1, 1.) for m in (.5, 1.))


def prepare(args, hashes, provenance, tasks):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'scores').mkdir(exist_ok=True)
    names=('study_teacher_geometry.py','teacher_geometry.py','study_literature.py','literature_methods.py',
           'graph_calibration.py','calibration.py','multiview.py','data.py')
    plan={"schema_version":1,"kind":"external_teacher_geometry_validation_only","test_accessed":False,
          "inputs":hashes,"external_reference":provenance,"candidates":list(CANDIDATES),
          "tasks_content_sha256":digest_json(tasks),
          "source_hashes":{n:digest_file(Path(__file__).with_name(n)) for n in names},
          "fit":"6400 paired base-train images only; no teacher val/query features in fitting or prediction",
          "selection":"160 search tasks/shot, top 2; 600 confirmation tasks/shot, largest paired lower95 bound; ties by ID",
          "gate":"Both shot paired lower95>0 and both mean gains>=.01; never auto-test or publish",
          "limitations":["External teacher lineage; not comparable as no-external-training improvement.",
             "Inspired by RFS representation distillation, NOT the RFS method or evidence of neural student distillation.",
             "Teacher trained by author; complete original training cannot be independently verified.",
             "Validation classes/images are reused, and confirmation participates in selection.",
             "No change to image encoder; linear geometry cannot recover missing semantic information."]}
    path=out/'plan.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='registered_utc'}!=plan or read(out/'tasks.json')!=tasks:
            raise ValueError('Immutable geometry study changed')
    else:
        if any((out/'scores').iterdir()) or (out/'tasks.json').exists():raise ValueError('Unexpected partial study')
        write(out/'tasks.json',tasks);write(path,{**plan,'registered_utc':datetime.now(timezone.utc).isoformat()})
    return read(path)


def run(args,plan,tasks,pools,external):
    out=Path(args.output);plan_sha=digest_file(out/'plan.json')
    features,cfg=spaces(args,pools)
    paired_train=aligned_indices(pools['train'],external['train'])
    student=combine_views(pools['train']['original'][paired_train],pools['train']['flipped'][paired_train],'mean')
    transforms={c['id']:fit_teacher_geometry(student,external['train']['features'],
                  ridge_ratio=c['ridge_ratio'],mix=c['mix']) for c in CANDIDATES}
    transformed={key:apply_teacher_geometry(features[1]['raw'],t) for key,t in transforms.items()}
    result={'status':'complete','plan_sha256':plan_sha,'test_accessed':False,'shots':{},'limitations':plan['limitations']}
    for shot in (1,5):
        graph_cfg=GraphConfig(**cfg[str(shot)]['config'])
        references={}
        for phase in ('search','confirmation'):
            group=f'{phase}/{shot}'
            references[phase]=score(args,plan_sha,group,{'id':'continued_published','algorithm':'graph'},
                 features[shot]['whitened'],tasks['groups'][group],graph_cfg)
        rows=[]
        for candidate in CANDIDATES:
            values=score(args,plan_sha,f'search/{shot}',{'id':candidate['id'],'algorithm':'graph'},
                 transformed[candidate['id']],tasks['groups'][f'search/{shot}'],graph_cfg)
            rows.append({**candidate,'search':summary(values),'search_paired':paired(values,references['search'])})
        rows.sort(key=lambda r:(-r['search']['accuracy'],r['id']))
        for row in rows[:2]:
            values=score(args,plan_sha,f'confirmation/{shot}',{'id':row['id'],'algorithm':'graph'},
                transformed[row['id']],tasks['groups'][f'confirmation/{shot}'],graph_cfg)
            row['confirmation']=summary(values);row['confirmation_paired']=paired(values,references['confirmation'])
        winner=min(rows[:2],key=lambda r:(-r['confirmation_paired']['low'],r['id']))
        result['shots'][str(shot)]={'candidates':rows,'selected':winner['id'],
            'reference':{k:summary(v) for k,v in references.items()},
            'eligible':winner['confirmation_paired']['low']>0 and winner['confirmation_paired']['mean']>=.01}
    result['eligible']=all(r['eligible'] for r in result['shots'].values())
    result['publication']='not_published; external-teacher lineage; validation-only'
    write(out/'selection.json',result)
    # Store selected compact geometry; exclude full teacher weights and training features.
    selected={}
    for shot,row in result['shots'].items():
        for key,value in transforms[row['selected']].items():
            if key!='regression':selected[f'shot{shot}_{key}']=value
    path=out/'selected_transform.npz'
    if not path.exists():np.savez_compressed(path,**selected)
    else:
        with np.load(path,allow_pickle=False) as previous:
            if set(previous.files)!=set(selected) or any(not np.array_equal(previous[k],v) for k,v in selected.items()):
                raise ValueError('Selected geometry changed')


def audit(args,plan,tasks):
    out=Path(args.output);plan_sha=digest_file(out/'plan.json')
    result=read(out/'selection.json');values={}
    for p in (out/'scores').glob('*.json'):
        r=read(p);key=(r['group'],r['candidate'])
        if key in values:raise ValueError('Duplicate candidate record')
        values[key]=validate_scores(r,*key,tasks['groups'][r['group']],plan_sha)
    eligibility=[]
    for shot in (1,5):
        reference={p:values[(f'{p}/{shot}','continued_published')] for p in ('search','confirmation')}
        rank=sorted([c['id'] for c in CANDIDATES],key=lambda n:(-np.mean(values[(f'search/{shot}',n)]),n))
        row=result['shots'][str(shot)];records=row['candidates']
        if [r['id'] for r in records]!=rank:raise ValueError('Search ranking mismatch')
        for r in records:
            v=values[(f'search/{shot}',r['id'])]
            if r['search']!=summary(v) or r['search_paired']!=paired(v,reference['search']):raise ValueError('Search summary mismatch')
            if r['id'] in rank[:2]:
                v=values[(f'confirmation/{shot}',r['id'])]
                if r['confirmation']!=summary(v) or r['confirmation_paired']!=paired(v,reference['confirmation']):raise ValueError('Confirmation mismatch')
        winner=min(records[:2],key=lambda r:(-r['confirmation_paired']['low'],r['id']))
        eligible=winner['confirmation_paired']['low']>0 and winner['confirmation_paired']['mean']>=.01
        if row['selected']!=winner['id'] or row['eligible'] is not eligible:raise ValueError('Selection mismatch')
        if row['reference']!={k:summary(v) for k,v in reference.items()}:raise ValueError('Reference mismatch')
        eligibility.append(eligible)
    if result['eligible'] is not all(eligibility):raise ValueError('Eligibility mismatch')
    report={'status':'passed','test_accessed':False,'plan_sha256':plan_sha,'score_files':len(values),
            'episode_predictions':sum(len(v) for v in values.values()),'selection_recomputed':True,
            'files':{str(p.relative_to(out)):digest_file(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='audit.json'}}
    write(out/'audit.json',report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',default='.local/lab04-continuation-validation-20261125')
    p.add_argument('--external',default='.local/lab04-reference-feat')
    p.add_argument('--artifacts',default='outputs/lab04')
    p.add_argument('--output',default='outputs/lab04/next_study/teacher_geometry_20261125')
    p.add_argument('--phase',choices=('register','run','audit'),default='run')
    args=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        pools,external,hashes,provenance=inputs(args);tasks=make_tasks(pools,external)
        plan=prepare(args,hashes,provenance,tasks)
        print(json.dumps({'registered':True,'phase':args.phase,'test_accessed':False}),flush=True)
        if args.phase=='register':return
        if args.phase=='run':run(args,plan,tasks,pools,external)
        audit(args,plan,tasks)


if __name__=='__main__':main()
