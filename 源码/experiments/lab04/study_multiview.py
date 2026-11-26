"""Search deterministic multi-view/capacity candidates on validation classes only.

Fixed candidate grid and task seeds are recorded before looking at scores.
No test data is accessed by this command; candidates do not modify the website.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import asdict
import argparse,json,time
import numpy as np
import torch
from PIL import Image
from .data import image_records,normalized_tensor,digest_file
from .experiment import load_checkpoint,write_json,log,summarize
from .refine import read_cache,validation_partition,tasks_for
from .graph_calibration import centered_whitening,apply_transform,GraphConfig
from .multiview import combine_views,predict


def extract_flip(args,split):
    if split not in ('train','val'):raise ValueError('Only train/validation are allowed in selection')
    out=Path(args.output);source=Path(args.features)
    cache=out/f'{split}_views.npz';manifest_file=out/f'{split}_views.json'
    original=read_cache(source,'aligned',split)
    identity={'checkpoint_sha256':digest_file(Path(args.artifacts)/'aligned.pt'),
              'original_cache_sha256':digest_file(source/f'aligned_{split}_features.npz'),
              'view':'normalized tensor horizontal flip; no crop/randomness',
              'data_source_sha256':digest_file(Path(__file__).with_name('data.py'))}
    if cache.exists() and manifest_file.exists():
        manifest=json.loads(manifest_file.read_text())
        if manifest['identity']!=identity:raise ValueError('Multi-view cache identity changed')
        if digest_file(cache)!=manifest['sha256']:raise ValueError('Multi-view cache was modified')
        with np.load(cache,allow_pickle=False) as src:return {k:src[k] for k in src.files}
    model,_=load_checkpoint(Path(args.artifacts)/'aligned.pt');model.eval()
    records=image_records(args.data_root,split);by_id={row[2]:row for row in records}
    features=[];start_time=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(original['ids']),64):
            images=[]
            for id in original['ids'][start:start+64]:
                with Image.open(by_id[str(id)][1]) as image:images.append(np.asarray(image.convert('RGB')))
            tensor=normalized_tensor(np.stack(images)).flip(-1)
            features.append(model.features(tensor).numpy())
            if (start//64+1)%150==0:log('views_encode',split=split,images=start+len(images),seconds=time.perf_counter()-start_time)
    value={'original':original['features'],'flipped':np.concatenate(features),
           'labels':original['labels'],'ids':original['ids']}
    np.savez_compressed(cache,**value)
    write_json(manifest_file,{'identity':identity,'sha256':digest_file(cache)})
    return value


def evaluate(features,tasks,config,assignment):
    values=[]
    for task in tasks:
        result=predict(features[task['support_indices']],task['support_labels'],features[task['query_indices']],config,assignment)
        values.append(float(np.mean(result['predictions']==task['query_labels'])))
    return summarize(values)


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads)
    plan={'seed':args.seed,'views':['original','mean','concat'],'exponents':[.25,.5,.75],
          'strength':1.,'assignments':['soft','balanced_capacity'],'search_episodes':120,'confirmation_episodes':300,
          'graph_config':GraphConfig(neighbors=5,alpha=.95,power=3.,temperature=.1,prototype_weight=.8,prototype_steps=8,graph_weight=.5,balanced=True).to_dict(),
          'checkpoint_sha256':digest_file(Path(args.artifacts)/'aligned.pt'),'test_accessed':False}
    plan_path=output/'selection_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Cannot change a running selection plan')
    write_json(plan_path,plan)
    training=extract_flip(args,'train');validation=extract_flip(args,'val')
    search,confirmation=validation_partition(validation['labels'],args.seed)
    graph=GraphConfig(**plan['graph_config'])
    full={'status':'selecting','plan':plan,'shots':{}}
    cache_rows={}
    journal=output/'selection_records.jsonl'
    if journal.exists():
        for line in journal.read_text().splitlines():
            row=json.loads(line);cache_rows[row['key']]=row['score']
    def cached_score(key,features,tasks,assignment):
        if key in cache_rows:return cache_rows[key]
        score=evaluate(features,tasks,graph,assignment)
        with journal.open('a',encoding='utf-8') as f:f.write(json.dumps({'key':key,'score':score})+'\n')
        cache_rows[key]=score;return score
    for shots in (1,5):
        tasks=tasks_for(validation['labels'][search],args.seed+10+shots,shots,120)
        confirmation_tasks=tasks_for(validation['labels'][confirmation],args.seed+20+shots,shots,300)
        records=[]
        for views in plan['views']:
            base_train=combine_views(training['original'],training['flipped'],views)
            base_val=combine_views(validation['original'],validation['flipped'],views)
            for exponent in plan['exponents']:
                transform=centered_whitening(base_train,strength=1.,exponent=exponent)
                features=apply_transform(base_val[search],transform)
                for assignment in plan['assignments']:
                    key=f'{shots}/{views}/{exponent}/{assignment}/search'
                    row={'views':views,'space':{'strength':1.,'exponent':exponent},'assignment':assignment,
                         'config':plan['graph_config'],**cached_score(key,features,tasks,assignment)}
                    records.append(row)
                log('views_candidate',shots=shots,views=views,exponent=exponent)
        shortlist=sorted(records,key=lambda r:-r['accuracy'])[:5]
        # Always retain the published original-graph reference on the same
        # confirmation tasks, even when it is not selected in the shortlist.
        reference=next(r for r in records if r['views']=='original' and r['space']['exponent']==.5 and r['assignment']=='soft')
        check=shortlist if reference in shortlist else shortlist+[reference]
        for row in check:
            base_train=combine_views(training['original'],training['flipped'],row['views'])
            base_val=combine_views(validation['original'],validation['flipped'],row['views'])
            transform=centered_whitening(base_train,**row['space'])
            features=apply_transform(base_val[confirmation],transform)
            key=f"{shots}/{row['views']}/{row['space']['exponent']}/{row['assignment']}/confirmation"
            row['confirmation']=cached_score(key,features,confirmation_tasks,row['assignment'])
        winner=max(shortlist,key=lambda r:r['confirmation']['accuracy'])
        full['shots'][str(shots)]={'candidates':records,'shortlist':shortlist,'selected':winner,'graph_reference':reference}
        write_json(output/'validation.json',full);log('views_validation_selected',shots=shots,selected=winner,reference=reference['confirmation'])
    full['status']='complete';full['completed_utc']=datetime.now(timezone.utc).isoformat();write_json(output/'validation.json',full)
    locked={'status':'locked','locked_utc':datetime.now(timezone.utc).isoformat(),'test_accessed':False,
            'checkpoint_sha256':plan['checkpoint_sha256'],'selection_sha256':digest_file(output/'validation.json'),
            'selected':{str(n):{k:full['shots'][str(n)]['selected'][k] for k in ('views','space','assignment','config')} for n in (1,5)}}
    write_json(output/'locked.json',locked);log('views_selection_complete')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',default='.local/mini-imagenet')
    p.add_argument('--artifacts',default='outputs/lab04');p.add_argument('--features',default='.local/lab04-v2')
    p.add_argument('--output',default='.local/lab04-multiview');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--seed',type=int,default=20261126);run(p.parse_args())

if __name__=='__main__':main()
