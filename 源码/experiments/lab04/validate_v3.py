"""Validate a completed wider self-trained candidate without touching test data."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from PIL import Image
from .data import image_records,normalized_tensor,digest_file
from .experiment import load_checkpoint,write_json,log,summarize
from .refine import tasks_for,validation_partition,read_cache
from .multiview import combine_views
from .multiview_extension import inference_inputs as multiview_inputs
from .graph_extension import graph_inputs
from .graph_calibration import GraphConfig,centered_whitening,apply_transform,predict


def cached_features(args,split):
    directory=Path(args.candidate);checkpoint=directory/'best.pt';digest=digest_file(checkpoint)
    cache=directory/f'{split}_features.npz';stamp=directory/f'{split}_cache.json'
    if cache.exists() and stamp.exists():
        metadata=json.loads(stamp.read_text())
        if metadata['checkpoint_sha256']==digest:
            if metadata.get('cache_sha256')!=digest_file(cache):raise ValueError('Candidate feature cache checksum mismatch')
            with np.load(cache,allow_pickle=False) as src:return {k:src[k] for k in src.files}
    if split not in ('train','val'):raise ValueError('No test access allowed')
    records=image_records(args.data_root,split);names=sorted({r[0] for r in records});mapping={n:i for i,n in enumerate(names)}
    model,_=load_checkpoint(checkpoint);model.eval();features=[];labels=[];ids=[]
    with torch.inference_mode():
        for start in range(0,len(records),64):
            images=[]
            for name,path,id in records[start:start+64]:
                with Image.open(path) as image:images.append(np.asarray(image.convert('RGB')))
                labels.append(mapping[name]);ids.append(id)
            features.append(model.features(normalized_tensor(np.stack(images))).numpy())
            if (start//64+1)%150==0:log('wide_features',split=split,images=start+len(images))
    value={'features':np.concatenate(features),'labels':np.array(labels),'ids':np.array(ids)}
    np.savez_compressed(cache,**value);write_json(stamp,{'checkpoint_sha256':digest,'images':len(labels),'split':split,'cache_sha256':digest_file(cache)})
    return value


def evaluate_features(features,tasks,config):
    values=[]
    for task in tasks:
        prediction=predict(features[task['support_indices']],task['support_labels'],features[task['query_indices']],config)
        values.append(float(np.mean(prediction['predictions']==task['query_labels'])))
    return summarize(values)


def run(args):
    directory=Path(args.candidate);training=json.loads((directory/'training.json').read_text())
    if training['status']!='complete':raise ValueError('Training is not complete')
    torch.set_num_threads(args.threads)
    train=cached_features(args,'train');val=cached_features(args,'val')
    selection,confirmation=validation_partition(val['labels'],args.seed)
    results={'test_accessed':False,'checkpoint_sha256':digest_file(directory/'best.pt'),'shots':{}}
    # A reference is meaningful only on exactly these validation image IDs and
    # tasks. This is still validation, not a second independent test estimate.
    existing=read_cache(Path(args.reference_features),'aligned','val')
    if not np.array_equal(existing['ids'],val['ids']):raise ValueError('Reference validation sample IDs differ')
    with np.load(Path(args.multiview_features)/'val_views.npz',allow_pickle=False) as source:
        views={key:source[key] for key in source.files}
    if not np.array_equal(views['ids'],val['ids']):raise ValueError('Multi-view validation sample IDs differ')
    for shots in (1,5):
        tasks=tasks_for(val['labels'][selection],args.seed+10+shots,shots,120)
        check_tasks=tasks_for(val['labels'][confirmation],args.seed+20+shots,shots,300)
        records=[]
        for center in (0.,1.):
            for exponent in (0.,.25,.5):
                transform=centered_whitening(train['features'],center,exponent)
                x=apply_transform(val['features'][selection],transform)
                for graph in (0.,.5,1.):
                    config=GraphConfig(neighbors=5,alpha=.95,power=3.,temperature=.1,prototype_weight=.8,prototype_steps=8,graph_weight=graph,balanced=True)
                    records.append({'space':{'strength':center,'exponent':exponent},'config':config.to_dict(),**evaluate_features(x,tasks,config)})
        shortlist=sorted(records,key=lambda r:-r['accuracy'])[:5]
        for row in shortlist:
            transform=centered_whitening(train['features'],**row['space'])
            x=apply_transform(val['features'][confirmation],transform)
            row['confirmation']=evaluate_features(x,check_tasks,GraphConfig(**row['config']))
        winner=max(shortlist,key=lambda r:r['confirmation']['accuracy'])
        reference_transform,reference_cfg=graph_inputs(args.artifacts,shots)
        reference_embedding=apply_transform(existing['features'][confirmation],reference_transform)
        reference_score=evaluate_features(reference_embedding,check_tasks,reference_cfg)
        mt,mg,mv=multiview_inputs(args.artifacts,shots)
        multiview_embedding=apply_transform(combine_views(views['original'][confirmation],views['flipped'][confirmation],mv.views),mt)
        if mv.assignment!='soft':raise ValueError('Explicit capacity comparison must use multiview.predict')
        multiview_score=evaluate_features(multiview_embedding,check_tasks,mg)
        results['shots'][str(shots)]={'candidates':records,'shortlist':shortlist,'selected':winner,
            'references':{'graph':reference_score,'multiview':multiview_score}}
        write_json(directory/'validation_selection.json',results);log('wide_validation',shots=shots,selected=winner)
    results['status']='complete';write_json(directory/'validation_selection.json',results)
    write_json(directory/'selection_lock.json',{'status':'locked','locked_utc':datetime.now(timezone.utc).isoformat(),
        'test_accessed':False,'checkpoint_sha256':results['checkpoint_sha256'],
        'training_plan_sha256':digest_file(directory/'plan.json'),
        'selection_sha256':digest_file(directory/'validation_selection.json'),
        'selected':{str(n):{k:results['shots'][str(n)]['selected'][k] for k in ('space','config')} for n in (1,5)},
        'plan':'24-epoch wider backbone, validation-only checkpoint and classifier selection; no test feedback'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',default='.local/lab04-v3-wide')
    p.add_argument('--data-root',default='.local/mini-imagenet');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261125)
    p.add_argument('--artifacts',default='outputs/lab04')
    p.add_argument('--reference-features',default='.local/lab04-v2')
    p.add_argument('--multiview-features',default='.local/lab04-multiview')
    run(p.parse_args())

if __name__=='__main__':main()
