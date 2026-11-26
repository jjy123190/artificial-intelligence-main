"""Fixed downstream validation of the continued wider candidate; no test access.

The hyperparameters are copied from the previously published two-view strategy,
not searched again. Only a completed, frozen training plan may enter this stage.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .data import digest_file,image_records,normalized_tensor
from .experiment import load_checkpoint,log,summarize,write_json
from .graph_calibration import GraphConfig,apply_transform,predict
from .multiview import combine_views
from .refine import tasks_for
from .supervised_whitening import fit
from .study_cached_refinement import paired_stats


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def initial_plan(args):
    source=Path(args.candidate);artifacts=Path(args.artifacts);references=Path(args.reference_views)
    training_plan=read(source/'plan.json')
    if training_plan.get('test_accessed') is not False:raise ValueError('Training must not use test data')
    frozen=read(artifacts/'wideview_configuration.json')
    names=('validate_continuation.py','train_continuation.py','data.py','models.py','supervised_whitening.py','graph_calibration.py','multiview.py','calibration.py','refine.py')
    return {'status':'registered','seed':args.seed,'episodes_per_shot':600,'test_accessed':False,
            'candidate_training_plan_sha256':digest_file(source/'plan.json'),
            'reference_weights_sha256':digest_file(artifacts/'wide.pt'),'reference_configuration_sha256':digest_file(artifacts/'wideview_configuration.json'),
            'reference_transform_sha256':digest_file(artifacts/'wideview_transform.npz'),
            'reference_val_cache_sha256':digest_file(references/'val_views.npz'),
            'reference_val_stamp_sha256':digest_file(references/'val_views.json'),
            'fixed_configurations':frozen['selected'],'source_hashes':{name:digest_file(Path(__file__).with_name(name)) for name in names},
            'rule':'No hyperparameter selection. Compare selected continued checkpoint vs frozen wideview on 600 identical validation tasks/shot; test eligible only if paired lower 95% bound > 0 for both shots.',
            'limitation':'Same repeatedly used validation classes; checkpoint selection has used a subset of these classes/images. This is validation, not independent test.'}


def extract(args,split):
    if split not in ('train','val'):raise ValueError('Only train/validation allowed')
    source=Path(args.candidate);output=Path(args.output);checkpoint=source/'best.pt'
    path=output/f'{split}_views.npz';stamp=output/f'{split}_views.json'
    identity={'checkpoint_sha256':digest_file(checkpoint),'split':split,'source_sha256':digest_file(Path(__file__)),
              'preprocessing_sha256':digest_file(Path(__file__).with_name('data.py'))}
    if path.exists() or stamp.exists():
        if not(path.is_file() and stamp.is_file()):raise ValueError('Incomplete candidate cache; inspect before retry')
        metadata=read(stamp)
        if metadata['identity']!=identity or metadata['sha256']!=digest_file(path):raise ValueError('Candidate feature cache changed')
        with np.load(path,allow_pickle=False) as src:return {k:src[k] for k in src.files}
    records=image_records(args.data_root,split);names=sorted({r[0] for r in records});mapping={n:i for i,n in enumerate(names)}
    if len(names)!=(64 if split=='train' else 16):raise ValueError('Invalid class count')
    model,_=load_checkpoint(checkpoint);model.eval();normal=[];flipped=[];ids=[];labels=[];start_time=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(records),32):
            images=[]
            for name,file,identity_id in records[start:start+32]:
                with Image.open(file) as image:images.append(np.asarray(image.convert('RGB').resize((84,84),Image.Resampling.BICUBIC)))
                ids.append(identity_id);labels.append(mapping[name])
            tensor=normalized_tensor(np.stack(images));normal.append(model.features(tensor).numpy());flipped.append(model.features(tensor.flip(-1)).numpy())
            if (start//32+1)%200==0:log('continued_features',split=split,images=start+len(images),seconds=time.perf_counter()-start_time)
    data={'original':np.concatenate(normal),'flipped':np.concatenate(flipped),'ids':np.array(ids),'labels':np.array(labels)}
    temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,**data);temporary.replace(path)
    write_json(stamp,{'identity':identity,'sha256':digest_file(path),'images':len(ids)})
    return data


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);source=Path(args.candidate);artifacts=Path(args.artifacts)
    plan=initial_plan(args);path=out/'validation_plan.json'
    if path.exists() and read(path)!=plan:raise ValueError('Registered downstream validation plan changed')
    if not path.exists():write_json(path,plan)
    if args.register_only:log('continuation_validation_registered',path=str(path));return
    training=read(source/'training.json')
    if training.get('status')!='complete' or training.get('test_accessed') is not False:raise ValueError('Training must be complete before validation')
    for name,digest in read(source/'plan.json')['source_hashes'].items():
        if digest_file(Path(__file__).with_name(name))!=digest:raise ValueError('Training source changed after training plan')
    result_path=out/'validation.json'
    if result_path.exists() and read(result_path).get('status')=='complete':log('continued_validation_already_complete');return
    frozen={'training_sha256':digest_file(source/'training.json'),'checkpoint_sha256':digest_file(source/'best.pt'),
            'plan_sha256':digest_file(path),'best_epoch':training['best_epoch'],'test_accessed':False}
    checkpoint_lock=out/'checkpoint_lock.json'
    if checkpoint_lock.exists() and read(checkpoint_lock)!=frozen:raise ValueError('Validation checkpoint changed')
    if not checkpoint_lock.exists():write_json(checkpoint_lock,frozen)
    if training['best_epoch']==0:
        write_json(result_path,{'status':'complete','test_accessed':False,'selected_original':True,'promising':False,
                               'reason':'Epoch zero retained; no new model selected by training validation.','checkpoint_lock':frozen})
        log('continued_validation_original_retained');return
    torch.set_num_threads(args.threads);train=extract(args,'train');val=extract(args,'val')
    if {str(i).split('/')[1] for i in train['ids']}&{str(i).split('/')[1] for i in val['ids']}:raise ValueError('Train/val class overlap')
    reference_file=Path(args.reference_views)/'val_views.npz';reference_stamp=read(Path(args.reference_views)/'val_views.json')
    if reference_stamp['sha256']!=digest_file(reference_file) or reference_stamp['identity']['checkpoint_sha256']!=plan['reference_weights_sha256']:raise ValueError('Unverified reference cache')
    with np.load(reference_file,allow_pickle=False) as src:reference={k:src[k] for k in src.files}
    if not np.array_equal(reference['ids'],val['ids']) or not np.array_equal(reference['labels'],val['labels']):raise ValueError('Validation samples differ')
    result={'status':'validating','test_accessed':False,'checkpoint_lock':frozen,'shots':{},'validation_ids_sha256':hashlib.sha256('\n'.join(val['ids']).encode()).hexdigest()}
    transform_values={};records=[];journal=out/'validation_tasks.jsonl'
    if journal.exists():records=[json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()]
    done={(r['shots'],r['episode']):r for r in records}
    if len(done)!=len(records):raise ValueError('Duplicate validation tasks')
    for shot in (1,5):
        cfg=plan['fixed_configurations'][str(shot)];mode=cfg['views'];config=GraphConfig(**cfg['config'])
        transform=fit(combine_views(train['original'],train['flipped'],mode),train['labels'],strength=1.,exponent=cfg['exponent'],residual_mix=cfg['residual_mix'])
        embedded=apply_transform(combine_views(val['original'],val['flipped'],mode),transform)
        with np.load(artifacts/'wideview_transform.npz',allow_pickle=False) as src:
            prefix=f'shot{shot}_';reference_transform={k[len(prefix):]:src[k] for k in src.files if k.startswith(prefix)}
        reference_features=apply_transform(combine_views(reference['original'],reference['flipped'],mode),reference_transform)
        tasks=tasks_for(val['labels'],args.seed+100+shot,shot,plan['episodes_per_shot'])
        for i,task in enumerate(tasks):
            key=(shot,i);support_ids=val['ids'][task['support_indices']].tolist();query_ids=val['ids'][task['query_indices']].tolist()
            if key in done:
                if done[key]['support_ids']!=support_ids or done[key]['query_ids']!=query_ids:raise ValueError('Resume task differs')
                continue
            old=predict(reference_features[task['support_indices']],task['support_labels'],reference_features[task['query_indices']],config)
            new=predict(embedded[task['support_indices']],task['support_labels'],embedded[task['query_indices']],config)
            labels=task['query_labels'];row={'shots':shot,'episode':i,'support_ids':support_ids,'query_ids':query_ids,'query_labels':labels.tolist(),
                'reference_predictions':old['predictions'].tolist(),'predictions':new['predictions'].tolist(),
                'reference_accuracy':float(np.mean(old['predictions']==labels)),'accuracy':float(np.mean(new['predictions']==labels))}
            with journal.open('a',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(row)+'\n')
            records.append(row)
        tasks=[r for r in records if r['shots']==shot];values=[r['accuracy'] for r in tasks];references=[r['reference_accuracy'] for r in tasks]
        pair=paired_stats(values,references)
        result['shots'][str(shot)]={'candidate':summarize(values),'reference':summarize(references),'paired':pair,'promising':pair['mean']-pair['ci95']>0}
        transform_values.update({f'shot{shot}_{k}':v for k,v in transform.items()})
        write_json(result_path,result);log('continued_validation_shot',shot=shot,**result['shots'][str(shot)])
    np.savez_compressed(out/'transform.npz',**transform_values)
    result.update({'status':'complete','promising':all(r['promising'] for r in result['shots'].values()),'completed_utc':datetime.now(timezone.utc).isoformat(),
                   'journal_sha256':digest_file(journal),'transform_sha256':digest_file(out/'transform.npz')})
    write_json(result_path,result);write_json(out/'selection_lock.json',{'status':'locked','test_accessed':False,'plan_sha256':digest_file(path),
        'checkpoint_sha256':frozen['checkpoint_sha256'],'validation_sha256':digest_file(result_path),'transform_sha256':result['transform_sha256'],'locked_utc':datetime.now(timezone.utc).isoformat()})
    log('continued_validation_complete',promising=result['promising'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',default='.local/lab04-continuation-20261125')
    p.add_argument('--output',default='.local/lab04-continuation-validation-20261125');p.add_argument('--artifacts',default='outputs/lab04')
    p.add_argument('--data-root',default='.local/mini-imagenet');p.add_argument('--reference-views',default='.local/lab04-wide-views-20261125')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=20261129);p.add_argument('--register-only',action='store_true')
    run(p.parse_args())


if __name__=='__main__':main()
