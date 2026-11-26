"""One frozen 1200-task test after independent continuation validation passes.

Adds no search or model selection. Original published weights remain untouched.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from .continuation_audit import audit as audit_validation
from .data import digest_file,load_split,normalized_tensor,sample_episode,selection_digest
from .experiment import load_checkpoint,write_json,log,summarize
from .graph_calibration import GraphConfig,apply_transform,predict
from .multiview import combine_views


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def run(args):
    candidate=Path(args.candidate);validation=Path(args.validation);core=Path(args.artifacts);out=Path(args.output)
    verified=audit_validation(candidate,validation,core)
    if not verified['test_eligible']:raise ValueError('Prerecorded validation gate did not pass; do not access test')
    out.mkdir(parents=True,exist_ok=True)
    lock=read(validation/'selection_lock.json');validation_plan=read(validation/'validation_plan.json');training=read(candidate/'training.json');metadata=read(core/'results.json')
    configurations=validation_plan['fixed_configurations']
    references=('results.json','episodes.json','graph_episodes.json','multiview_episodes.json','wide_episodes.json','wideview_episodes.json','wideview_results.json')
    sources=('evaluate_continuation.py','continuation_audit.py','data.py','models.py','graph_calibration.py','multiview.py','calibration.py')
    plan={'status':'locked','checkpoint_sha256':lock['checkpoint_sha256'],'selection_lock_sha256':digest_file(validation/'selection_lock.json'),
          'validation_sha256':digest_file(validation/'validation.json'),'validation_plan_sha256':digest_file(validation/'validation_plan.json'),
          'training_plan_sha256':digest_file(candidate/'plan.json'),'training_sha256':digest_file(candidate/'training.json'),
          'transform_sha256':digest_file(validation/'transform.npz'),'seed':metadata['protocol']['seed'],'episodes_per_shot':600,
          'queries_per_class':15,'test_pool_sha256':metadata['protocol']['test_selection_sha256'],
          'reference_files':{name:digest_file(core/name) for name in references},
          'sources':{name:digest_file(Path(__file__).with_name(name)) for name in sources},
          'selected_configurations':configurations,'test_selection':'Original fixed paired tasks; all retained; no hyperparameter changes permitted'}
    plan_path=out/'evaluation_plan.json'
    if plan_path.exists() and read(plan_path)!=plan:raise ValueError('Frozen test plan or selected candidate changed')
    if not plan_path.exists():write_json(plan_path,plan)
    if (out/'continued_results.json').exists() and read(out/'continued_results.json').get('status')=='complete':log('continued_test_already_complete');return
    torch.set_num_threads(args.threads)
    test=load_split(args.data_root,'test',max_per_class=600,seed=metadata['protocol']['seed']+4)
    if selection_digest(test)!=plan['test_pool_sha256']:raise ValueError('Test pool not the original shared selection')
    model,_=load_checkpoint(candidate/'best.pt');model.eval();cache=out/'test_views.npz';stamp=out/'test_views.json'
    identity={'checkpoint_sha256':lock['checkpoint_sha256'],'selection_sha256':selection_digest(test),'sources':plan['sources']}
    if cache.exists() or stamp.exists():
        if not(cache.is_file() and stamp.is_file()):raise ValueError('Partial test feature cache')
        value=read(stamp)
        if value['identity']!=identity or value['sha256']!=digest_file(cache):raise ValueError('Test feature cache mismatch')
        with np.load(cache,allow_pickle=False) as src:views={key:src[key] for key in src.files}
        if not np.array_equal(views['ids'],test.ids):raise ValueError('Test feature sample IDs changed')
    else:
        normal=[];flipped=[];started=time.perf_counter()
        with torch.inference_mode():
            for start in range(0,len(test.images),32):
                x=normalized_tensor(test.images[start:start+32]);normal.append(model.features(x).numpy());flipped.append(model.features(x.flip(-1)).numpy())
                if (start//32+1)%100==0:log('continued_test_encode',images=start+len(x),seconds=time.perf_counter()-started)
        views={'original':np.concatenate(normal),'flipped':np.concatenate(flipped),'ids':np.array(test.ids),'labels':test.labels}
        np.savez_compressed(cache,**views);write_json(stamp,{'identity':identity,'sha256':digest_file(cache)})
    previous={(r['shots'],r['episode']):r for r in read(core/'episodes.json')['episodes']}
    extensions={name:{(r['shots'],r['episode']):r for r in read(core/f'{name}_episodes.json')['episodes']} for name in ('graph','multiview','wide','wideview')}
    journal=out/'continued_episodes.jsonl';records=[json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()] if journal.exists() else []
    done={(r['shots'],r['episode']):r for r in records}
    if len(done)!=len(records):raise ValueError('Duplicate resumed test task')
    for shot in (1,5):
        selected=configurations[str(shot)];config=GraphConfig(**selected['config'])
        with np.load(validation/'transform.npz',allow_pickle=False) as src:
            prefix=f'shot{shot}_';transform={key[len(prefix):]:src[key] for key in src.files if key.startswith(prefix)}
        features=apply_transform(combine_views(views['original'],views['flipped'],selected['views']),transform)
        rng=np.random.default_rng(metadata['protocol']['seed']+200+shot)
        for number in range(600):
            task=sample_episode(test.labels,rng,shots=shot,queries=15);key=(shot,number);old=previous[key]
            supports=[test.ids[i] for i in task['support_indices']];queries=[test.ids[i] for i in task['query_indices']]
            if supports!=old['support_ids'] or queries!=old['query_ids'] or task['query_labels'].tolist()!=old['query_labels']:raise ValueError('Task pairing changed')
            if key in done:
                if done[key]['support_ids']!=supports or done[key]['query_ids']!=queries:raise ValueError('Resume task pairing changed')
                continue
            started=time.perf_counter();answer=predict(features[task['support_indices']],task['support_labels'],features[task['query_indices']],config);seconds=time.perf_counter()-started
            row={'shots':shot,'episode':number,'support_ids':supports,'query_ids':queries,'query_labels':task['query_labels'].tolist(),
                 'predictions':answer['predictions'].tolist(),'probabilities':answer['probabilities'].tolist(),
                 'accuracy':float(np.mean(answer['predictions']==task['query_labels'])),'seconds':seconds,
                 'baseline_accuracy':old['methods']['baseline']['accuracy'],'full_accuracy':old['methods']['full']['accuracy'],
                 **{name+'_accuracy':value[key]['accuracy'] for name,value in extensions.items()}}
            with journal.open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
            records.append(row)
            if (number+1)%150==0:log('continued_test',shot=shot,tasks=number+1)
    summary=[]
    for shot in (1,5):
        tasks=sorted((r for r in records if r['shots']==shot),key=lambda r:r['episode']);values=np.array([r['accuracy'] for r in tasks])
        if len(tasks)!=600:raise ValueError('Incomplete test rows')
        row={'method':'continued','shots':shot,**summarize(values),'mean_episode_seconds':float(np.mean([r['seconds'] for r in tasks])),
             'timing_scope':'inference_from_cached_features','transductive':True,'balanced_queries':True}
        for name in ('baseline','full','graph','multiview','wide','wideview'):
            delta=values-np.array([r[name+'_accuracy'] for r in tasks]);mean='paired_delta' if name=='baseline' else 'paired_vs_'+name;ci='paired_ci95' if name=='baseline' else mean+'_ci95'
            row[mean]=float(delta.mean());row[ci]=float(1.96*delta.std(ddof=1)/600**.5)
        summary.append(row)
    write_json(out/'continued_episodes.json',{'episodes':records})
    configuration={'status':'locked','test_accessed':False,'locked_utc':lock['locked_utc'],'checkpoint_sha256':lock['checkpoint_sha256'],
                   'transform_sha256':lock['transform_sha256'],'selected':configurations,'selection_lock_sha256':digest_file(validation/'selection_lock.json')}
    write_json(out/'continued_configuration.json',configuration)
    copies={'continued.pt':candidate/'best.pt','continued_transform.npz':validation/'transform.npz',
            'continued_training.json':candidate/'training.json','continued_training_plan.json':candidate/'plan.json','continued_initial_validation.json':candidate/'initial_validation.json',
            'continued_validation.json':validation/'validation.json','continued_validation_plan.json':validation/'validation_plan.json',
            'continued_validation_tasks.jsonl':validation/'validation_tasks.jsonl','continued_checkpoint_lock.json':validation/'checkpoint_lock.json','continued_selection_lock.json':validation/'selection_lock.json'}
    for name,source in copies.items():(out/name).write_bytes(source.read_bytes())
    files=(*copies,'continued_configuration.json','continued_episodes.json')
    write_json(out/'continued_results.json',{'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':lock['checkpoint_sha256'],
        'model':{'name':'ResNet12-Metric-Continued','widths':list(model.widths),'parameters':sum(p.numel() for p in model.parameters())},
        'training':{'additional_epochs':len(training['rows']),'selected_additional_epoch':training['best_epoch'],'original_wide_epochs':24,'same_capacity':True},
        'rows':summary,'evaluation_plan':plan,'files':{name:{'path':name,'sha256':digest_file(out/name)} for name in files},
        'scope':'Same-capacity continued training from wide.pt; additional 16-epoch budget; frozen wideview downstream settings; not training-budget matched.',
        'limitations':['Single training seed and reused public benchmark/validation classes.','Balanced transductive query protocol.','Additional training budget; no isolated algorithm-only attribution.']})
    log('continued_test_complete',rows=summary)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',default='.local/lab04-continuation-20261125')
    p.add_argument('--validation',default='.local/lab04-continuation-validation-20261125');p.add_argument('--artifacts',default='outputs/lab04')
    p.add_argument('--output',default='.local/lab04-continued-test-20261125');p.add_argument('--data-root',default='.local/mini-imagenet');p.add_argument('--threads',type=int,default=2)
    run(p.parse_args())


if __name__=='__main__':main()
