"""Locked wider-backbone comparison: same test tasks, no task or seed selection.

A wider candidate is a separate capacity/training-budget study. This command
never relabels its gain as a calibration-only gain. Publication is not automatic.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from .experiment import load_checkpoint,extract_features,write_json,log,summarize
from .data import load_split,selection_digest,sample_episode,digest_file
from .graph_calibration import GraphConfig,centered_whitening,apply_transform,predict


def run(args):
    candidate,core,out=Path(args.candidate),Path(args.artifacts),Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    training=json.loads((candidate/'training.json').read_text())
    if training['status']!='complete':raise ValueError('Training incomplete')
    lock=json.loads((candidate/'selection_lock.json').read_text())
    if lock['status']!='locked' or lock['test_accessed'] is not False:raise ValueError('Missing selection lock')
    if digest_file(candidate/'best.pt')!=lock['checkpoint_sha256']:raise ValueError('Candidate weights changed after selection')
    if digest_file(candidate/'validation_selection.json')!=lock['selection_sha256']:raise ValueError('Selection evidence changed')
    if digest_file(candidate/'plan.json')!=lock['training_plan_sha256']:raise ValueError('Training plan changed')
    meta=json.loads((core/'results.json').read_text(encoding='utf-8'))
    plan={'status':'locked','selection_lock_sha256':digest_file(candidate/'selection_lock.json'),
        'seed':meta['protocol']['seed'],'episodes_per_shot':600,'pool_per_class':600,
        'baseline_checkpoint_sha256':meta['weights']['baseline']['sha256'],
        'candidate_checkpoint_sha256':lock['checkpoint_sha256'],
        'validation_reference_sha256':digest_file(candidate/'validation_selection.json'),
        'source_sha256':{name:digest_file(Path(__file__).with_name(name)) for name in ('data.py','models.py','graph_calibration.py','evaluate_wide.py')}}
    plan_file=out/'evaluation_plan.json'
    if plan_file.exists() and json.loads(plan_file.read_text())!=plan:raise ValueError('Changed running evaluation plan')
    write_json(plan_file,plan)
    train_cache=candidate/'train_features.npz'
    cache_manifest=json.loads((candidate/'train_cache.json').read_text())
    if cache_manifest['checkpoint_sha256']!=lock['checkpoint_sha256'] or digest_file(train_cache)!=cache_manifest['cache_sha256']:
        raise ValueError('Invalid training-only transform input')
    with np.load(train_cache,allow_pickle=False) as src:
        transforms={n:centered_whitening(src['features'],**lock['selected'][str(n)]['space']) for n in (1,5)}
    np.savez_compressed(out/'wide_transform.npz',**{f'shot{n}_{key}':v for n,t in transforms.items() for key,v in t.items()})
    torch.set_num_threads(args.threads)
    model,checkpoint=load_checkpoint(candidate/'best.pt');model.eval()
    test=load_split(args.data_root,'test',max_per_class=600,seed=meta['protocol']['seed']+4)
    if selection_digest(test)!=meta['protocol']['test_selection_sha256']:raise ValueError('Test pool is not paired')
    features_path=out/'test_features.npz'
    features_manifest=out/'test_features.json'
    feature_identity={'checkpoint_sha256':lock['checkpoint_sha256'], 'selection_sha256':selection_digest(test)}
    if features_path.exists() or features_manifest.exists():
        if not (features_path.is_file() and features_manifest.is_file()):raise ValueError('Incomplete wide feature cache; do not silently trust partial output')
        cached=json.loads(features_manifest.read_text())
        if cached['identity']!=feature_identity or cached['sha256']!=digest_file(features_path):raise ValueError('Wide test cache identity mismatch')
        with np.load(features_path,allow_pickle=False) as src:features=src['features']
    else:
        features=extract_features(model,test.images,batch_size=64)
        np.savez_compressed(features_path,features=features)
        write_json(features_manifest,{'identity':feature_identity,'sha256':digest_file(features_path)})
    if features.shape!=(len(test.images),model.widths[-1]) or not np.isfinite(features).all():raise ValueError('Invalid wider test features')
    published=json.loads((core/'episodes.json').read_text(encoding='utf-8'))['episodes']
    published={(r['shots'],r['episode']):r for r in published}
    graph=json.loads((core/'graph_episodes.json').read_text(encoding='utf-8'))['episodes'];graph={(r['shots'],r['episode']):r for r in graph}
    multiview=json.loads((core/'multiview_episodes.json').read_text(encoding='utf-8'))['episodes'];multiview={(r['shots'],r['episode']):r for r in multiview}
    journal=out/'wide_episodes.jsonl';records=[]
    if journal.exists():records=[json.loads(x) for x in journal.read_text().splitlines() if x]
    complete={(r['shots'],r['episode']):r for r in records}
    if len(complete)!=len(records):raise ValueError('Duplicate candidate tasks')
    for shots in (1,5):
        embedded=apply_transform(features,transforms[shots]);cfg=GraphConfig(**lock['selected'][str(shots)]['config'])
        rng=np.random.default_rng(meta['protocol']['seed']+200+shots)
        for i in range(600):
            task=sample_episode(test.labels,rng,shots=shots,queries=15)
            support_ids=[test.ids[j] for j in task['support_indices']];query_ids=[test.ids[j] for j in task['query_indices']]
            old=published[(shots,i)]
            if old['support_ids']!=support_ids or old['query_ids']!=query_ids:raise ValueError('Unpaired task')
            if (shots,i) in complete:
                if complete[(shots,i)]['query_ids']!=query_ids:raise ValueError('Invalid resume record')
                continue
            t=time.perf_counter();result=predict(embedded[task['support_indices']],task['support_labels'],embedded[task['query_indices']],cfg)
            elapsed=time.perf_counter()-t
            raw=predict(features[task['support_indices']],task['support_labels'],features[task['query_indices']],
                        GraphConfig(prototype_steps=0,graph_weight=0,balanced=False))
            row={'shots':shots,'episode':i,'support_ids':support_ids,'query_ids':query_ids,'query_labels':task['query_labels'].tolist(),
                'predictions':result['predictions'].tolist(),'probabilities':result['probabilities'].tolist(),
                'accuracy':float(np.mean(result['predictions']==task['query_labels'])),'seconds':elapsed,
                'raw_predictions':raw['predictions'].tolist(),'raw_accuracy':float(np.mean(raw['predictions']==task['query_labels'])),
                'baseline_accuracy':old['methods']['baseline']['accuracy'],'full_accuracy':old['methods']['full']['accuracy'],
                'graph_accuracy':graph[(shots,i)]['accuracy'],'multiview_accuracy':multiview[(shots,i)]['accuracy']}
            with journal.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n');stream.flush()
            records.append(row)
            if (i+1)%100==0:log('wide_test',shots=shots,episodes=i+1)
    rows=[]
    for shots in (1,5):
        tasks=[r for r in records if r['shots']==shots]
        if len(tasks)!=600:raise ValueError('Incomplete evaluation')
        values=np.array([r['accuracy'] for r in tasks]);row={'method':'wide','shots':shots,**summarize(values),
          'raw_accuracy':float(np.mean([r['raw_accuracy'] for r in tasks])),
          'mean_episode_seconds':float(np.mean([r['seconds'] for r in tasks])),
          'timing_scope':'inference_from_cached_features','transductive':True,'balanced_queries':True}
        for field,prefix in [('baseline_accuracy','paired'),('full_accuracy','paired_vs_full'),('graph_accuracy','paired_vs_graph'),('multiview_accuracy','paired_vs_multiview')]:
            delta=values-np.array([r[field] for r in tasks]);row['paired_delta' if prefix=='paired' else prefix]=float(delta.mean())
            row[prefix+'_ci95']=float(1.96*delta.std(ddof=1)/600**.5)
        rows.append(row)
    write_json(out/'wide_episodes.json',{'episodes':records})
    write_json(out/'wide_configuration.json',lock)
    write_json(out/'wide_results.json',{'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),
      'candidate_checkpoint_sha256':lock['checkpoint_sha256'],'model':{'name':'ResNet12-Metric-Wide','widths':list(model.widths),'parameters':sum(p.numel() for p in model.parameters())},
      'training':{'selected_epoch':training['best_epoch'],'epochs_completed':len(training['rows']),'warm_start':'Function-preserving widening from self-trained v2 baseline; no external weights'},
      'rows':rows,'selection_lock_sha256':digest_file(out/'wide_configuration.json'),'evaluation_plan':plan,
      'files':{name:{'path':name,'sha256':digest_file(out/name)} for name in ('wide_transform.npz','wide_configuration.json','wide_episodes.json')},
      'limitations':['Separate larger-backbone and extra-training-budget study, not a calibration-only comparison.',
                    'One training seed; fixed public benchmark; validation-based selection.',
                    'Uses jointly unlabelled balanced query batches.']})
    log('wide_evaluation_complete',rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',default='.local/lab04-v3-wide')
    p.add_argument('--artifacts',default='outputs/lab04');p.add_argument('--output',default='.local/lab04-wide-test')
    p.add_argument('--data-root',default='.local/mini-imagenet');p.add_argument('--threads',type=int,default=4)
    run(p.parse_args())

if __name__=='__main__':main()
