"""One locked, paired test for validation-selected multi-view classification.

All candidate settings are locked first. Reference task IDs are checked before
scoring. Saved predictions, labels and timings permit independent recomputation.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import replace
import argparse,json,time
import numpy as np
import torch
from .data import load_split,digest_file,selection_digest,sample_episode
from .experiment import load_checkpoint,extract_features,write_json,log,summarize
from .graph_calibration import GraphConfig,centered_whitening,apply_transform
from .supervised_whitening import fit as fit_metric
from .multiview import combine_views,predict
from .data import normalized_tensor


def run(args):
    study,core,out=Path(args.study),Path(args.artifacts),Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    lock=json.loads((study/'locked.json').read_text(encoding='utf-8'))
    if lock.get('status')!='locked' or lock.get('test_accessed') is not False:raise ValueError('No validation lock')
    if digest_file(core/'aligned.pt')!=lock['checkpoint_sha256']:raise ValueError('Locked backbone changed')
    metadata=json.loads((core/'results.json').read_text(encoding='utf-8'))
    plan={'selection_lock_sha256':digest_file(study/'locked.json'),
          'checkpoint_sha256':lock['checkpoint_sha256'],'seed':metadata['protocol']['seed'],
          'episodes_per_shot':600,'test_images_per_class':600,
          'source_sha256':{n:digest_file(Path(__file__).with_name(n)) for n in ('multiview.py','graph_calibration.py','supervised_whitening.py','evaluate_multiview.py','data.py')}}
    plan_file=out/'evaluation_plan.json'
    if plan_file.exists() and json.loads(plan_file.read_text())!=plan:raise ValueError('Changed evaluation plan')
    write_json(plan_file,plan)
    with np.load(Path(args.views)/'train_views.npz',allow_pickle=False) as train:
        transforms={}
        for n in (1,5):
            settings=lock['selected'][str(n)]
            base_features=combine_views(train['original'],train['flipped'],settings['views'])
            transforms[n]=(fit_metric(base_features,train['labels'],**settings['space'])
                           if 'residual_mix' in settings['space'] else centered_whitening(base_features,**settings['space']))
    np.savez_compressed(out/'multiview_transform.npz',**{f'shot{n}_{key}':value for n,t in transforms.items() for key,value in t.items()})
    torch.set_num_threads(args.threads)
    test=load_split(args.data_root,'test',max_per_class=600,seed=metadata['protocol']['seed']+4)
    if selection_digest(test)!=metadata['protocol']['test_selection_sha256']:raise ValueError('Unpaired test pool')
    model,_=load_checkpoint(core/'aligned.pt');model.eval()
    feature_path=out/'test_views.npz'
    if feature_path.exists():
        with np.load(feature_path,allow_pickle=False) as src:original,flipped=src['original'],src['flipped']
    else:
        original=extract_features(model,test.images,batch_size=64)
        flipped=[]
        with torch.inference_mode():
            for start in range(0,len(test.images),64):
                x=normalized_tensor(test.images[start:start+64]).flip(-1)
                flipped.append(model.features(x).numpy())
        flipped=np.concatenate(flipped);np.savez_compressed(feature_path,original=original,flipped=flipped)
    old=json.loads((core/'episodes.json').read_text(encoding='utf-8'))['episodes']
    old={(r['shots'],r['episode']):r for r in old}
    graph=json.loads((core/'graph_episodes.json').read_text(encoding='utf-8'))['episodes']
    graph={(r['shots'],r['episode']):r for r in graph}
    journal=out/'multiview_episodes.jsonl';records=[]
    if journal.exists():records=[json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines() if line]
    completed={(r['shots'],r['episode']):r for r in records}
    if len(completed)!=len(records):raise ValueError('Duplicate evaluation records')
    for shots in (1,5):
        cfg=lock['selected'][str(shots)];graph_cfg=GraphConfig(**cfg['config'])
        transformed=apply_transform(combine_views(original,flipped,cfg['views']),transforms[shots])
        rng=np.random.default_rng(metadata['protocol']['seed']+200+shots)
        for i in range(600):
            task=sample_episode(test.labels,rng,shots=shots,queries=15)
            support_ids=[test.ids[j] for j in task['support_indices']];query_ids=[test.ids[j] for j in task['query_indices']]
            previous=old[(shots,i)];g=graph[(shots,i)]
            if previous['support_ids']!=support_ids or previous['query_ids']!=query_ids or g['query_ids']!=query_ids:
                raise ValueError('Tasks differ from published references')
            if (shots,i) in completed:
                existing=completed[(shots,i)]
                if existing['support_ids']!=support_ids or existing['query_ids']!=query_ids:raise ValueError('Bad resume samples')
                continue
            support=transformed[task['support_indices']];query=transformed[task['query_indices']]
            started=time.perf_counter()
            result=predict(support,task['support_labels'],query,graph_cfg,cfg['assignment'])
            elapsed=time.perf_counter()-started
            accuracy=float(np.mean(result['predictions']==task['query_labels']))
            # Ablation: preserve exact features/config, only remove hard capacities.
            soft_result=result if cfg['assignment']=='soft' else predict(support,task['support_labels'],query,graph_cfg,'soft')
            entry={'shots':shots,'episode':i,'support_ids':support_ids,'query_ids':query_ids,
                   'query_labels':task['query_labels'].tolist(),'predictions':result['predictions'].tolist(),
                   'probabilities':result['probabilities'].tolist(),'accuracy':accuracy,'seconds':elapsed,
                   'soft_predictions':soft_result['predictions'].tolist(),
                   'soft_accuracy':float(np.mean(soft_result['predictions']==task['query_labels'])),
                   'baseline_accuracy':previous['methods']['baseline']['accuracy'],
                   'full_accuracy':previous['methods']['full']['accuracy'],'graph_accuracy':g['accuracy']}
            with journal.open('a',encoding='utf-8') as f:f.write(json.dumps(entry)+'\n');f.flush()
            records.append(entry)
            if (i+1)%100==0:log('multiview_test',shots=shots,episodes=i+1)
    rows=[]
    for shots in (1,5):
        tasks=[r for r in records if r['shots']==shots]
        if len(tasks)!=600:raise ValueError('Incomplete evaluation')
        values=np.array([r['accuracy'] for r in tasks])
        row={'method':'multiview','shots':shots,**summarize(values),'transductive':True,'balanced_queries':True,
             'decision_rule':lock['selected'][str(shots)]['assignment'],'views':lock['selected'][str(shots)]['views'],
             'mean_episode_seconds':float(np.mean([r['seconds'] for r in tasks])),
             'timing_scope':'inference_from_cached_features'}
        for reference,prefix in [('baseline_accuracy','paired'),('full_accuracy','paired_vs_full'),('graph_accuracy','paired_vs_graph')]:
            delta=values-np.array([r[reference] for r in tasks])
            row[prefix+'_delta' if prefix=='paired' else prefix]=float(delta.mean())
            row[prefix+'_ci95']=float(1.96*delta.std(ddof=1)/len(delta)**.5)
        row['soft_accuracy']=float(np.mean([r['soft_accuracy'] for r in tasks]))
        rows.append(row)
    # Exact selection lock and episode journal remain independently verifiable.
    write_json(out/'multiview_configuration.json',lock)
    write_json(out/'multiview_episodes.json',{'episodes':records})
    manifest={'status':'complete','study':'deterministic multi-view incremental study','completed_utc':datetime.now(timezone.utc).isoformat(),
              'base_weights_sha256':lock['checkpoint_sha256'],'rows':rows,'selection':'validation selected before test access',
              'evaluation_plan':plan,'files':{name:{'path':name,'sha256':digest_file(out/name)} for name in
                ('multiview_transform.npz','multiview_configuration.json','multiview_episodes.json')},
              'limitations':['Two-view encoding costs more than single-view inference.',
                 'Balanced capacity decisions use the declared 15-query-per-class task design, not query labels.',
                 'Assignment scores are not calibrated probability guarantees; constrained labels may not be score argmax.',
                 'Repeated development on the same public benchmark is not external validation.']}
    write_json(out/'multiview_results.json',manifest)
    log('multiview_evaluation_complete',rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',default='.local/lab04-metric');p.add_argument('--views',default='.local/lab04-multiview')
    p.add_argument('--artifacts',default='outputs/lab04');p.add_argument('--output',default='.local/lab04-multiview-test')
    p.add_argument('--data-root',default='.local/mini-imagenet');p.add_argument('--threads',type=int,default=1)
    run(p.parse_args())

if __name__=='__main__':main()
