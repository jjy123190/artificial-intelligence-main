"""Verified optional multi-view study artifacts, separate from the fixed v2 study."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from datetime import datetime
import json,statistics
import numpy as np
from .data import digest_file
from .graph_calibration import GraphConfig
from .multiview import MultiViewConfig


NAMES=('multiview_transform.npz','multiview_configuration.json','multiview_episodes.json')


def read_extension(directory):
    directory=Path(directory).resolve();path=directory/'multiview_results.json'
    if not path.exists():return None
    raw=path.read_bytes();result=json.loads(raw)
    if result.get('status')!='complete':raise ValueError('Incomplete multi-view extension')
    if set(result.get('files',{}))!=set(NAMES):raise ValueError('Unexpected multi-view artifact set')
    observations=[]
    for name in (*NAMES,'aligned.pt'):
        candidate=(directory/name).resolve()
        if not candidate.is_relative_to(directory) or candidate.is_symlink():raise ValueError('Unsafe multi-view artifact')
        info=candidate.stat();observations.append((name,info.st_size,info.st_mtime_ns))
    _check_hashes(str(directory),raw,tuple(observations))
    return result


@lru_cache(maxsize=3)
def _check_hashes(directory,raw,observations):
    directory=Path(directory);result=json.loads(raw)
    if digest_file(directory/'aligned.pt')!=result['base_weights_sha256']:raise ValueError('Multi-view backbone mismatch')
    for name in NAMES:
        if result['files'][name]['path']!=name or digest_file(directory/name)!=result['files'][name]['sha256']:
            raise ValueError('Multi-view artifact checksum mismatch: '+name)


def inference_inputs(directory,shots):
    result=read_extension(directory)
    if result is None:raise ValueError('Multi-view strategy is not installed')
    directory=Path(directory)
    lock=json.loads((directory/'multiview_configuration.json').read_text(encoding='utf-8'))
    config=lock['selected'][str(shots)]
    with np.load(directory/'multiview_transform.npz',allow_pickle=False) as source:
        prefix=f'shot{shots}_';transform={k[len(prefix):]:source[k] for k in source.files if k.startswith(prefix)}
    return transform,GraphConfig(**config['config']),MultiViewConfig(config['views'],config['assignment'])


def verify_extension(directory):
    from .verify import require,close
    directory=Path(directory);result=read_extension(directory)
    if result is None:return None
    lock=json.loads((directory/'multiview_configuration.json').read_text(encoding='utf-8'))
    require(lock['status']=='locked' and lock['test_accessed'] is False,'Missing validation lock')
    require(lock['checkpoint_sha256']==result['base_weights_sha256'],'Selection/backbone mismatch')
    require(datetime.fromisoformat(lock['locked_utc'])<=datetime.fromisoformat(result['completed_utc']),'Incorrect test chronology')
    require(digest_file(directory/'multiview_configuration.json')==result['evaluation_plan']['selection_lock_sha256'], 'Selection bytes changed')
    records=json.loads((directory/'multiview_episodes.json').read_text(encoding='utf-8'))['episodes']
    reference=json.loads((directory/'episodes.json').read_text(encoding='utf-8'))['episodes']
    reference={(r['shots'],r['episode']):r for r in reference}
    graph=json.loads((directory/'graph_episodes.json').read_text(encoding='utf-8'))['episodes']
    graph={(r['shots'],r['episode']):r for r in graph}
    require(len(records)==1200,'Incomplete multi-view tasks')
    require(len(result['rows'])==2,'Unexpected multi-view result rows')
    for shots in (1,5):
        tasks=sorted((r for r in records if r['shots']==shots),key=lambda r:r['episode'])
        require([r['episode'] for r in tasks]==list(range(600)),'Bad multi-view episode IDs')
        accuracy=[];delta=[];vs_full=[];vs_graph=[]
        mode=lock['selected'][str(shots)]['assignment']
        for r in tasks:
            old=reference[(shots,r['episode'])];g=graph[(shots,r['episode'])]
            require(r['support_ids']==old['support_ids'] and r['query_ids']==old['query_ids'],'Unpaired multi-view samples')
            require(r['query_labels']==old['query_labels'],'Changed score labels')
            require(len(r['predictions'])==len(r['query_labels'])==75,'Invalid prediction length')
            require(all(type(p) is int and 0<=p<5 for p in r['predictions']),'Invalid prediction classes')
            if mode=='balanced_capacity':require(all(r['predictions'].count(c)==15 for c in range(5)),'Capacity violated')
            p=np.asarray(r['probabilities']);require(p.shape==(75,5) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all(),'Invalid model scores')
            require(np.allclose(p.sum(1),1,atol=1e-9),'Scores not normalized')
            if mode=='soft':require(np.array_equal(p.argmax(1),r['predictions']),'Soft prediction/score mismatch')
            score=sum(a==b for a,b in zip(r['predictions'],r['query_labels']))/75
            close(score,r['accuracy'],'multi-view correct count')
            close(r['baseline_accuracy'],old['methods']['baseline']['accuracy'],'baseline reference')
            close(r['graph_accuracy'],g['accuracy'],'graph reference')
            soft=sum(a==b for a,b in zip(r['soft_predictions'],r['query_labels']))/75
            close(soft,r['soft_accuracy'],'soft ablation score')
            require(np.isfinite(r['seconds']) and r['seconds']>=0,'Bad inference duration')
            accuracy.append(score);delta.append(score-r['baseline_accuracy'])
            vs_full.append(score-old['methods']['full']['accuracy']);vs_graph.append(score-g['accuracy'])
        row=next(r for r in result['rows'] if r['shots']==shots)
        for values,mean_key,ci_key in [(accuracy,'accuracy','ci95'),(delta,'paired_delta','paired_ci95'),
            (vs_full,'paired_vs_full','paired_vs_full_ci95'),(vs_graph,'paired_vs_graph','paired_vs_graph_ci95')]:
            close(row[mean_key],statistics.mean(values),mean_key)
            close(row[ci_key],1.96*statistics.stdev(values)/600**.5,ci_key)
        close(row['soft_accuracy'],statistics.mean(r['soft_accuracy'] for r in tasks),'soft mean')
        close(row['mean_episode_seconds'],statistics.mean(r['seconds'] for r in tasks),'timing mean')
    return {'verified':True,'episodes_recomputed':1200,'rows':2}
