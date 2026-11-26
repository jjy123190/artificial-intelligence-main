"""Independent audit for the completed larger self-trained model study."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json,statistics,math
from .data import digest_file
from .artifact_audit import require,close,task_ids,index_rows,predictions,summary,paired


def verify(directory,core_directory,candidate_directory):
    directory,core,candidate=Path(directory),Path(core_directory),Path(candidate_directory)
    result=json.loads((directory/'wide_results.json').read_text(encoding='utf-8'))
    require(result['status']=='complete','Wide result incomplete')
    require(result['evaluation_plan']['validation_reference_sha256']==digest_file(candidate/'validation_selection.json'),'Wide reference changed')
    expected={'wide_transform.npz','wide_configuration.json','wide_episodes.json'}
    require(set(result['files'])==expected,'Wide dependencies incomplete')
    for name,entry in result['files'].items():
        require(entry['path']==name and not(directory/name).is_symlink(),'Unsafe wide dependency')
        require(digest_file(directory/name)==entry['sha256'],'Wide file hash mismatch: '+name)
    lock=json.loads((directory/'wide_configuration.json').read_text(encoding='utf-8'))
    require(lock['status']=='locked' and lock['test_accessed'] is False,'No pre-test wide lock')
    require(digest_file(directory/'wide_configuration.json')==result['selection_lock_sha256'],'Wide selection lock changed')
    require(datetime.fromisoformat(lock['locked_utc'])<=datetime.fromisoformat(result['completed_utc']),'Wide chronology')
    require(digest_file(candidate/'best.pt')==result['candidate_checkpoint_sha256']==lock['checkpoint_sha256'],'Wide checkpoint mismatch')
    require(digest_file(candidate/'validation_selection.json')==lock['selection_sha256'],'Wide validation selection changed')
    require(digest_file(candidate/'plan.json')==lock['training_plan_sha256'],'Wide training plan changed')
    for name,digest in result['evaluation_plan']['source_sha256'].items():
        source=Path(__file__).with_name(name)
        require(source.parent==Path(__file__).parent and digest_file(source)==digest,'Wide calculation code changed')
    records=json.loads((directory/'wide_episodes.json').read_text())['episodes']
    require(len(records)==1200,'Incomplete wide task set')
    def refs(name):
        value=json.loads((core/name).read_text(encoding='utf-8'))['episodes']
        return {(r['shots'],r['episode']):r for r in value}
    base,graph,multi=refs('episodes.json'),refs('graph_episodes.json'),refs('multiview_episodes.json')
    rows=index_rows(result['rows'],{'wide'},600)
    for shots in (1,5):
        tasks=sorted((r for r in records if r['shots']==shots),key=lambda r:r['episode'])
        require([r['episode'] for r in tasks]==list(range(600)),'Wide task IDs invalid')
        values=[]
        for r in tasks:
            key=(shots,r['episode']);old=base[key]
            task_ids(r,shots,75)
            require(r['support_ids']==old['support_ids'] and r['query_ids']==old['query_ids'],'Unpaired wider study')
            require(r['query_labels']==old['query_labels'],'Wide score labels changed')
            pred=predictions(r,r['query_labels'],75)
            predictions(r,r['query_labels'],75,'raw_predictions','raw_accuracy')
            probs=r['probabilities'];require(len(probs)==75,'Bad wide score count')
            for y,p in zip(pred,probs):
                require(len(p)==5 and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in p),'Invalid wide score')
                close(sum(p),1,'wide probability sum');require(y==max(range(5),key=lambda c:p[c]),'Wide prediction/score mismatch')
            for field,target in [('baseline_accuracy',old['methods']['baseline']['accuracy']),('full_accuracy',old['methods']['full']['accuracy']),
                                 ('graph_accuracy',graph[key]['accuracy']),('multiview_accuracy',multi[key]['accuracy'])]:
                close(r[field],target,field)
            values.append(r['accuracy'])
        row=rows[('wide',shots)];summary(values,row,600,'wide')
        for field,prefix in [('baseline_accuracy','paired'),('full_accuracy','paired_vs_full'),('graph_accuracy','paired_vs_graph'),('multiview_accuracy','paired_vs_multiview')]:
            paired(values,[r[field] for r in tasks],row,'paired_delta' if prefix=='paired' else prefix,prefix+'_ci95')
        close(row['raw_accuracy'],statistics.mean(r['raw_accuracy'] for r in tasks),'raw wider baseline mean')
        close(row['mean_episode_seconds'],statistics.mean(r['seconds'] for r in tasks),'wide duration')
    return {'verified':True,'episodes_recomputed':1200,'rows':2}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',default='.local/lab04-wide-test')
    p.add_argument('--core',default='outputs/lab04');p.add_argument('--candidate',default='.local/lab04-v3-wide')
    args=p.parse_args();print(json.dumps(verify(args.artifacts,args.core,args.candidate),ensure_ascii=False,indent=2))
