"""Standard-library audit for the frozen wider two-view publication.

Recomputes every score/paired interval; no NumPy, torch or prediction execution.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import math
import statistics

FILES=("wideview_episodes.json","wideview_configuration.json","wideview_transform.npz","wideview_selection.json","wideview_plan.json")


def audit_wideview(directory, core, reference, graph, multiview, wide, reference_directory=None):
    from .artifact_audit import read_json, require, validate_hash, file_hash, safe_path, close, index_rows, predictions, finite, summary, paired
    directory=Path(directory).resolve();original=Path(reference_directory or directory).resolve()
    if not (directory/"wideview_results.json").is_file(): return None
    result=read_json(directory,"wideview_results.json")
    require(wide is not None and graph is not None and multiview is not None,"Wide-view references missing")
    require(result.get("status")=="complete","Wide-view results incomplete")
    require(set(result.get("files",{}))==set(FILES),"Unexpected wide-view file set")
    for name in FILES:
        entry=result['files'][name];require(entry['path']==name,"Unsafe wide-view manifest path")
        validate_hash(directory,name,entry['sha256'])
    require(result["base_weights_sha256"]==wide["candidate_checkpoint_sha256"],"Wide-view checkpoint differs from frozen wider backbone")
    lock=read_json(directory,"wideview_configuration.json");selected=read_json(directory,"wideview_selection.json");plan=read_json(directory,"wideview_plan.json")
    require(lock['status']=='locked' and lock['test_accessed'] is False,"Wide-view selection not locked")
    require(selected['status']=='complete' and selected['test_accessed'] is False and plan['test_accessed'] is False,"Wide-view selection accessed test data")
    require(lock['base_weights_sha256']==result['base_weights_sha256'],"Wide-view selected model mismatch")
    require(lock['selection_sha256']==file_hash(directory/'wideview_selection.json'),"Wide-view selection changed")
    require(lock['plan_sha256']==file_hash(directory/'wideview_plan.json')==selected['plan_sha256'],"Wide-view selection plan changed")
    require(lock['transform_sha256']==file_hash(directory/'wideview_transform.npz')==selected['transform_sha256'],"Wide-view transform changed")
    require(datetime.fromisoformat(selected['completed_utc'])<=datetime.fromisoformat(lock['locked_utc'])<=datetime.fromisoformat(result['completed_utc']),"Wide-view test chronology")
    for shot in ('1','5'):
        winner=selected['shots'][shot]['selected']
        require(selected['shots'][shot]['promising'] is True,"Validation gate not passed")
        require(lock['selected'][shot]=={**winner['candidate'],'config':winner['config']},"Wide-view inference differs from validated config")
        require(winner['candidate'] in plan['candidates'],"Candidate not preregistered")
    evaluation=result['evaluation_plan']
    require(evaluation['seed']==core['protocol']['seed'] and evaluation['episodes_per_shot']==core['protocol']['test_episodes_per_shot']==600 and evaluation['query_per_class']==15,"Wide-view evaluation protocol mismatch")
    require(evaluation['checkpoint_sha256']==result['base_weights_sha256'] and evaluation['selection_sha256']==lock['selection_sha256'] and evaluation['transform_sha256']==lock['transform_sha256'],"Wide-view evaluation lock mismatch")
    require(evaluation['test_pool_sha256']==core['protocol']['test_selection_sha256'],"Wide-view test pool mismatch")
    expected_references={'results.json','episodes.json','graph_episodes.json','multiview_episodes.json','wide_episodes.json','wide_results.json'}
    require(set(evaluation['reference_files'])==expected_references,"Wide-view reference set mismatch")
    for name,digest in evaluation['reference_files'].items():validate_hash(original,name,digest)
    wide_records={(r['shots'],r['episode']):r for r in read_json(original,'wide_episodes.json')['episodes']}
    # Bind the original pre-test lock and recompute selection, not just its
    # summary flags. Temporary audits can read the same frozen journal from
    # their explicit reference directory; no arbitrary manifest path is used.
    journal_name='next_study/wide_views_20261125/scores.jsonl'
    lock_name='next_study/wide_views_20261125/selection_lock.json'
    evidence=directory if (directory/journal_name).is_file() else original
    selection_lock=read_json(evidence,lock_name)
    require(evaluation['selection_lock_sha256']==result['selection_lock_sha256'],'Wide-view test selection lock mismatch')
    validate_hash(evidence,lock_name,result['selection_lock_sha256'])
    require(selection_lock['status']=='locked' and selection_lock['test_accessed'] is False,'Wide-view original selection lock invalid')
    require(selection_lock['selection_sha256']==lock['selection_sha256'] and selection_lock['plan_sha256']==lock['plan_sha256'] and
            selection_lock['transform_sha256']==lock['transform_sha256'] and selection_lock['checkpoint_sha256']==result['base_weights_sha256'] and
            selection_lock['locked_utc']==lock['locked_utc'],'Wide-view original lock differs from runtime config')
    journal=safe_path(evidence,journal_name)
    require(journal.stat().st_size<=8*1024*1024,'Validation score journal too large')
    from .selection_audit import validate_selection
    validated=validate_selection(plan,selected,journal.read_text(encoding='utf-8').splitlines())
    require(all(row['promising'] for row in validated['shots'].values()),'Wide-view original validation gate fails')
    prior_lock=read_json(original,'wide_configuration.json')
    for shot in ('1','5'):
        require(lock['selected'][shot]['config']==prior_lock['selected'][shot]['config'],'Wide-view propagation config differs from frozen reference')
    records=read_json(directory,'wideview_episodes.json')['episodes'];count=core['protocol']['test_episodes_per_shot']
    require(len(records)==count*2,'Incomplete wide-view tasks')
    rows=index_rows(result['rows'],{'wideview'},count);index={}
    for record in records:
        shots,number=record['shots'],record['episode'];key=(shots,number)
        require(type(shots) is int and type(number) is int and key in reference and key not in index,'Invalid/duplicate wide-view task')
        old=reference[key]
        require(record['support_ids']==old['support_ids'] and record['query_ids']==old['query_ids'] and record['query_labels']==old['query_labels'],'Unpaired wide-view samples or labels')
        pred=predictions(record,record['query_labels'],75)
        require(len(record['probabilities'])==75,'Invalid wide-view probability rows')
        for predicted,scores in zip(pred,record['probabilities']):
            require(isinstance(scores,list) and len(scores)==5 and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in scores),'Invalid wide-view scores')
            close(sum(scores),1,'wide-view probability sum')
            require(predicted==max(range(5),key=lambda c:scores[c]),'Wide-view label differs from maximum score')
        close(record['baseline_accuracy'],old['scores']['baseline'],'wideview baseline reference')
        close(record['full_accuracy'],old['scores']['full'],'wideview full reference')
        for name,entries in (('graph',graph),('multiview',multiview),('wide',wide_records)):
            close(record[name+'_accuracy'],entries[key]['accuracy'],'wideview '+name+' reference')
        require(finite(record['seconds'],'wideview duration')>=0,'Negative wideview duration')
        index[key]=record
    for shot in (1,5):
        tasks=[index[(shot,i)] for i in range(count)];row=rows[('wideview',shot)];values=[r['accuracy'] for r in tasks]
        summary(values,row,count,'wideview')
        for name in ('baseline','full','graph','multiview','wide'):
            mean='paired_delta' if name=='baseline' else 'paired_vs_'+name
            ci='paired_ci95' if name=='baseline' else 'paired_vs_'+name+'_ci95'
            paired(values,[r[name+'_accuracy'] for r in tasks],row,mean,ci)
        close(row['mean_episode_seconds'],statistics.mean(r['seconds'] for r in tasks),'wideview mean duration')
    return result


def verify(directory, reference_directory=None):
    from .artifact_audit import audit_core,audit_extension,audit_wide
    original=Path(reference_directory or directory).resolve()
    core,reference,_=audit_core(original)
    graph,graph_records=audit_extension(original,'graph',core,reference)
    multiview,multi_records=audit_extension(original,'multiview',core,reference,graph_records)
    wide=audit_wide(original,core,reference,graph_records,multi_records)
    result=audit_wideview(directory,core,reference,graph_records,multi_records,wide,original)
    if result is None:raise ValueError('Wide-view publication missing')
    return {'verified':True,'episodes_recomputed':1200,'rows':result['rows']}


def main():
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',default='.local/lab04-wideview-test-20261125');p.add_argument('--reference',default='outputs/lab04')
    args=p.parse_args();print(json.dumps(verify(args.artifacts,args.reference),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
