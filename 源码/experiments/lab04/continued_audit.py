"""Dependency-light audit for an optional continued-training result.

Training, validation and original fixed test tasks are recomputed independently;
checks do not treat a positive flag or updated manifest as experimental proof.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import math
import statistics

FILES=('continued.pt','continued_transform.npz','continued_training.json','continued_training_plan.json','continued_initial_validation.json',
       'continued_validation.json','continued_validation_plan.json','continued_validation_tasks.jsonl','continued_checkpoint_lock.json',
       'continued_selection_lock.json','continued_configuration.json','continued_episodes.json')


def audit_continued(directory,core,reference,wide,wideview,reference_directory=None):
    from .artifact_audit import read_json,safe_path,require,validate_hash,file_hash,close,index_rows,predictions,finite,summary,paired
    from .continuation_audit import audit_epoch_selection,audit_tasks
    directory=Path(directory).resolve();original=Path(reference_directory or directory).resolve()
    if not(directory/'continued_results.json').is_file():return None
    result=read_json(directory,'continued_results.json')
    require(wide is not None and wideview is not None,'Continued study requires frozen wide/wideview references')
    require(result.get('status')=='complete' and set(result.get('files',{}))==set(FILES),'Incomplete continued publication')
    for name in FILES:
        entry=result['files'][name];require(entry['path']==name,'Unsafe continued file path');validate_hash(directory,name,entry['sha256'])
    require(result['checkpoint_sha256']==result['files']['continued.pt']['sha256'],'Continued checkpoint mismatch')
    training=read_json(directory,'continued_training.json');training_plan=read_json(directory,'continued_training_plan.json')
    initial=read_json(directory,'continued_initial_validation.json');chosen=audit_epoch_selection(training_plan,training,initial)
    require(chosen>0,'Epoch zero is not a new continued model')
    require(training_plan['initial_sha256']==wide['candidate_checkpoint_sha256'],'Continuation did not start from frozen wider weights')
    require(result['model']['widths']==wide['model']['widths'] and result['model']['parameters']==wide['model']['parameters'],'Continued study changed model capacity')
    require(result['training']['same_capacity'] is True and result['training']['additional_epochs']==len(training['rows']) and result['training']['selected_additional_epoch']==chosen,'Continued budget/selected epoch mismatch')
    require(result['training']['original_wide_epochs']==wide['training']['epochs_completed'],'Original training budget mismatch')
    validation=read_json(directory,'continued_validation.json');validation_plan=read_json(directory,'continued_validation_plan.json')
    checkpoint=read_json(directory,'continued_checkpoint_lock.json');selection=read_json(directory,'continued_selection_lock.json');configuration=read_json(directory,'continued_configuration.json')
    for record in (validation,validation_plan,checkpoint,selection,configuration):require(record.get('test_accessed') is False,'Test used during continuation selection')
    require(validation['status']=='complete' and selection['status']==configuration['status']=='locked','Unfinished continued selection')
    require(validation_plan['candidate_training_plan_sha256']==file_hash(directory/'continued_training_plan.json'),'Continued training plan changed')
    require(checkpoint['training_sha256']==file_hash(directory/'continued_training.json') and checkpoint['plan_sha256']==file_hash(directory/'continued_validation_plan.json'),'Continued checkpoint lock changed')
    require(checkpoint['checkpoint_sha256']==result['checkpoint_sha256'] and checkpoint['best_epoch']==chosen and validation['checkpoint_lock']==checkpoint,'Different validation checkpoint')
    for name,key in (('continued_validation.json','validation_sha256'),('continued_validation_plan.json','plan_sha256'),('continued_transform.npz','transform_sha256')):
        require(file_hash(directory/name)==selection[key],'Continued selection lock mismatch: '+name)
    require(selection['checkpoint_sha256']==result['checkpoint_sha256'] and configuration['checkpoint_sha256']==result['checkpoint_sha256'],'Continuation inference model differs from selected model')
    require(configuration['transform_sha256']==selection['transform_sha256']==validation['transform_sha256'],'Continuation inference transform differs from validation')
    require(configuration['selection_lock_sha256']==file_hash(directory/'continued_selection_lock.json'),'Continued runtime selection lock differs')
    require(configuration['locked_utc']==selection['locked_utc'],'Continued configuration date differs')
    frozen=read_json(original,'wideview_configuration.json')
    require(configuration['selected']==validation_plan['fixed_configurations']==frozen['selected'],'Classifier parameters changed after preregistration')
    require(validation_plan['reference_weights_sha256']==wide['candidate_checkpoint_sha256'],'Validation reference weights changed')
    validate_hash(original,'wideview_configuration.json',validation_plan['reference_configuration_sha256'])
    validate_hash(original,'wideview_transform.npz',validation_plan['reference_transform_sha256'])
    journal=safe_path(directory,'continued_validation_tasks.jsonl');require(journal.stat().st_size<=32*1024*1024,'Validation journal too large')
    require(file_hash(journal)==validation['journal_sha256'],'Continued validation predictions changed')
    import json
    values=audit_tasks([json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()],validation['shots'],validation_plan['episodes_per_shot'])
    require(validation['promising'] is True and all(row['promising'] for row in values.values()),'Continued candidate did not pass the fixed validation gate')
    require(datetime.fromisoformat(training['completed_utc'])<=datetime.fromisoformat(validation['completed_utc'])<=datetime.fromisoformat(selection['locked_utc'])<=datetime.fromisoformat(result['completed_utc']),'Invalid continued chronology')
    plan=result['evaluation_plan']
    require(plan['checkpoint_sha256']==result['checkpoint_sha256'] and plan['selected_configurations']==configuration['selected'],'Continued test/model configuration mismatch')
    for filename,key in (('continued_selection_lock.json','selection_lock_sha256'),('continued_validation.json','validation_sha256'),('continued_validation_plan.json','validation_plan_sha256'),('continued_training_plan.json','training_plan_sha256'),('continued_training.json','training_sha256'),('continued_transform.npz','transform_sha256')):
        require(file_hash(directory/filename)==plan[key],'Continued test evidence changed: '+filename)
    require(plan['seed']==core['protocol']['seed'] and plan['episodes_per_shot']==core['protocol']['test_episodes_per_shot']==600 and plan['queries_per_class']==15,'Continued test protocol changed')
    require(plan['test_pool_sha256']==core['protocol']['test_selection_sha256'],'Continued test pool changed')
    expected={'results.json','episodes.json','graph_episodes.json','multiview_episodes.json','wide_episodes.json','wideview_episodes.json','wideview_results.json'}
    require(set(plan['reference_files'])==expected,'Continued reference files incomplete')
    for name,digest in plan['reference_files'].items():validate_hash(original,name,digest)
    controls={}
    for name in ('graph','multiview','wide','wideview'):
        records=read_json(original,name+'_episodes.json')['episodes'];controls[name]={(r['shots'],r['episode']):r['accuracy'] for r in records}
    records=read_json(directory,'continued_episodes.json')['episodes'];count=core['protocol']['test_episodes_per_shot'];require(len(records)==count*2,'Continued test tasks incomplete')
    rows=index_rows(result['rows'],{'continued'},count);index={}
    for record in records:
        shot,number=record['shots'],record['episode'];key=(shot,number)
        require(type(shot) is int and type(number) is int and key in reference and key not in index,'Invalid/duplicate continued task')
        old=reference[key];require(record['support_ids']==old['support_ids'] and record['query_ids']==old['query_ids'] and record['query_labels']==old['query_labels'],'Continued support/query/labels not paired')
        pred=predictions(record,record['query_labels'],75)
        require(isinstance(record['probabilities'],list) and len(record['probabilities'])==75,'Continued probability rows incomplete')
        for y,scores in zip(pred,record['probabilities']):
            require(isinstance(scores,list) and len(scores)==5 and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in scores),'Invalid continued scores')
            close(sum(scores),1,'continued probability sum');require(y==max(range(5),key=lambda c:scores[c]),'Continued prediction differs from argmax')
        close(record['baseline_accuracy'],old['scores']['baseline'],'continued baseline');close(record['full_accuracy'],old['scores']['full'],'continued full reference')
        for name,source in controls.items():close(record[name+'_accuracy'],source[key],'continued '+name+' reference')
        require(finite(record['seconds'],'continued duration')>=0,'Negative continued duration');index[key]=record
    for shot in (1,5):
        tasks=[index[(shot,i)] for i in range(count)];row=rows[('continued',shot)];accuracy=[r['accuracy'] for r in tasks]
        summary(accuracy,row,count,'continued')
        for name in ('baseline','full','graph','multiview','wide','wideview'):
            mean='paired_delta' if name=='baseline' else 'paired_vs_'+name;ci='paired_ci95' if name=='baseline' else mean+'_ci95'
            paired(accuracy,[r[name+'_accuracy'] for r in tasks],row,mean,ci)
        close(row['mean_episode_seconds'],statistics.mean(r['seconds'] for r in tasks),'continued mean duration')
    return result


def verify(directory,reference_directory):
    from .artifact_audit import audit_core,audit_extension,audit_wide
    from .wideview_audit import audit_wideview
    original=Path(reference_directory).resolve();core,reference,_=audit_core(original)
    graph,graph_records=audit_extension(original,'graph',core,reference);multi,multi_records=audit_extension(original,'multiview',core,reference,graph_records)
    wide=audit_wide(original,core,reference,graph_records,multi_records);wideview=audit_wideview(original,core,reference,graph_records,multi_records,wide)
    value=audit_continued(directory,core,reference,wide,wideview,original)
    if value is None:raise ValueError('No continued test result')
    return {'verified':True,'episodes_recomputed':1200,'training':value['training'],'rows':value['rows']}


def main():
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',default='.local/lab04-continued-test-20261125');p.add_argument('--reference',default='outputs/lab04')
    a=p.parse_args();print(json.dumps(verify(a.artifacts,a.reference),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
