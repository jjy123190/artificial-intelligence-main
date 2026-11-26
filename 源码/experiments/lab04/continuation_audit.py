"""Independent audit of completed continuation and fixed validation; never reads test.

This is a gate, not an optimizer. Recomputes epoch selection and all validation
predictions/statistics; incomplete work cannot silently become test-eligible.
"""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(condition,message):
    if not condition:raise ValueError('Continuation audit: '+message)


def read(path):
    def bad(value):raise ValueError('Invalid JSON numeric literal: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=bad)


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def finite(value):return type(value) in (int,float) and math.isfinite(value)
def close(actual,expected,name):require(finite(actual) and math.isclose(actual,expected,rel_tol=1e-8,abs_tol=1e-10),'statistic mismatch '+name)
def half(values):return 1.96*statistics.stdev(values)/len(values)**.5 if len(values)>1 else 0.


def audit_epoch_selection(plan,training,initial):
    require(plan.get('test_accessed') is False and training.get('test_accessed') is False,'test used during training')
    require(training.get('status')=='complete','training incomplete')
    count=plan['epochs'];require(type(count) is int and 1<=count<=32,'invalid registered budget')
    history=training['rows'];require(len(history)==count and [r['epoch'] for r in history]==list(range(1,count+1)),'incomplete/reordered epochs')
    def score(row):
        for key in ('1shot','5shot','mean'):require(finite(row[key]) and 0<=row[key]<=1,'invalid validation score')
        close(row['mean'],(row['1shot']+row['5shot'])/2,'validation mean');return row['mean']
    best=score(initial);best_epoch=0
    for row in history:
        require(all(finite(row[k]) and row[k]>=0 for k in ('ce_loss','metric_loss','seconds')),'invalid training loss/duration')
        require(finite(row['training_accuracy']) and 0<=row['training_accuracy']<=1,'invalid training accuracy')
        mean=score(row['validation'])
        if mean>best:best,best_epoch=mean,row['epoch']
    require(training['best_epoch']==best_epoch,'selected epoch does not follow first strict validation maximum')
    close(training['best_mean'],best,'best validation score')
    return best_epoch


def audit_tasks(records,reported,episodes):
    require(type(episodes) is int and episodes>=2,'invalid validation task count')
    require(len(records)==episodes*2,'incomplete fixed validation task set');index={}
    for record in records:
        shot,number=record['shots'],record['episode'];key=(shot,number)
        require(type(shot) is int and shot in (1,5) and type(number) is int and 0<=number<episodes and key not in index,'duplicate/invalid validation task')
        support,query=record['support_ids'],record['query_ids']
        require(isinstance(support,list) and isinstance(query,list) and len(support)==shot*5 and len(query)==75,'invalid support/query lengths')
        require(all(isinstance(i,str) and i.startswith('val/') for i in support+query),'non-validation image in candidate scoring')
        require(len(set(support))==len(support) and len(set(query))==75 and not set(support)&set(query),'support/query duplicate or overlap')
        truth=record['query_labels'];require(isinstance(truth,list) and len(truth)==75 and all(type(y) is int and 0<=y<5 for y in truth),'invalid query scoring labels')
        require(all(truth.count(i)==15 for i in range(5)),'query class imbalance')
        for key_pred,key_score in (('predictions','accuracy'),('reference_predictions','reference_accuracy')):
            pred=record[key_pred];require(isinstance(pred,list) and len(pred)==75 and all(type(y) is int and 0<=y<5 for y in pred),'invalid prediction')
            close(record[key_score],sum(a==b for a,b in zip(pred,truth))/75,key_score)
        index[key]=record
    result={}
    def summary(values,row,name):
        close(row['accuracy'],statistics.mean(values),name+' mean');close(row['ci95'],half(values),name+' ci95')
        require(type(row['episodes']) is int and row['episodes']==episodes,'wrong summary count')
    for shot in (1,5):
        tasks=[index[(shot,i)] for i in range(episodes)];candidate=[r['accuracy'] for r in tasks];control=[r['reference_accuracy'] for r in tasks]
        row=reported[str(shot)];summary(candidate,row['candidate'],'candidate');summary(control,row['reference'],'reference')
        differences=[a-b for a,b in zip(candidate,control)];mean=statistics.mean(differences);ci=half(differences)
        pair=row['paired'];close(pair['mean'],mean,'paired mean');close(pair['ci95'],ci,'paired ci95')
        for key,value in (('wins',sum(d>0 for d in differences)),('ties',sum(d==0 for d in differences)),('losses',sum(d<0 for d in differences))):
            require(type(pair[key]) is int and pair[key]==value,'wrong paired '+key)
        expected=mean-ci>0;require(type(row['promising']) is bool and row['promising']==expected,'validation gate changed')
        result[str(shot)]={'accuracy':statistics.mean(candidate),'reference_accuracy':statistics.mean(control),'paired_delta':mean,'paired_ci95':ci,'promising':expected}
    return result


def audit(candidate,validation,artifacts):
    candidate=Path(candidate).resolve();validation=Path(validation).resolve();artifacts=Path(artifacts).resolve()
    plan=read(candidate/'plan.json');training=read(candidate/'training.json');initial=read(candidate/'initial_validation.json')
    selected=audit_epoch_selection(plan,training,initial)
    downstream=read(validation/'validation_plan.json');result=read(validation/'validation.json');checkpoint=read(validation/'checkpoint_lock.json')
    require(downstream.get('test_accessed') is False and result.get('status')=='complete' and result.get('test_accessed') is False,'fixed validation incomplete/test used')
    require(downstream['candidate_training_plan_sha256']==digest(candidate/'plan.json'),'training plan changed')
    require(checkpoint['training_sha256']==digest(candidate/'training.json') and checkpoint['checkpoint_sha256']==digest(candidate/'best.pt'),'training/selected model changed')
    require(checkpoint['plan_sha256']==digest(validation/'validation_plan.json') and checkpoint['best_epoch']==selected and checkpoint['test_accessed'] is False,'checkpoint freeze changed')
    require(result['checkpoint_lock']==checkpoint,'result belongs to another checkpoint')
    for source in (plan,downstream):
        for name,sha in source['source_hashes'].items():
            path=Path(__file__).with_name(name);require(path.parent==Path(__file__).parent and digest(path)==sha,'registered source changed: '+name)
    require(downstream['reference_weights_sha256']==digest(artifacts/'wide.pt'),'reference model changed')
    require(downstream['reference_configuration_sha256']==digest(artifacts/'wideview_configuration.json') and downstream['reference_transform_sha256']==digest(artifacts/'wideview_transform.npz'),'reference transform/config changed')
    require(downstream['fixed_configurations']==read(artifacts/'wideview_configuration.json')['selected'],'new hyperparameters substituted after registration')
    if selected==0:
        require(result['selected_original'] is True and result['promising'] is False,'epoch zero must not claim new improvement')
        return {'verified':True,'selected_epoch':0,'test_eligible':False,'reason':'Original model retained by epoch selection'}
    lock=read(validation/'selection_lock.json')
    require(lock['status']=='locked' and lock['test_accessed'] is False,'missing post-validation lock')
    for file,key in (('validation.json','validation_sha256'),('validation_plan.json','plan_sha256'),('transform.npz','transform_sha256')):
        require(digest(validation/file)==lock[key],'validation lock mismatch: '+file)
    require(lock['checkpoint_sha256']==checkpoint['checkpoint_sha256'],'validation/checkpoint weights differ')
    require(datetime.fromisoformat(training['completed_utc'])<=datetime.fromisoformat(result['completed_utc'])<=datetime.fromisoformat(lock['locked_utc']),'wrong validation chronology')
    journal=validation/'validation_tasks.jsonl';require(result['journal_sha256']==digest(journal),'validation prediction journal changed')
    require(result['transform_sha256']==lock['transform_sha256'],'validation transform mismatch')
    rows=audit_tasks([json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()],result['shots'],downstream['episodes_per_shot'])
    eligible=all(row['promising'] for row in rows.values());require(type(result['promising']) is bool and result['promising']==eligible,'overall validation gate mismatch')
    return {'verified':True,'selected_epoch':selected,'additional_training_epochs':len(training['rows']),'test_eligible':eligible,'shots':rows}


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',default='.local/lab04-continuation-20261125')
    p.add_argument('--validation',default='.local/lab04-continuation-validation-20261125');p.add_argument('--artifacts',default='outputs/lab04')
    a=p.parse_args();print(json.dumps(audit(a.candidate,a.validation,a.artifacts),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
