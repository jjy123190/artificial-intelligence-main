"""Post-test diagnostics from frozen predictions: no tuning, filtering, or rescoring.

Reports relative confidence reliability, task quantiles, class recalls and
paired-difference bootstrap intervals. Never changes a model or its predictions.
"""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from .data import digest_file
from .experiment import write_json
from .artifact_audit import audit_publication


def diagnosis(records,shots,reference='graph_accuracy'):
    tasks=sorted((r for r in records if r['shots']==shots),key=lambda r:r['episode'])
    accuracies=np.array([r['accuracy'] for r in tasks]);targets=[];predictions=[];scores=[];by_class={}
    for r in tasks:
        p=np.asarray(r['probabilities']);truth=np.asarray(r['query_labels']);pred=np.asarray(r['predictions'])
        if p.shape!=(len(truth),5) or len(pred)!=len(truth):raise ValueError('Invalid frozen prediction shapes')
        targets.extend(truth.tolist());predictions.extend(pred.tolist());scores.extend(p.tolist())
        for id,true,predicted in zip(r['query_ids'],truth,pred):
            # Every occurrence is weighted exactly as in the original task protocol.
            wnid=id.split('/')[1] if len(id.split('/'))>2 else 'unknown'
            item=by_class.setdefault(wnid,{'occurrences':0,'correct':0})
            item['occurrences']+=1;item['correct']+=int(true==predicted)
    targets=np.array(targets);predictions=np.array(predictions);scores=np.array(scores)
    correct=predictions==targets;confidence=scores[np.arange(len(targets)),predictions]
    bins=[];ece=0.
    for i in range(10):
        low,high=i/10,(i+1)/10
        selected=(confidence>=low)&((confidence<high) if i<9 else (confidence<=high))
        n=int(selected.sum())
        mean=float(confidence[selected].mean()) if n else None
        acc=float(correct[selected].mean()) if n else None
        if n:ece+=n/len(targets)*abs(mean-acc)
        bins.append({'lower':low,'upper':high,'occurrences':n,'mean_score':mean,'accuracy':acc})
    expected=np.eye(5)[targets]
    brier=float(np.square(scores-expected).sum(1).mean())
    nll=float(-np.log(np.maximum(scores[np.arange(len(targets)),targets],1e-15)).mean())
    difference=np.array([r['accuracy']-r[reference] for r in tasks])
    rng=np.random.default_rng(20261124+shots)
    bootstrap=np.mean(difference[rng.integers(0,len(tasks),(4000,len(tasks)))],1)
    class_rows=[{'wnid':name,**row,'recall':row['correct']/row['occurrences']} for name,row in by_class.items()]
    return {'shots':shots,'episodes':len(tasks),'query_occurrences':len(targets),
        'accuracy':float(correct.mean()),'brier_score':brier,'negative_log_likelihood':nll,
        'expected_calibration_error_10bins':float(ece),'score_bins':bins,
        'task_accuracy_quantiles':{str(q):float(np.quantile(accuracies,q)) for q in (.05,.25,.5,.75,.95)},
        'paired_delta':float(difference.mean()),'paired_bootstrap95':[float(np.quantile(bootstrap,.025)),float(np.quantile(bootstrap,.975))],
        'episode_wins':int((difference>0).sum()),'episode_ties':int((difference==0).sum()),'episode_losses':int((difference<0).sum()),
        'class_recall':sorted(class_rows,key=lambda row:row['recall'])}


def build(directory):
    directory=Path(directory);audit_publication(directory)
    rows=[]
    if (directory/'multiview_episodes.json').is_file():
        records=json.loads((directory/'multiview_episodes.json').read_text(encoding='utf-8'))['episodes']
        rows=[diagnosis(records,n) for n in (1,5)]
    if not rows:raise ValueError('No probabilistic paired predictions available')
    return {'status':'complete','scope':'Post-test diagnostics of fixed multiview predictions; not used to choose thresholds or model parameters',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'method':'multiview','reference':'graph',
        'source_sha256':digest_file(directory/'multiview_episodes.json'),'rows':rows,
        'limitations':['Query occurrences repeat across episodes; these are task-weighted metrics, not independent images.',
                      'Confidence bins, Brier and NLL diagnose scores; no probability calibration guarantee is made.',
                      'Episode bootstrap describes sampling conditional on one trained model; not training-seed variation.',
                      'Class recalls are descriptive on this test and never used to select or exclude classes.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',default='outputs/lab04')
    args=p.parse_args();path=Path(args.artifacts)/'reliability.json';write_json(path,build(args.artifacts));print(path)

if __name__=='__main__':main()
