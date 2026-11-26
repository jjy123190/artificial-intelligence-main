"""Locked test of the validation-selected graph classifier on unchanged v2 weights.

This is a separate algorithm track. Future wider-backbone selection may only
use validation results, never this test. The official website is not modified.
"""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from .data import load_split,sample_episode,digest_file,selection_digest
from .experiment import load_checkpoint,extract_features,log,summarize,write_json
from .refine import read_cache
from .graph_calibration import centered_whitening,apply_transform,GraphConfig,predict


def run(args):
    source=Path(args.source);study=Path(args.study);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    selection=json.loads((study/'validation.json').read_text())
    if selection['status']!='complete':raise ValueError('Validation selection incomplete')
    official=json.loads((source/'results.json').read_text(encoding='utf-8'))
    selected={str(n):{k:selection['shots'][str(n)]['selected'][k] for k in ('space','config')} for n in (1,5)}
    lock={'selected':selected,'weights_sha256':digest_file(source/'aligned.pt'),
          'validation_sha256':digest_file(study/'validation.json'),'episodes':600,'seed':official['protocol']['seed'],
          'test_accessed':False,'protocol':'same full test pool and seeded tasks as v2',
          'code_sha256':{n:digest_file(Path(__file__).with_name(n)) for n in ('graph_calibration.py','evaluate_graph_candidate.py')}}
    lock_file=out/'locked.json'
    if lock_file.exists() and json.loads(lock_file.read_text())!=lock:raise ValueError('Lock changed')
    write_json(lock_file,lock)
    train=read_cache(args.features,'aligned','train')
    spaces={n:centered_whitening(train['features'],**selected[str(n)]['space']) for n in (1,5)}
    torch.set_num_threads(args.threads)
    test=load_split(args.data_root,'test',max_per_class=600,seed=official['protocol']['seed']+4)
    if selection_digest(test)!=official['protocol']['test_selection_sha256']:raise ValueError('Test pool differs')
    model,_=load_checkpoint(source/'aligned.pt');base=extract_features(model,test.images,batch_size=64)
    old_episodes=json.loads((source/'episodes.json').read_text(encoding='utf-8'))['episodes']
    records=[];rows=[]
    for shots in (1,5):
        features=apply_transform(base,spaces[shots]);cfg=GraphConfig(**selected[str(shots)]['config'])
        old=sorted((r for r in old_episodes if r['shots']==shots),key=lambda r:r['episode'])
        rng=np.random.default_rng(official['protocol']['seed']+200+shots)
        values=[];delta=[]
        for i in range(600):
            task=sample_episode(test.labels,rng,shots=shots,queries=15)
            support_ids=[test.ids[x] for x in task['support_indices']];query_ids=[test.ids[x] for x in task['query_indices']]
            if old[i]['support_ids']!=support_ids or old[i]['query_ids']!=query_ids:raise ValueError('Unpaired task')
            prediction=predict(features[task['support_indices']],task['support_labels'],features[task['query_indices']],cfg)
            accuracy=float(np.mean(prediction['predictions']==task['query_labels']));previous=old[i]['methods']['full']['accuracy']
            values.append(accuracy);delta.append(accuracy-previous)
            records.append({'shots':shots,'episode':i,'support_ids':support_ids,'query_ids':query_ids,
                'predictions':prediction['predictions'].tolist(),'query_labels':task['query_labels'].tolist(),
                'accuracy':accuracy,'v2_accuracy':previous})
        row={'shots':shots,**summarize(values),'paired_delta_vs_v2':float(np.mean(delta)),
             'paired_ci95':float(1.96*np.std(delta,ddof=1)/np.sqrt(len(delta))),'target_95_reached':float(np.mean(values))>=.95}
        rows.append(row);log('graph_test_complete',**row)
    write_json(out/'episodes.json',{'episodes':records})
    write_json(out/'results.json',{'status':'complete','kind':'same-backbone graph candidate, not deployed',
        'completed_utc':datetime.now(timezone.utc).isoformat(),'lock_sha256':digest_file(lock_file),'rows':rows,
        'target_95_reached':all(r['target_95_reached'] for r in rows),
        'limitation':'Only method selection by validation. Same previously public benchmark; no external validation.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default='outputs/lab04')
    p.add_argument('--features',type=Path,default=Path('.local/lab04-v2'))
    p.add_argument('--study',default='.local/lab04-v3-graph');p.add_argument('--output',default='.local/lab04-v3-graph-test')
    p.add_argument('--data-root',default='.local/mini-imagenet');p.add_argument('--threads',type=int,default=1)
    run(p.parse_args())

if __name__=='__main__':main()
