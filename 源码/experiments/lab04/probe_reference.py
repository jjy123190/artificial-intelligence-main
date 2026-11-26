"""Validation-only external ResNet12 reference benchmark, clearly attributed."""
from __future__ import annotations
import argparse,json,csv,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .reference_model import load_reference,normalized_reference
from .data import image_records,digest_file
from .experiment import write_json,log,summarize
from .refine import tasks_for
from .calibration import calibrated_predict,CalibrationConfig,base_statistics
from .feature_space import transform_features,fit_feature_space
from .graph_calibration import GraphConfig,predict


def run(args):
    directory=Path(args.output);directory.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads)
    weights=Path(args.checkpoint);provenance=json.loads(weights.with_name('provenance.json').read_text())
    if digest_file(weights)!=provenance['sha256']:raise ValueError('Reference checkpoint hash mismatch')
    ref=load_reference(weights)
    cache_identity={'checkpoint_sha256':digest_file(weights),'seed':args.seed,
                    'preprocessing':'tensor_bilinear_resize92_crop84_author_mean_std',
                    'train_per_class':100,'val_per_class':200,'encoder_only':True,'test_accessed':False}
    identity_file=directory/'cache_identity.json'
    if identity_file.exists() and json.loads(identity_file.read_text())!=cache_identity:
        raise ValueError('Reference cache belongs to a different checkpoint or preprocessing')
    write_json(identity_file,cache_identity)
    arrays={}
    for split,count in [('train',100),('val',200)]:
        cache=directory/f'{split}_features.npz'
        if cache.exists():
            with np.load(cache,allow_pickle=False) as src:arrays[split]={k:src[k] for k in src.files}
            continue
        all_records=image_records(args.data_root,split);names=sorted({r[0] for r in all_records});mapping={n:i for i,n in enumerate(names)}
        records=[];rng=np.random.default_rng(args.seed)
        for name in names:
            rows=[r for r in all_records if r[0]==name];order=np.sort(rng.choice(len(rows),count,replace=False));records.extend(rows[i] for i in order)
        features=[];labels=[];ids=[];started=time.perf_counter()
        with torch.inference_mode():
            for start in range(0,len(records),16):
                images=[]
                for name,path,id in records[start:start+16]:
                    with Image.open(path) as image:images.append(np.asarray(image.convert('RGB')))
                    labels.append(mapping[name]);ids.append(id)
                features.append(ref(normalized_reference(np.stack(images))).numpy())
                if (start//16+1)%100==0:log('reference_extract',split=split,images=start+len(images),seconds=time.perf_counter()-started)
        value={'features':np.concatenate(features),'labels':np.array(labels),'ids':np.array(ids)}
        np.savez_compressed(cache,**value);arrays[split]=value
    train,val=arrays['train'],arrays['val'];records=[]
    for shots in (1,5):
        tasks=tasks_for(val['labels'],args.seed+shots,shots,300)
        for centering in (False,True):
            prior=fit_feature_space(train['features'],train['labels'],center_strength=float(centering))
            x=transform_features(val['features'],prior)
            for method in ('nearest_prototype','balanced_refinement','graph_refinement'):
                values=[]
                for task in tasks:
                    support=x[task['support_indices']];query=x[task['query_indices']];labels=task['support_labels']
                    if method=='nearest_prototype':prediction=calibrated_predict(support,labels,query,prior)
                    else:prediction=predict(support,labels,query,GraphConfig(neighbors=10,alpha=.95,temperature=.15,prototype_weight=.4,prototype_steps=10,graph_weight=.5 if method=='graph_refinement' else 0.,balanced=True))
                    values.append(float(np.mean(prediction['predictions']==task['query_labels'])))
                row={'shots':shots,'centering':centering,'method':method,**summarize(values)};records.append(row);log('reference_validation',**row)
    result={'status':'complete','kind':'external research reference, not self-trained course model','test_accessed':False,
        'reference':provenance,'preprocessing':cache_identity,'parameters':sum(p.numel() for p in ref.parameters()),'rows':records,
        'target_reached_on_validation':{str(n):max(r['accuracy'] for r in records if r['shots']==n)>=.95 for n in (1,5)},
        'limitations':['Author-declared training provenance and matching split lists, not an independent audit of original training.',
                      '640-channel external encoder is not the lightweight model required by the course.',
                      'Only validation scores are measured; none is an independent test claim.']}
    write_json(directory/'validation.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',default='.local/lab04-reference-feat/feat-1-shot.pth')
    p.add_argument('--data-root',default='.local/mini-imagenet');p.add_argument('--output',default='.local/lab04-reference-feat')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=20261125)
    run(p.parse_args())

if __name__=='__main__':main()
