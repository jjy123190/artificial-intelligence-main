"""Bounded train/validation-only continuation of the fixed wider ResNet12.

No channel expansion, pretrained downloads or test inputs. Stream train images
per minibatch to keep memory bounded. Epoch zero is a selectable baseline; the
plan, class/sample identities, source and initial weights are frozen before
scoring. Optimizer/RNG checkpoints allow deterministic epoch-boundary resume.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .data import image_records,load_split,digest_file,selection_digest
from .experiment import load_checkpoint,save_checkpoint,write_json,log
from .metric_training import balanced_batches,augment_images,episode_loss
from .refine import tasks_for
from .train_v3 import score_validation


def read_train_records(data_root):
    records=image_records(data_root,'train')
    names=sorted({row[0] for row in records})
    if len(names)!=64:raise ValueError('Expected 64 train classes')
    mapping={name:i for i,name in enumerate(names)}
    labels=np.array([mapping[row[0]] for row in records],dtype=np.int64)
    if not np.array_equal(np.bincount(labels),np.full(64,600)):raise ValueError('Expected complete 64x600 train pool')
    return records,labels,names


def read_batch(records,indices):
    images=np.empty((len(indices),84,84,3),dtype=np.uint8)
    for offset,index in enumerate(indices):
        name,path,identity=records[int(index)]
        if not identity.startswith('train/'):
            raise ValueError('Only train images can enter optimizer batches')
        with Image.open(path) as image:
            images[offset]=np.asarray(image.convert('RGB').resize((84,84),Image.Resampling.BICUBIC))
    return images


def guard_output(output,initial):
    output=Path(output).resolve();initial=Path(initial).resolve()
    published=Path(__file__).resolve().parents[2]/'outputs/lab04'
    if output==initial.parent or output==published or output.is_relative_to(published):
        raise ValueError('Never train into a published/model input directory')
    return output


def train(args):
    if type(args.epochs) is not int or not 1<=args.epochs<=32:raise ValueError('Training must be bounded to 1..32 epochs')
    if not 0<args.learning_rate<=.001 or not 0<=args.metric_weight<=1 or not 1<=args.threads<=8:raise ValueError('Invalid fixed training parameters')
    out=guard_output(args.output,args.initial);out.mkdir(parents=True,exist_ok=True)
    sources=('train_continuation.py','train_v3.py','data.py','models.py','metric_training.py','experiment.py','refine.py','calibration.py')
    plan={**vars(args),'initial_sha256':digest_file(args.initial),'widths':[16,32,64,128],
          'test_accessed':False,'source_hashes':{name:digest_file(Path(__file__).with_name(name)) for name in sources},
          'optimizer':'AdamW weight_decay=.0005; fresh optimizer, cosine schedule eta_min=.00001',
          'objective':'CE label_smoothing=.05 + .25 metric loss, same augmentation as previous wider study',
          'selection':'Epoch zero and each epoch, same fixed 100 validation tasks/shot; maximize mean of 1/5-shot; never use test scores',
          'publication':'No auto-publish or auto-test. Frozen downstream validation confirmation required.'}
    plan_file=out/'plan.json'
    if plan_file.exists() and json.loads(plan_file.read_text(encoding='utf-8'))!=plan:raise ValueError('Immutable continuation plan changed')
    if not plan_file.exists():write_json(plan_file,plan)
    result_path=out/'training.json'
    if result_path.exists() and json.loads(result_path.read_text(encoding='utf-8')).get('status')=='complete':
        log('continuation_already_complete');return
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    records,labels,train_classes=read_train_records(args.data_root)
    validation=load_split(args.data_root,'val',max_per_class=200,seed=args.seed+1)
    if len(validation.class_names)!=16 or set(train_classes)&set(validation.class_names):raise ValueError('Train/validation class isolation failed')
    data={'train_images':len(records),'validation_images':len(validation.images),
          'train_classes':train_classes,'validation_classes':validation.class_names,'test_accessed':False,
          'train_ids_sha256':hashlib.sha256('\n'.join(row[2] for row in records).encode()).hexdigest(),
          'validation_ids_sha256':selection_digest(validation),
          'train_pixels_sha256':None,'note':'ID selection and immutable source are recorded; raw dataset pixels are not embedded in this candidate.'}
    identity=out/'data.json'
    if identity.exists() and json.loads(identity.read_text(encoding='utf-8'))!=data:raise ValueError('Selected training/validation data changed')
    if not identity.exists():write_json(identity,data)
    network,_=load_checkpoint(args.initial)
    if list(network.widths)!=plan['widths'] or network.classifier.out_features!=64:raise ValueError('Expected exact wider self-trained backbone')
    optimizer=torch.optim.AdamW(network.parameters(),lr=args.learning_rate,weight_decay=.0005)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=.00001)
    tasks={n:tasks_for(validation.labels,args.seed+30+n,n,100) for n in (1,5)}
    rng=np.random.default_rng(args.seed);generator=torch.Generator().manual_seed(args.seed)
    history=[];best=-1.;best_epoch=0;resume=out/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        if saved['plan']!=plan:raise ValueError('Resume plan mismatch')
        network.load_state_dict(saved['state']);optimizer.load_state_dict(saved['optimizer']);schedule.load_state_dict(saved['scheduler'])
        rng.bit_generator.state=saved['rng'];generator.set_state(saved['generator']);torch.set_rng_state(saved['torch_rng'])
        history,best,best_epoch=saved['history'],saved['best'],saved['best_epoch']
        log('continuation_resume',completed_epochs=len(history),best_epoch=best_epoch)
    else:
        initial=score_validation(network,validation,tasks);best=initial['mean']
        save_checkpoint(out/'best.pt',network,purpose='continued wider candidate, epoch zero control',metadata={**plan,'selected_epoch':0,'validation':initial})
        write_json(out/'initial_validation.json',initial)
        write_json(result_path,{'status':'running','rows':[],'best_epoch':0,'best_mean':best,'test_accessed':False})
        log('continuation_initial',**initial)
    for epoch in range(len(history),args.epochs):
        network.train();started=time.perf_counter();ce_total=metric_total=0.;correct=count=0
        for index,ids in enumerate(balanced_batches(labels,rng)):
            x=augment_images(read_batch(records,ids),generator);y=torch.from_numpy(labels[ids])
            optimizer.zero_grad(set_to_none=True)
            logits,features=network(x,return_features=True)
            ce=F.cross_entropy(logits,y,label_smoothing=.05);metric=episode_loss(features,y)
            loss=ce+args.metric_weight*metric
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss; never publish')
            loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),10.);optimizer.step()
            ce_total+=float(ce.detach())*len(ids);metric_total+=float(metric.detach())*len(ids)
            correct+=int((logits.detach().argmax(1)==y).sum());count+=len(ids)
            if (index+1)%150==0:log('continuation_batch',epoch=epoch+1,batches=index+1,total_batches=600,seconds=time.perf_counter()-started)
        schedule.step();score=score_validation(network,validation,tasks)
        row={'epoch':epoch+1,'ce_loss':ce_total/count,'metric_loss':metric_total/count,'training_accuracy':correct/count,
             'validation':score,'seconds':time.perf_counter()-started,'next_learning_rate':schedule.get_last_lr()[0]}
        history.append(row)
        if score['mean']>best:
            best,best_epoch=score['mean'],epoch+1
            temporary=out/'best.pt.tmp'
            save_checkpoint(temporary,network,purpose='train/validation-only continuation',metadata={**plan,'selected_epoch':best_epoch,'validation':score})
            temporary.replace(out/'best.pt')
        temporary=resume.with_suffix('.tmp')
        torch.save({'plan':plan,'state':network.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':schedule.state_dict(),
                    'rng':rng.bit_generator.state,'generator':generator.get_state(),'torch_rng':torch.get_rng_state(),
                    'history':history,'best':best,'best_epoch':best_epoch},temporary);temporary.replace(resume)
        write_json(result_path,{'status':'running','rows':history,'best_epoch':best_epoch,'best_mean':best,'test_accessed':False})
        log('continuation_epoch',**row)
    write_json(result_path,{'status':'complete','rows':history,'best_epoch':best_epoch,'best_mean':best,'test_accessed':False,
        'completed_utc':datetime.now(timezone.utc).isoformat(),'additional_epochs':len(history),'same_capacity':True})
    log('continuation_complete',best_epoch=best_epoch,best_mean=best)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='.local/lab04-continuation-20261125')
    p.add_argument('--data-root',default='.local/mini-imagenet');p.add_argument('--initial',default='outputs/lab04/wide.pt')
    p.add_argument('--epochs',type=int,default=16);p.add_argument('--learning-rate',type=float,default=.0002)
    p.add_argument('--metric-weight',type=float,default=.25);p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=20261128)
    train(p.parse_args())


if __name__=='__main__':main()
