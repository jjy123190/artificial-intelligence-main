"""Explicit opt-in, bounded feature-distillation experiment (never auto-published).

Default invocation ONLY registers a plan. --run is required for expensive real
training. --synthetic-smoke runs a random-data optimizer step, not a dataset run.
The official FEAT teacher has no 64-class classifier here: use feature/relational
losses, not fabricated teacher logits. This is inspired by RFS, not a reproduction.
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
from threadpoolctl import threadpool_limits

from .calibration import unit_features
from .data import digest_file,image_records,sample_episode
from .experiment import load_checkpoint
from .feature_distillation import FeatureDistillationConfig,FeatureProjector,training_loss
from .metric_training import augment_images,balanced_batches,episode_loss
from .reference_model import load_reference
from .study_literature import read,write,digest_json


METHODS=('control','distill')


def safe_output(value):
    output=Path(value).resolve();source=Path(__file__).resolve().parents[2]
    local=(source/'.local').resolve()
    if not output.is_relative_to(local) or output==local:
        raise ValueError('Training output must be a dedicated child of source/.local')
    return output


def pixels_for(records, indices, allowed):
    if allowed not in ('train','val'):raise ValueError('Only train/val images allowed')
    output=np.empty((len(indices),84,84,3),dtype=np.uint8)
    for j,index in enumerate(indices):
        _,path,identity=records[int(index)]
        if not identity.startswith(allowed+'/'):raise ValueError('Image split mismatch')
        with Image.open(path) as image:output[j]=np.asarray(image.convert('RGB').resize((84,84),Image.Resampling.BICUBIC))
    return output


def normalize_same_view(student_tensor):
    """Recover the EXACT augmented RGB view, then use the teacher's train scale.

    No further crop/resize on one side: randomly cropped teacher and student
    inputs must describe the same view. Author resize92->crop84 belongs to the
    old evaluation adapter, not an extra augmentation added to teacher alone.
    """
    if student_tensor.ndim!=4 or student_tensor.shape[1]!=3:raise ValueError('Expected NCHW RGB tensors')
    mean=student_tensor.new_tensor((.485,.456,.406))[None,:,None,None]
    std=student_tensor.new_tensor((.229,.224,.225))[None,:,None,None]
    rgb=(student_tensor*std+mean).clamp(0,1)
    tm=student_tensor.new_tensor((120.39586422,115.59361427,104.54012653))[None,:,None,None]/255
    ts=student_tensor.new_tensor((70.68188272,68.27635443,72.54505529))[None,:,None,None]/255
    return (rgb-tm)/ts


def atomic_torch(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def validation_selection(model,records,labels,indices,tasks,device):
    """Select epochs using fixed inductive tasks; no teacher/val fitting."""
    model.eval();features=[]
    with torch.inference_mode():
        for offset in range(0,len(indices),32):
            pixels=pixels_for(records,indices[offset:offset+32],'val')
            from .data import normalized_tensor
            features.append(model.features(normalized_tensor(pixels).to(device)).cpu().numpy())
    x=unit_features(np.concatenate(features));scores={}
    for shot in (1,5):
        predictions=[]
        for task in tasks[str(shot)]:
            support=x[task['support_indices']];support_labels=np.array(task['support_labels']);query=x[task['query_indices']]
            proto=np.stack([support[support_labels==cls].mean(0) for cls in range(5)])
            predictions.append(((query[:,None]-proto[None])**2).sum(2).argmin(1))
        # Validation labels only score predictions after the forward pass.
        scores[str(shot)]=float(np.mean([np.mean(p==np.asarray(t['query_labels'])) for p,t in zip(predictions,tasks[str(shot)])]))
    return {'1shot':scores['1'],'5shot':scores['5'],'mean':(scores['1']+scores['5'])/2}


def data_identity(data_root):
    records={s:image_records(data_root,s) for s in ('train','val')};labels={};classes={}
    for split,count in (('train',64),('val',16)):
        names=sorted({r[0] for r in records[split]});mapping={n:i for i,n in enumerate(names)}
        if len(names)!=count:raise ValueError('Unexpected class split')
        labels[split]=np.array([mapping[r[0]] for r in records[split]],dtype=np.int64)
        if not np.array_equal(np.bincount(labels[split]),np.full(count,600)):raise ValueError('Expected 600 images/class')
        classes[split]=names
    if set(classes['train'])&set(classes['val']):raise ValueError('Train/val class overlap')
    return records,labels,classes


def register(args):
    if args.method not in METHODS or not 1<=args.epochs<=32 or not 1<=args.threads<=8:raise ValueError('Invalid training budget')
    if not 0<args.learning_rate<=.001:raise ValueError('Invalid learning rate')
    if args.device not in ('cpu','cuda') or (args.run and args.device=='cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device unavailable')
    out=safe_output(args.output)
    for value in (args.initial,args.teacher):
        if Path(value).resolve().is_relative_to(out):raise ValueError('Input checkpoint must not be in training output')
    out.mkdir(parents=True,exist_ok=True)
    # Bind actual train/val image bytes, not just IDs. This never reads test files.
    pixel_digest=hashlib.sha256();pixel_files=0
    for split in ('train','val'):
        for _,path,identity in image_records(args.data_root,split):
            pixel_digest.update(identity.encode('utf-8'));pixel_digest.update(b'\0')
            pixel_digest.update(bytes.fromhex(digest_file(path)));pixel_files+=1
    records,labels,classes=data_identity(args.data_root)
    rng=np.random.default_rng(args.seed+1);val_indices=[]
    for cls in range(16):val_indices.extend(rng.permutation(np.flatnonzero(labels['val']==cls))[:100])
    val_indices=np.array(val_indices,dtype=int)
    tasks={str(shot):[{k:v.tolist() for k,v in sample_episode(labels['val'][val_indices],np.random.default_rng(args.seed+shot*100+i),shots=shot,queries=15).items()}
                     for i in range(100)] for shot in (1,5)}
    sources=('train_feature_distillation.py','feature_distillation.py','reference_model.py','models.py','data.py','metric_training.py','calibration.py','experiment.py','study_literature.py')
    teacher_provenance=read(Path(args.teacher).with_name('provenance.json'))
    if teacher_provenance['sha256']!=digest_file(args.teacher):raise ValueError('Teacher provenance mismatch')
    config=FeatureDistillationConfig(feature_weight=1. if args.method=='distill' else 0.,relation_weight=.5 if args.method=='distill' else 0.)
    plan={'status':'registered','method':args.method,'epochs':args.epochs,'seed':args.seed,'learning_rate':args.learning_rate,
          'device':args.device,'threads':args.threads,'test_accessed':False,'source_hashes':{n:digest_file(Path(__file__).with_name(n)) for n in sources},
          'initial_sha256':digest_file(args.initial),'teacher_sha256':digest_file(args.teacher),'teacher_provenance':teacher_provenance,
          'runtime':{'torch':str(torch.__version__),'numpy':np.__version__},
          'raw_train_val_pixels_sha256':pixel_digest.hexdigest(),'raw_files_fingerprinted':pixel_files,
          'external_teacher_used_for_training':args.method=='distill','loss':config.to_dict(),'optimizer':'fresh AdamW wd=.0005; cosine eta_min=.00001',
          'batch':'4 base classes x 4 examples =16; each train image once per epoch; exactly paired views',
          'augmentation':'independent random crop/flip/color per view; student and teacher see exactly the same augmented RGB',
          'selection':'epoch0 plus every epoch; maximize fixed100 inductive tasks/shot mean; no test, no early stopping, no auto-publication',
          'data':{'train_images':len(records['train']),'val_selection_images':len(val_indices),'class_names':classes,
                  'ids_sha256':{s:hashlib.sha256('\n'.join(r[2] for r in records[s]).encode()).hexdigest() for s in ('train','val')},
                  'val_indices':val_indices.tolist(),'tasks_content_sha256':digest_json(tasks)},
          'limitations':['Same reused validation classes; this is not independent confirmation.',
                         'Control/distill share same 16-image minibatches, augmentation, initialization, epochs and losses except KD.',
                         'Projector 128->640 is training-only; selected best.pt contains student encoder and classifier only.',
                         'This runner has NOT established improved accuracy. A separate frozen validation gate is mandatory before any test.']}
    path=out/'plan.json'
    if path.exists():
        if read(path)!=plan or read(out/'validation_tasks.json')!=tasks:raise ValueError('Immutable training plan/data changed')
    else:
        if any(out.iterdir()):raise ValueError('Unregistered nonempty training directory')
        write(out/'plan.json',plan);write(out/'validation_tasks.json',tasks)
    return out,plan,records,labels,val_indices,tasks


def train(args):
    out,plan,records,labels,val_indices,tasks=register(args)
    print(json.dumps({'registered':True,'run_requested':args.run,'test_accessed':False,'output':str(out)}),flush=True)
    if not args.run:return
    result_path=out/'training.json'
    if result_path.exists() and read(result_path).get('status')=='complete':
        print('Training already complete; refusing duplicate execution.');return
    torch.manual_seed(args.seed);torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    device=torch.device(args.device);student,checkpoint=load_checkpoint(args.initial)
    if list(student.widths)!=[16,32,64,128]:raise ValueError('This recipe is for the fixed 509856-parameter student')
    student.to(device);projector=FeatureProjector(128,640).to(device)
    teacher=None
    if args.method=='distill':teacher=load_reference(args.teacher).to(device).eval();teacher.requires_grad_(False)
    parameters=list(student.parameters())+(list(projector.parameters()) if teacher is not None else [])
    optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=.0005)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=.00001)
    rng=np.random.default_rng(args.seed);generator=torch.Generator().manual_seed(args.seed)
    rows=[];best=-float('inf');best_epoch=0;plan_sha=digest_file(out/'plan.json')
    resume=out/'resume.pt';cfg=FeatureDistillationConfig(**plan['loss'])
    def save_best(epoch,score):
        atomic_torch(out/'best.pt',{'state_dict':{k:v.detach().cpu() for k,v in student.state_dict().items()},
              'widths':list(student.widths),'classes':64,'purpose':'feature distillation candidate, NOT PUBLISHED',
              'metadata':{'plan_sha256':plan_sha,'epoch':epoch,'validation':score,'method':args.method,
                          'external_teacher_used':teacher is not None,'teacher_sha256':plan['teacher_sha256'] if teacher is not None else None}})
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        if saved['plan_sha256']!=plan_sha:raise ValueError('Resume belongs to another plan')
        student.load_state_dict(saved['student']);projector.load_state_dict(saved['projector']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
        rng.bit_generator.state=saved['numpy_rng'];generator.set_state(saved['generator']);torch.set_rng_state(saved['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
        rows,best,best_epoch=saved['rows'],saved['best'],saved['best_epoch']
    else:
        score=validation_selection(student,records['val'],labels['val'],val_indices,tasks,device)
        best=score['mean'];save_best(0,score);write(out/'initial_validation.json',score)
    for epoch in range(len(rows),args.epochs):
        student.train();projector.train();projector.requires_grad_(teacher is not None);started=time.perf_counter();count=0;totals={k:0. for k in ('total','ce','feature','relation','alignment','metric')}
        for batch_index,indices in enumerate(balanced_batches(labels['train'],rng,ways=4,samples=4)):
            pixels=pixels_for(records['train'],indices,'train')
            a=augment_images(pixels,generator).to(device);b=augment_images(pixels,generator).to(device)
            y=torch.from_numpy(labels['train'][indices]).to(device);optimizer.zero_grad(set_to_none=True)
            logits,za=student(a,return_features=True);zb=student.features(b)
            if teacher is not None:
                with torch.no_grad():ta=teacher(normalize_same_view(a));tb=teacher(normalize_same_view(b))
            else:
                # Zero-weight KD terms permit the exact same loss computation in control.
                ta=torch.zeros(len(a),640,device=device);tb=torch.zeros_like(ta)
            metric=(episode_loss(za,y,ways=4,samples=4)+episode_loss(zb,y,ways=4,samples=4))/2
            losses=training_loss(logits,y,za,zb,ta,tb,projector,metric,cfg)
            losses['total'].backward();torch.nn.utils.clip_grad_norm_(parameters,10.);optimizer.step()
            count+=len(indices)
            for k in totals:totals[k]+=float(losses[k].detach())*len(indices)
            if (batch_index+1)%300==0:print(json.dumps({'epoch':epoch+1,'batches':batch_index+1,'seconds':time.perf_counter()-started}),flush=True)
        scheduler.step();score=validation_selection(student,records['val'],labels['val'],val_indices,tasks,device)
        row={'epoch':epoch+1,'losses':{k:v/count for k,v in totals.items()},'validation':score,'seconds':time.perf_counter()-started,'train_images':count};rows.append(row)
        if score['mean']>best:best,best_epoch=score['mean'],epoch+1;save_best(best_epoch,score)
        atomic_torch(resume,{'plan_sha256':plan_sha,'student':student.state_dict(),'projector':projector.state_dict(),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'numpy_rng':rng.bit_generator.state,
            'generator':generator.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            'rows':rows,'best':best,'best_epoch':best_epoch})
        write(result_path,{'status':'running','plan_sha256':plan_sha,'rows':rows,'best_epoch':best_epoch,'test_accessed':False})
        print(json.dumps(row),flush=True)
    write(result_path,{'status':'complete','plan_sha256':plan_sha,'rows':rows,'best_epoch':best_epoch,'best_mean':best,
                      'best_sha256':digest_file(out/'best.pt'),'test_accessed':False,'publication':'not_published',
                      'completed_utc':datetime.now(timezone.utc).isoformat()})


def synthetic_smoke(args):
    torch.manual_seed(54);torch.set_num_threads(args.threads);student,_=load_checkpoint(args.initial)
    teacher=load_reference(args.teacher);teacher.requires_grad_(False);projector=FeatureProjector(128,640)
    student.train();teacher.eval();optimizer=torch.optim.AdamW(list(student.parameters())+list(projector.parameters()),lr=1e-4)
    pixels=np.random.default_rng(54).integers(0,256,size=(16,84,84,3),dtype=np.uint8);g=torch.Generator().manual_seed(54)
    a=augment_images(pixels,g);b=augment_images(pixels,g);labels=torch.arange(4).repeat_interleave(4)
    logits,za=student(a,return_features=True);zb=student.features(b)
    with torch.no_grad():ta=teacher(normalize_same_view(a));tb=teacher(normalize_same_view(b))
    metric=(episode_loss(za,labels,ways=4,samples=4)+episode_loss(zb,labels,ways=4,samples=4))/2
    losses=training_loss(logits,labels,za,zb,ta,tb,projector,metric);losses['total'].backward();optimizer.step()
    if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Teacher received gradients')
    print(json.dumps({'kind':'synthetic_smoke_only','passed':True,'test_accessed':False,'real_train_images':0,
                      'accuracy_claim':None,'losses':{k:float(v.detach()) for k,v in losses.items()}}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='.local/mini-imagenet');p.add_argument('--initial',default='outputs/lab04/continued.pt')
    p.add_argument('--teacher',default='.local/lab04-reference-feat/feat-1-shot.pth');p.add_argument('--output',default='.local/lab04-feature-distillation-20261125')
    p.add_argument('--method',choices=METHODS,default='distill');p.add_argument('--epochs',type=int,default=16)
    p.add_argument('--learning-rate',type=float,default=.0002);p.add_argument('--seed',type=int,default=20261202)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--run',action='store_true');p.add_argument('--synthetic-smoke',action='store_true');args=p.parse_args()
    if args.synthetic_smoke and args.run:raise ValueError('Synthetic smoke cannot be combined with dataset training')
    with threadpool_limits(limits=args.threads):
        if args.synthetic_smoke:synthetic_smoke(args)
        else:train(args)


if __name__=='__main__':main()
