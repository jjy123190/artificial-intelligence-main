"""Small honest CNN-training pilot for the duplicate-channel bottleneck.

Two equal-budget branches, 96 images/class x64 base classes x2 epochs. This is
NOT the full training protocol and never a publishable test result. Candidate
changes only zero-sum fan-out + pair-decorrelation versus matched control.
Original epoch0 stays selectable; no external teacher, no test image access.
"""
from __future__ import annotations
import argparse,copy,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from .data import image_records,sample_episode,digest_file,normalized_tensor
from .experiment import load_checkpoint
from .models import distribution_alignment
from .metric_training import balanced_batches,augment_images
from .channel_diversity import diversify_fanout,pair_correlation_penalty,cosine_episode_loss
from .train_continuation import read_batch
from .study_literature import read,write,digest_json,summary,paired

SEED=20261204


def extract(model,records,batch=32):
 from PIL import Image
 model.eval();features=[]
 with torch.inference_mode():
  for start in range(0,len(records),batch):
   pixels=[]
   for _,p,identity in records[start:start+batch]:
    if not identity.startswith('val/'):raise ValueError('Pilot evaluation only val')
    with Image.open(p) as im:pixels.append(np.asarray(im.convert('RGB').resize((84,84),Image.Resampling.BICUBIC)))
   features.append(model.features(normalized_tensor(np.stack(pixels))).numpy())
 return np.concatenate(features)


def evaluate(model,records,tasks,path,plan_sha):
 if path.exists():return read(path)
 features=extract(model,records);z=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12);result={}
 for shot in (1,5):
  predictions=[]
  for task in tasks[str(shot)]:
   support=z[task['support_indices']];labels=np.asarray(task['support_labels']);prototypes=np.stack([support[labels==c].mean(0) for c in range(5)])
   predictions.append(((z[task['query_indices']][:,None]-prototypes[None])**2).sum(2).argmin(1).tolist())
  truth=np.asarray([t['query_labels'] for t in tasks[str(shot)]]);values=(np.asarray(predictions)==truth).mean(1).tolist()
  result[str(shot)]={'predictions':predictions,'values':values,'summary':summary(values)}
 result={'plan_sha256':plan_sha,'test_accessed':False,'shots':result,'selection_mean':float(np.mean([r['summary']['accuracy'] for r in result.values()]))}
 write(path,result);return result


def records_for(root):
 values={};labels={}
 for split,per_class,count,offset in [('train',96,64,1),('val',80,16,2)]:
  records=image_records(root,split);names=sorted({r[0] for r in records});rng=np.random.default_rng(SEED+offset)
  if len(names)!=count:raise ValueError('Invalid split')
  selected=[]
  for c in names:
   pool=[r for r in records if r[0]==c];order=np.sort(rng.choice(len(pool),per_class,replace=False));selected.extend(pool[i] for i in order)
  values[split]=selected;labels[split]=np.repeat(np.arange(count),per_class)
 if {r[0] for r in values['train']}&{r[0] for r in values['val']}:raise ValueError('Class leakage')
 return values,labels


def run(args):
 out=Path(args.output).resolve();local=Path('.local').resolve()
 if not out.is_relative_to(local) or out==local:raise ValueError('Pilot requires a dedicated local output')
 out.mkdir(parents=True,exist_ok=True);records,labels=records_for(args.data_root)
 tasks={str(s):[] for s in (1,5)}
 for shot in (1,5):
  rng=np.random.default_rng(SEED+100+shot)
  tasks[str(shot)]=[{k:v.tolist() for k,v in sample_episode(labels['val'],rng,shots=shot,queries=15).items()} for _ in range(200)]
 source_names=('pilot_channel_diversity.py','channel_diversity.py','metric_training.py','models.py','data.py','experiment.py','train_continuation.py','study_literature.py')
 plan={'kind':'same_capacity_no_external_CNN_mechanism_pilot','status':'registered','seed':SEED,'test_accessed':False,'external_weights':False,
       'branches':['control','diversity'],'epochs':2,'train_images':6144,'train_per_class':96,'val_images':1280,'tasks_per_shot':200,
       'initial_sha256':digest_file(args.initial),'source_hashes':{n:digest_file(Path(__file__).with_name(n)) for n in source_names},
       'ids':{s:[r[2] for r in v] for s,v in records.items()},'pixel_hashes':{s:digest_json({r[2]:digest_file(r[1]) for r in v}) for s,v in records.items()},
       'tasks_content_sha256':digest_json(tasks),'threads':2,'runtime':{'torch':str(torch.__version__),'numpy':np.__version__},
       'optimizer':'AdamW lr=.0008,wd=.0005; cosine to .0001; fresh, same seed/order/augmentations in both branches',
       'objective':'CE(label_smoothing=.05) + .25*5way alternating1/5shot cosine metric + .5*mean/variance of augmented view and its flip',
       'only_branch_changes':'diversity: +/− fanout perturb scale .25 seed fixed, plus .02 pair-correlation penalty; control neither',
       'selection':'inductive cosine-normalized Euclidean prototypes; epoch0 plus each epoch, same200val tasks/shot; choose mean shots. No confirmation/test/publication.',
       'limitation':'Small pilot subset and2epochs are not full training; same validation tasks select epochs, so not independent improvement evidence.'}
 if (out/'plan.json').exists():
  old=read(out/'plan.json')
  if {k:v for k,v in old.items() if k!='registered_utc'}!=plan or read(out/'tasks.json')!=tasks:raise ValueError('Frozen pilot changed')
 else:write(out/'plan.json',{**plan,'registered_utc':datetime.now(timezone.utc).isoformat()});write(out/'tasks.json',tasks)
 plan_sha=digest_file(out/'plan.json');print(json.dumps({'pilot_registered':True,'test_accessed':False,'plan_sha256':plan_sha}),flush=True)
 original,_=load_checkpoint(args.initial);initial=evaluate(original,records['val'],tasks,out/'initial.json',plan_sha);del original
 for branch in ('control','diversity'):
  directory=out/branch;directory.mkdir(exist_ok=True);history_file=directory/'training.json'
  if history_file.exists() and read(history_file).get('status')=='complete':continue
  torch.manual_seed(SEED);model,_=load_checkpoint(args.initial)
  if branch=='diversity':diversify_fanout(model,relative_scale=.25,seed=SEED)
  optimizer=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.0005);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2,eta_min=.0001)
  rng=np.random.default_rng(SEED);generator=torch.Generator().manual_seed(SEED);history=[];best=initial['selection_mean'];best_epoch=0
  resume=directory/'resume.pt'
  if resume.exists():
   saved=torch.load(resume,weights_only=True,map_location='cpu')
   if saved['plan_sha256']!=plan_sha:raise ValueError('Wrong pilot resume')
   model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);rng.bit_generator.state=saved['rng'];generator.set_state(saved['generator']);torch.set_rng_state(saved['torch_rng']);history=saved['history'];best=saved['best'];best_epoch=saved['best_epoch']
  else:
   # Always preserve ORIGINAL unperturbed epoch zero as candidate, even in diversity arm.
   base=torch.load(args.initial,weights_only=True,map_location='cpu');torch.save(base,directory/'best.pt')
  for epoch in range(len(history),2):
   model.train();start=time.perf_counter();count=0;loss_total=0.;ce_total=0.;correlation_total=0.;correct=0
   for batch_index,ids in enumerate(balanced_batches(labels['train'],rng,ways=8,samples=8)):
    pixels=read_batch(records['train'],ids);x=augment_images(pixels,generator);y=torch.from_numpy(labels['train'][ids]);optimizer.zero_grad(set_to_none=True)
    logits,z=model(x,return_features=True);z2=model.features(x.flip(-1));ce=F.cross_entropy(logits,y,label_smoothing=.05)
    metric=cosine_episode_loss(z,y,shots=1 if batch_index%2==0 else 5,generator=generator)
    alignment=distribution_alignment(z,z2);correlation=pair_correlation_penalty(z)
    loss=ce+.25*metric+.5*alignment+(.02*correlation if branch=='diversity' else 0.)
    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step();count+=len(ids);loss_total+=float(loss.detach())*len(ids);ce_total+=float(ce.detach())*len(ids);correlation_total+=float(correlation.detach())*len(ids);correct+=int((logits.argmax(1)==y).sum())
    if (batch_index+1)%32==0:print(json.dumps({'branch':branch,'epoch':epoch+1,'batches':batch_index+1,'seconds':time.perf_counter()-start}),flush=True)
   scheduler.step();validation=evaluate(model,records['val'],tasks,directory/f'epoch{epoch+1}_validation.json',plan_sha)
   row={'epoch':epoch+1,'train_images':count,'loss':loss_total/count,'ce':ce_total/count,'pair_penalty':correlation_total/count,'augmented_train_accuracy':correct/count,'selection_mean':validation['selection_mean'],'validation':{s:r['summary'] for s,r in validation['shots'].items()},'seconds':time.perf_counter()-start};history.append(row)
   if row['selection_mean']>best:
    best=row['selection_mean'];best_epoch=epoch+1;torch.save({'state_dict':model.state_dict(),'widths':list(model.widths),'classes':64,'purpose':'no-external diversity PILOT not published','metadata':{'plan_sha256':plan_sha,'branch':branch,'epoch':best_epoch,'test_accessed':False}},directory/'best.pt')
   temp=directory/'resume.tmp';torch.save({'plan_sha256':plan_sha,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng.bit_generator.state,'generator':generator.get_state(),'torch_rng':torch.get_rng_state(),'history':history,'best':best,'best_epoch':best_epoch},temp);temp.replace(resume)
   write(history_file,{'status':'running','history':history,'best_epoch':best_epoch,'best':best,'test_accessed':False});print(json.dumps({'branch':branch,**row}),flush=True)
  write(history_file,{'status':'complete','history':history,'best_epoch':best_epoch,'best':best,'plan_sha256':plan_sha,'best_sha256':digest_file(directory/'best.pt'),'test_accessed':False})
 result={'status':'complete','kind':'mechanism_pilot_not_full_benchmark','plan_sha256':plan_sha,'test_accessed':False,'external_weights':False,'initial':{s:r['summary'] for s,r in initial['shots'].items()},'branches':{},'not_publishable':True}
 scores={}
 for branch in ('control','diversity'):
  training=read(out/branch/'training.json');epoch=training['best_epoch'];v=initial if epoch==0 else read(out/branch/f'epoch{epoch}_validation.json');scores[branch]={s:r['values'] for s,r in v['shots'].items()}
  result['branches'][branch]={'selected_epoch':epoch,'best_sha256':training['best_sha256'],'validation':{s:r['summary'] for s,r in v['shots'].items()},'paired_vs_initial':{s:paired(scores[branch][s],initial['shots'][s]['values']) for s in ('1','5')}}
 result['paired_diversity_vs_control']={s:paired(scores['diversity'][s],scores['control'][s]) for s in ('1','5')};write(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',default='.local/mini-imagenet');p.add_argument('--initial',default='outputs/lab04/continued.pt');p.add_argument('--output',default='.local/lab04-channel-pilot-20261125');args=p.parse_args();torch.set_num_threads(2)
 with threadpool_limits(limits=2):run(args)


if __name__=='__main__':main()
