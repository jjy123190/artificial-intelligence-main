"""Bounded train/val-only meta-trained calibration, no external teacher/backbone.

Pre-registers both candidates and all tasks before scoring; each fits the complete
38400 base-train feature cache. Epoch zero is eligible. Queries labels score only
after prediction; meta-training query labels are from base classes only.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from .data import digest_file,sample_episode
from .graph_calibration import apply_transform,GraphConfig,predict
from .multiview import combine_views
from .metric_training import balanced_batches
from .learned_calibration import LearnedCalibrationConfig,LearnedCalibration,objective,encode_calibration
from .study_literature import read,write,digest_json,summary,paired

SEED=20261203

def data(args):
 cache,art=Path(args.cache),Path(args.artifacts);arrays={};hashes={}
 for split,count in [('train',64),('val',16)]:
  p=cache/f'{split}_views.npz';stamp=read(cache/f'{split}_views.json')
  if stamp['identity']['split']!=split or stamp['sha256']!=digest_file(p) or stamp['identity']['checkpoint_sha256']!=digest_file(art/'continued.pt'):raise ValueError('Wrong input cache')
  with np.load(p,allow_pickle=False) as x:arrays[split]={k:x[k] for k in x.files}
  a=arrays[split]
  if len(np.unique(a['labels']))!=count or not all(str(i).startswith(split+'/') for i in a['ids']) or len(set(a['ids']))!=len(a['ids']):raise ValueError('Invalid class/ID split')
  if not np.array_equal(np.bincount(a['labels']),np.full(count,600)):raise ValueError('Wrong number of images')
  hashes[str(p)]=digest_file(p);hashes[str(cache/f'{split}_views.json')]=digest_file(cache/f'{split}_views.json')
 for file in ('continued.pt','continued_configuration.json','continued_transform.npz'):hashes[str(art/file)]=digest_file(art/file)
 train_names={str(x).split('/')[1] for x in arrays['train']['ids']};val_names={str(x).split('/')[1] for x in arrays['val']['ids']}
 if train_names&val_names:raise ValueError('Train/val class leakage')
 return arrays,hashes


def tasks_for(arrays):
 rng=np.random.default_rng(SEED);indices={'search':[],'confirmation':[]}
 for c in range(16):
  ids=rng.permutation(np.flatnonzero(arrays['val']['labels']==c));indices['search'].extend(ids[:200].tolist());indices['confirmation'].extend(ids[200:].tolist())
 groups={}
 for phase,count,offset in [('search',160,100),('confirmation',600,200)]:
  ids=np.array(indices[phase]);lab=arrays['val']['labels'][ids]
  for shot in (1,5):
   rng=np.random.default_rng(SEED+offset+shot);tasks=[]
   for _ in range(count):
    t=sample_episode(lab,rng,shots=shot,queries=15);t['support_indices']=ids[t['support_indices']];t['query_indices']=ids[t['query_indices']];tasks.append({k:v.tolist() for k,v in t.items()})
   groups[f'{phase}/{shot}']=tasks
 return {'indices':indices,'groups':groups,'val_ids':arrays['val']['ids'].tolist()}


def cache_score(path,features,tasks,config,plan_sha):
 if path.exists():
  r=read(path)
  if r['plan_sha256']!=plan_sha:raise ValueError('Mismatched cached score')
 else:
  predictions=[]
  for t in tasks:
   predictions.append(predict(features[t['support_indices']],t['support_labels'],features[t['query_indices']],config)['predictions'].tolist())
  # Ground truth used only after every prediction has been returned.
  truth=np.asarray([t['query_labels'] for t in tasks]);values=(np.asarray(predictions)==truth).mean(1).tolist()
  r={'plan_sha256':plan_sha,'predictions':predictions,'accuracies':values,'test_accessed':False};write(path,r)
 predictions=np.asarray(r['predictions']);truth=np.asarray([t['query_labels'] for t in tasks])
 if predictions.shape!=truth.shape or not np.issubdtype(predictions.dtype,np.integer) or np.any(predictions<0) or np.any(predictions>4):raise ValueError('Bad task predictions')
 values=(predictions==truth).mean(1)
 if not np.allclose(values,r['accuracies'],atol=1e-14,rtol=0):raise ValueError('Forged score')
 return values.tolist()


def evaluate(adapter,features,tasks,configs,root,phase,identity,plan_sha):
 transformed=features if adapter is None else encode_calibration(adapter.eval(),features)
 values={}
 for shot in (1,5):
  name=f'{phase}_{shot}_{identity}.json';values[str(shot)]=cache_score(root/name,transformed,tasks['groups'][f'{phase}/{shot}'],configs[shot],plan_sha)
 return values


def train_one(args,config,arrays,x,xf,xs,val,tasks,configs,plan_sha):
 out=Path(args.output)/config.architecture;out.mkdir(exist_ok=True);score_dir=out/'scores';score_dir.mkdir(exist_ok=True)
 torch.manual_seed(SEED);adapter=LearnedCalibration(config);labels=arrays['train']['labels'];classes=range(64)
 head=torch.nn.Parameter(torch.tensor(np.stack([x[labels==c].mean(0) for c in classes]),dtype=torch.float32))
 optimizer=torch.optim.AdamW(list(adapter.parameters())+[head],lr=config.learning_rate,weight_decay=.0005)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epochs,eta_min=.00001)
 rng=np.random.default_rng(SEED);generator=torch.Generator().manual_seed(SEED);history=[];best=-1.;selected=0
 training=out/'training.json'
 if training.exists() and read(training).get('status')=='complete':return read(training)
 resume=out/'resume.pt'
 if resume.exists():
  s=torch.load(resume,map_location='cpu',weights_only=True)
  if s['plan_sha256']!=plan_sha:raise ValueError('Wrong training resume')
  adapter.load_state_dict(s['adapter']);head.data.copy_(s['classifier']);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler']);rng.bit_generator.state=s['rng'];generator.set_state(s['generator']);torch.set_rng_state(s['torch_rng']);history=s['history'];best=s['best'];selected=s['selected']
 else:
  score=evaluate(adapter,val,tasks,configs,score_dir,'search','epoch0',plan_sha);best=float(np.mean([np.mean(v) for v in score.values()]))
  torch.save({'state':adapter.state_dict(),'config':asdict(config),'plan_sha256':plan_sha,'epoch':0},out/'best.pt')
  write(out/'epoch0.json',{'mean':best,'shots':{s:summary(v) for s,v in score.items()}})
 for epoch in range(len(history),config.epochs):
  adapter.train();start=time.perf_counter();losses={k:0. for k in ('total','ce','episode','alignment','anchor')};count=0
  for batch_index,ids in enumerate(balanced_batches(labels,rng,ways=8,samples=8)):
   tensors=[torch.from_numpy(a[ids].astype(np.float32)) for a in (x,xf,xs)];y=torch.from_numpy(labels[ids]);optimizer.zero_grad(set_to_none=True)
   terms=objective(adapter,head,*tensors,y,shots=1 if batch_index%2==0 else 5,generator=generator);terms['total'].backward();torch.nn.utils.clip_grad_norm_(list(adapter.parameters())+[head],10.);optimizer.step();count+=len(ids)
   for k in losses:losses[k]+=float(terms[k].detach())*len(ids)
  scheduler.step();score=evaluate(adapter,val,tasks,configs,score_dir,'search',f'epoch{epoch+1}',plan_sha)
  mean=float(np.mean([np.mean(v) for v in score.values()]));row={'epoch':epoch+1,'training_images':count,'losses':{k:v/count for k,v in losses.items()},'search':{s:summary(v) for s,v in score.items()},'selection_mean':mean,'seconds':time.perf_counter()-start};history.append(row)
  if mean>best:best=mean;selected=epoch+1;torch.save({'state':adapter.state_dict(),'config':asdict(config),'plan_sha256':plan_sha,'epoch':selected},out/'best.pt')
  temporary=out/'resume.tmp';torch.save({'adapter':adapter.state_dict(),'classifier':head.detach(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng.bit_generator.state,'generator':generator.get_state(),'torch_rng':torch.get_rng_state(),'history':history,'best':best,'selected':selected,'plan_sha256':plan_sha},temporary);temporary.replace(resume)
  write(training,{'status':'running','plan_sha256':plan_sha,'history':history,'selected_epoch':selected,'test_accessed':False})
  print(json.dumps({'architecture':config.architecture,**row}),flush=True)
 result={'status':'complete','plan_sha256':plan_sha,'history':history,'selected_epoch':selected,'selected_mean':best,'parameters':sum(p.numel() for p in adapter.parameters()),'test_accessed':False,'best_sha256':digest_file(out/'best.pt')};write(training,result);return result


def run(args):
 out=Path(args.output).resolve();allowed=Path('.local').resolve()
 if not out.is_relative_to(allowed) or out==allowed:raise ValueError('Use a dedicated .local directory')
 out.mkdir(parents=True,exist_ok=True);arrays,hashes=data(args);tasks=tasks_for(arrays)
 configurations=[LearnedCalibrationConfig(architecture=a,epochs=12) for a in ('linear','residual')]
 names=('study_learned_calibration.py','learned_calibration.py','metric_training.py','graph_calibration.py','multiview.py','calibration.py','data.py')
 plan={'kind':'no_external_learned_calibration','test_accessed':False,'external_weights':False,'seed':SEED,'configs':[c.to_dict() for c in configurations],
       'source_hashes':{n:digest_file(Path(__file__).with_name(n)) for n in names},'inputs':hashes,'tasks_content_sha256':digest_json(tasks),
       'epochs_selection':'Epoch0 and all12 fixed epochs, mean both shots on160 search tasks/shot. After training both, each chosen checkpoint gets600 confirmation tasks/shot.',
       'selection':'Pick architecture by largest min paired lower bound across shots; eligible only if both means>=.01 and lower95>0. No test/publish.',
       'loss':'base64 CE + 5way alternating1/5shot meta metric + view mean/variance alignment + representation anchor',
       'training_features':'frozen locally trained continued original+flip, complete38400 base images; no pixel augmentation beyond cached two views',
       'limitations':['Added trainable module/budget is not same frozen-weight ablation.','Validation reused; confirmation images separate from search but same16 classes.','Not MAML, not exact Meta-Baseline, and no external weights.','Prediction uses original frozen train transform then new adapter; do not refit on query labels.']}
 if (out/'plan.json').exists():
  old=read(out/'plan.json')
  if {k:v for k,v in old.items() if k!='registered_utc'}!=plan or read(out/'tasks.json')!=tasks:raise ValueError('Registered plan changed')
 else:write(out/'plan.json',{**plan,'registered_utc':datetime.now(timezone.utc).isoformat()});write(out/'tasks.json',tasks)
 plan_sha=digest_file(out/'plan.json');print(json.dumps({'registered':True,'plan_sha256':plan_sha,'test_accessed':False}),flush=True)
 with np.load(Path(args.artifacts)/'continued_transform.npz',allow_pickle=False) as src:
  transform={k[len('shot1_'):]:src[k] for k in src.files if k.startswith('shot1_')}
  for k,v in transform.items():
   if not np.array_equal(v,src['shot5_'+k]):raise ValueError('Expected same1/5 transform')
 a=arrays['train'];x=apply_transform(combine_views(a['original'],a['flipped'],'mean'),transform);xf=apply_transform(a['original'],transform);xs=apply_transform(a['flipped'],transform)
 a=arrays['val'];val=apply_transform(combine_views(a['original'],a['flipped'],'mean'),transform)
 cfg=read(Path(args.artifacts)/'continued_configuration.json')['selected'];configs={s:GraphConfig(**cfg[str(s)]['config']) for s in (1,5)}
 training={}
 for c in configurations:training[c.architecture]=train_one(args,c,arrays,x,xf,xs,val,tasks,configs,plan_sha)
 scores=out/'scores';scores.mkdir(exist_ok=True);reference=evaluate(None,val,tasks,configs,scores,'confirmation','reference',plan_sha)
 results={}
 for config in configurations:
  saved=torch.load(out/config.architecture/'best.pt',map_location='cpu',weights_only=True);adapter=LearnedCalibration(config);adapter.load_state_dict(saved['state'])
  values=evaluate(adapter,val,tasks,configs,scores,'confirmation',config.architecture,plan_sha)
  pairs={s:paired(values[s],reference[s]) for s in ('1','5')};results[config.architecture]={'training':{'selected_epoch':saved['epoch'],'epochs':config.epochs,'extra_parameters':sum(p.numel() for p in adapter.parameters())},'shots':{s:{'accuracy':summary(values[s]),'paired':pairs[s]} for s in ('1','5')},'eligible':all(r['low']>0 and r['mean']>=.01 for r in pairs.values())}
 selected=min(results,key=lambda name:(-min(results[name]['shots'][s]['paired']['low'] for s in ('1','5')),name))
 result={'status':'complete','test_accessed':False,'external_weights':False,'plan_sha256':plan_sha,'reference':{s:summary(v) for s,v in reference.items()},'candidates':results,'selected':selected,'eligible':results[selected]['eligible'],'publication':'not_published'};write(out/'result.json',result)
 print(json.dumps(result,indent=2),flush=True)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',default='.local/lab04-continuation-validation-20261125');p.add_argument('--artifacts',default='outputs/lab04');p.add_argument('--output',default='.local/lab04-learned-calibration-20261125')
 args=p.parse_args();torch.set_num_threads(1)
 with threadpool_limits(limits=1):run(args)


if __name__=='__main__':main()
