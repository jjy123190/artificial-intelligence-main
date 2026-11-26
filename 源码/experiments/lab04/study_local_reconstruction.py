"""Frozen-backbone local-map study on validation only, no external weights.

No learned extra parameters; uses frozen continued encoder before average pool.
Five fixed ridge settings compare with a paired published head. Search/confirmation
images disjoint. Numerical scores are predictions-first and raw records retained.
"""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits

from .data import digest_file,image_records,normalized_tensor,sample_episode
from .experiment import load_checkpoint
from .graph_calibration import apply_transform,GraphConfig,predict
from .multiview import combine_views
from .local_reconstruction import reconstruct_predict
from .study_literature import read,write,digest_json,summary,paired

SEED=20261205
CANDIDATES=tuple((n,r) for n,r in [('channel_scale',.1),('channel_scale',1.),('channel_scale',10.),('local_l2',.1),('local_l2',1.)])


def run(args):
 out=Path(args.output).resolve();local=Path('.local').resolve()
 if not out.is_relative_to(local) or out==local:raise ValueError('Local study output required')
 out.mkdir(parents=True,exist_ok=True);(out/'scores').mkdir(exist_ok=True)
 cache=Path(args.cache);art=Path(args.artifacts);checkpoint=art/'continued.pt'
 with np.load(cache/'val_views.npz',allow_pickle=False) as x:val={k:x[k] for k in x.files}
 stamp=read(cache/'val_views.json')
 if stamp['sha256']!=digest_file(cache/'val_views.npz') or stamp['identity']['checkpoint_sha256']!=digest_file(checkpoint):raise ValueError('Invalid reference cache')
 records=image_records(args.data_root,'val');record_lookup={r[2]:r for r in records};labels=val['labels'];indices={'search':[],'confirmation':[]};rng=np.random.default_rng(SEED)
 for cls in range(16):
  order=rng.permutation(np.flatnonzero(labels==cls));indices['search'].extend(order[:200].tolist());indices['confirmation'].extend(order[200:].tolist())
 tasks={}
 for phase,count,offset in [('search',160,100),('confirmation',600,200)]:
  pool=np.array(indices[phase])
  for shot in (1,5):
   rng=np.random.default_rng(SEED+offset+shot);group=[]
   for _ in range(count):
    t=sample_episode(labels[pool],rng,shots=shot,queries=15);t['support_indices']=pool[t['support_indices']];t['query_indices']=pool[t['query_indices']];group.append({k:v.tolist() for k,v in t.items()})
   tasks[f'{phase}/{shot}']=group
 used=sorted({i for group in tasks.values() for task in group for key in ('support_indices','query_indices') for i in task[key]})
 sources=('study_local_reconstruction.py','local_reconstruction.py','models.py','data.py','experiment.py','graph_calibration.py','multiview.py','calibration.py','study_literature.py')
 plan={'kind':'frozen_self_trained_local_reconstruction','test_accessed':False,'external_weights':False,'seed':SEED,'candidates':[{'normalization':n,'ridge':r} for n,r in CANDIDATES],
       'checkpoint_sha256':digest_file(checkpoint),'source_hashes':{n:digest_file(Path(__file__).with_name(n)) for n in sources},
       'inputs':{str(p):digest_file(p) for p in [cache/'val_views.npz',cache/'val_views.json',art/'continued_transform.npz',art/'continued_configuration.json']},
       'tasks_content_sha256':digest_json(tasks),'val_ids_sha256':digest_json(val['ids'].tolist()),'images_used':len(used),
       'raw_val_pixels_sha256':digest_json({str(val['ids'][i]):digest_file(record_lookup[str(val['ids'][i])][1]) for i in used}),
       'map_source':'continued.encoder output Nx128x5x5, original image only, no spatial pooling, no feature fine-tuning',
       'selection':'top2 search160 per shot, confirm600 per shot, largest paired lower95; bothshot low>0 and mean>=.01 to qualify; no test or publish',
       'limitations':['FRN-inspired frozen-head test, NOT trained FRN reproduction.','Candidate is inductive original-image only; published reference is two-view transductive.','Validation reused across prior studies, confirmation participates in selection.','More per-task support computation; report cost even if no extra parameters.']}
 if (out/'plan.json').exists():
  saved=read(out/'plan.json')
  if {k:v for k,v in saved.items() if k!='registered_utc'}!=plan or read(out/'tasks.json')!=tasks:raise ValueError('Frozen local study changed')
 else:write(out/'plan.json',{**plan,'registered_utc':datetime.now(timezone.utc).isoformat()});write(out/'tasks.json',tasks)
 plan_sha=digest_file(out/'plan.json');print(json.dumps({'registered':True,'test_accessed':False,'images':len(used)}),flush=True)
 map_file=out/'val_maps.npy';map_stamp=out/'val_maps.json'
 if map_file.exists() or map_stamp.exists():
  if not(map_file.exists() and map_stamp.exists()) or read(map_stamp)['plan_sha256']!=plan_sha or read(map_stamp)['sha256']!=digest_file(map_file):raise ValueError('Incomplete or modified local-map cache')
  maps=np.load(map_file,mmap_mode='r',allow_pickle=False)
 else:
  model,_=load_checkpoint(checkpoint);model.eval();temp=out/'val_maps.tmp.npy';maps=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=(len(used),25,128))
  start=time.perf_counter()
  with torch.inference_mode():
   for first in range(0,len(used),32):
    batch=used[first:first+32];pixels=[]
    for index in batch:
     _,file,identity=record_lookup[str(val['ids'][index])]
     if not identity.startswith('val/'):raise ValueError('Wrong image split')
     with Image.open(file) as image:pixels.append(np.asarray(image.convert('RGB').resize((84,84),Image.Resampling.BICUBIC)))
    encoded=model.encoder(normalized_tensor(np.stack(pixels))).flatten(2).transpose(1,2).numpy();maps[first:first+len(batch)]=encoded
    if (first//32+1)%100==0:print(json.dumps({'extracted':first+len(batch),'seconds':time.perf_counter()-start}),flush=True)
  maps.flush();del maps;temp.replace(map_file);write(map_stamp,{'plan_sha256':plan_sha,'sha256':digest_file(map_file),'images':len(used),'shape':[len(used),25,128],'test_accessed':False});maps=np.load(map_file,mmap_mode='r',allow_pickle=False)
 index_lookup={i:j for j,i in enumerate(used)};cfg=read(art/'continued_configuration.json')['selected'];raw=combine_views(val['original'],val['flipped'],'mean');reference={}
 with np.load(art/'continued_transform.npz',allow_pickle=False) as src:
  for shot in (1,5):
   prefix=f'shot{shot}_';transform={k[len(prefix):]:src[k] for k in src.files if k.startswith(prefix)};reference[shot]=apply_transform(raw,transform)
 def score(phase,shot,candidate):
  name='reference' if candidate is None else f'{candidate[0]}_{candidate[1]:g}';path=out/'scores'/f'{phase}_{shot}_{name}.json';group=tasks[f'{phase}/{shot}']
  if path.exists():r=read(path)
  else:
   start=time.perf_counter();predictions=[]
   for t in group:
    if candidate is None:p=predict(reference[shot][t['support_indices']],t['support_labels'],reference[shot][t['query_indices']],GraphConfig(**cfg[str(shot)]['config']))['predictions']
    else:
     s=np.asarray(maps[[index_lookup[i] for i in t['support_indices']]]);q=np.asarray(maps[[index_lookup[i] for i in t['query_indices']]])
     p=reconstruct_predict(s,t['support_labels'],q,normalization=candidate[0],ridge=candidate[1])['predictions']
    predictions.append(p.tolist())
   truth=np.array([t['query_labels'] for t in group]);scores=(np.asarray(predictions)==truth).mean(1).tolist();r={'plan_sha256':plan_sha,'phase':phase,'shot':shot,'name':name,'predictions':predictions,'values':scores,'cached_head_seconds':time.perf_counter()-start,'test_accessed':False};write(path,r)
   print(json.dumps({'phase':phase,'shot':shot,'name':name,**summary(scores),'seconds':r['cached_head_seconds']}),flush=True)
  if r['plan_sha256']!=plan_sha:raise ValueError('Score identity mismatch')
  predicted=np.asarray(r['predictions']);truth=np.array([t['query_labels'] for t in group]);scores=(predicted==truth).mean(1)
  if predicted.shape!=truth.shape or not np.allclose(scores,r['values'],atol=1e-14,rtol=0):raise ValueError('Incorrect scores')
  return scores.tolist()
 result={'status':'complete','test_accessed':False,'external_weights':False,'plan_sha256':plan_sha,'shots':{},'limitations':plan['limitations']}
 for shot in (1,5):
  ref_search=score('search',shot,None);ref_confirm=score('confirmation',shot,None);rows=[]
  for candidate in CANDIDATES:
   values=score('search',shot,candidate);rows.append({'normalization':candidate[0],'ridge':candidate[1],'search':summary(values),'search_paired':paired(values,ref_search)})
  rows.sort(key=lambda r:(-r['search']['accuracy'],r['normalization'],r['ridge']))
  for row in rows[:2]:
   values=score('confirmation',shot,(row['normalization'],row['ridge']));row['confirmation']=summary(values);row['confirmation_paired']=paired(values,ref_confirm)
  selected=min(rows[:2],key=lambda r:(-r['confirmation_paired']['low'],r['normalization'],r['ridge']))
  result['shots'][str(shot)]={'candidates':rows,'selected':{'normalization':selected['normalization'],'ridge':selected['ridge']},'reference':{'search':summary(ref_search),'confirmation':summary(ref_confirm)},'eligible':selected['confirmation_paired']['low']>0 and selected['confirmation_paired']['mean']>=.01}
 result['eligible']=all(r['eligible'] for r in result['shots'].values());result['publication']='not_published';write(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='.local/lab04-local-reconstruction-20261125');p.add_argument('--data-root',default='.local/mini-imagenet');p.add_argument('--cache',default='.local/lab04-continuation-validation-20261125');p.add_argument('--artifacts',default='outputs/lab04');args=p.parse_args();torch.set_num_threads(1)
 with threadpool_limits(limits=1):run(args)


if __name__=='__main__':main()
