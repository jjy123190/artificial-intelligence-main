"""Offline evidence audit for no-external learned calibration and small CNN pilot.

Recomputes raw predictions, epoch choice and gates. Does not run inference or
read test images. Fails on missing epochs, mismatched inputs, tampered summaries,
wrong checkpoints, label/split/task violations, or attempted plan substitution.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .data import digest_file
from .study_literature import read,write,digest_json,summary,paired


def check_plan(root):
 plan=read(root/'plan.json')
 if plan.get('test_accessed') is not False or plan.get('external_weights') is not False:raise ValueError('Not a train/val no-external study')
 for name,value in plan['source_hashes'].items():
  if digest_file(Path(__file__).with_name(name))!=value:raise ValueError('Frozen source changed: '+name)
 for name,value in plan.get('inputs',{}).items():
  if digest_file(name)!=value:raise ValueError('Input changed: '+name)
 return plan,digest_file(root/'plan.json')


def raw_values(record,tasks,plan_sha,key='accuracies'):
 if record.get('plan_sha256')!=plan_sha or record.get('test_accessed') is not False:raise ValueError('Record identity mismatch')
 predicted=np.asarray(record['predictions']);truth=np.asarray([t['query_labels'] for t in tasks])
 if predicted.shape!=truth.shape or not np.issubdtype(predicted.dtype,np.integer) or np.any(predicted<0) or np.any(predicted>4):raise ValueError('Invalid prediction array')
 values=(predicted==truth).mean(1)
 if not np.allclose(values,record[key],atol=1e-14,rtol=0):raise ValueError('Accuracy not explained by predictions')
 return values.tolist()


def validate_tasks(tasks,ids=None):
 for task in tasks:
  s,q=task['support_indices'],task['query_indices'];sl,ql=task['support_labels'],task['query_labels']
  if len(set(s))!=len(s) or len(set(q))!=75 or set(s)&set(q):raise ValueError('Leaky or duplicate support/query')
  if len(ql)!=75 or np.bincount(ql,minlength=5).tolist()!=[15]*5 or len(sl)!=len(s):raise ValueError('Incorrect episode')
  if sorted(set(sl))!=list(range(5)) or len(s) not in (5,25):raise ValueError('Wrong support budget')
  if ids is not None:
   for i in s+q:
    if not 0<=i<len(ids) or not ids[i].startswith('val/'):raise ValueError('Invalid validation ID')
   mapping={}
   for i,label in zip(s+q,sl+ql):
    cls=ids[i].split('/')[1]
    if label in mapping and mapping[label]!=cls:raise ValueError('Episode labels disagree with image classes')
    mapping[label]=cls
   if len(set(mapping.values()))!=5:raise ValueError('Episode class overlap')


def learned(root):
 plan,sha=check_plan(root);tasks=read(root/'tasks.json');result=read(root/'result.json')
 if digest_json(tasks)!=plan['tasks_content_sha256'] or result['plan_sha256']!=sha:raise ValueError('Changed tasks/result')
 if set(tasks['indices']['search'])&set(tasks['indices']['confirmation']):raise ValueError('Search/confirmation image overlap')
 for name,group in tasks['groups'].items():
  validate_tasks(group,tasks['val_ids']);allowed=set(tasks['indices'][name.split('/')[0]])
  if any(not set(t['support_indices']+t['query_indices'])<=allowed for t in group):raise ValueError('Task outside partition')
 reference={};count=0
 for shot in ('1','5'):
  v=raw_values(read(root/'scores'/f'confirmation_{shot}_reference.json'),tasks['groups'][f'confirmation/{shot}'],sha)
  reference[shot]=v;count+=len(v)
  if result['reference'][shot]!=summary(v):raise ValueError('Reference summary mismatch')
 recomputed={}
 for cfg in plan['configs']:
  name=cfg['architecture'];directory=root/name;training=read(directory/'training.json');history=training['history']
  if training['status']!='complete' or len(history)!=cfg['epochs'] or [r['epoch'] for r in history]!=list(range(1,cfg['epochs']+1)):raise ValueError('Incomplete training history')
  epoch_means=[]
  for epoch in range(cfg['epochs']+1):
   scores={}
   for shot in ('1','5'):
    p=directory/'scores'/f'search_{shot}_epoch{epoch}.json';v=raw_values(read(p),tasks['groups'][f'search/{shot}'],sha);scores[shot]=summary(v);count+=len(v)
   mean=float(np.mean([s['accuracy'] for s in scores.values()]));epoch_means.append(mean)
   if epoch and (history[epoch-1]['search']!=scores or history[epoch-1]['selection_mean']!=mean or history[epoch-1]['training_images']!=38400):raise ValueError('Training/validation row forged')
  winner=int(np.argmax(epoch_means))
  if training['selected_epoch']!=winner or abs(training['selected_mean']-epoch_means[winner])>1e-14:raise ValueError('Epoch not selected on fixed rule')
  ck=torch.load(directory/'best.pt',weights_only=True,map_location='cpu')
  if ck['epoch']!=winner or ck['plan_sha256']!=sha or ck['config']!=cfg or digest_file(directory/'best.pt')!=training['best_sha256']:raise ValueError('Selected checkpoint mismatch')
  row=result['candidates'][name];pairs={}
  if row['training']['selected_epoch']!=winner or row['training']['epochs']!=cfg['epochs']:raise ValueError('Wrong model metadata')
  params=sum(v.numel() for v in ck['state'].values())
  if params!=row['training']['extra_parameters']:raise ValueError('Wrong parameter count')
  for shot in ('1','5'):
   v=raw_values(read(root/'scores'/f'confirmation_{shot}_{name}.json'),tasks['groups'][f'confirmation/{shot}'],sha);count+=len(v);p=paired(v,reference[shot]);pairs[shot]=p
   if row['shots'][shot]!={'accuracy':summary(v),'paired':p}:raise ValueError('Confirmation difference mismatch')
  eligible=all(p['low']>0 and p['mean']>=.01 for p in pairs.values())
  if row['eligible'] is not eligible:raise ValueError('Incorrect eligibility gate')
  recomputed[name]=pairs
 selected=min(recomputed,key=lambda n:(-min(p['low'] for p in recomputed[n].values()),n))
 if result['selected']!=selected or result['eligible'] is not result['candidates'][selected]['eligible']:raise ValueError('Architecture choice changed')
 return {'status':'passed','kind':'learned_calibration','plan_sha256':sha,'raw_method_episode_predictions_recomputed':count,'epochs_recomputed':26,'test_accessed':False,'external_weights':False,'eligible':result['eligible']}


def pilot(root):
 plan,sha=check_plan(root);tasks=read(root/'tasks.json');result=read(root/'result.json')
 if digest_json(tasks)!=plan['tasks_content_sha256'] or result['plan_sha256']!=sha:raise ValueError('Changed task/plan')
 if len(plan['ids']['train'])!=6144 or len(plan['ids']['val'])!=1280:raise ValueError('Incorrect pilot subset')
 if any(not i.startswith('train/') for i in plan['ids']['train']) or any(not i.startswith('val/') for i in plan['ids']['val']):raise ValueError('Wrong pilot split')
 for group in tasks.values():validate_tasks(group,plan['ids']['val'])
 def values(file):
  record=read(file);scores={}
  if record['plan_sha256']!=sha or record['test_accessed'] is not False:raise ValueError('Wrong validation record')
  for shot,row in record['shots'].items():
   v=raw_values({**row,'plan_sha256':sha,'test_accessed':False},tasks[shot],sha,key='values');scores[shot]=v
   if row['summary']!=summary(v):raise ValueError('Pilot summary mismatch')
  mean=float(np.mean([np.mean(v) for v in scores.values()]))
  if abs(mean-record['selection_mean'])>1e-14:raise ValueError('Selection mean mismatch')
  return scores,mean
 initial,initial_mean=values(root/'initial.json');selected_scores={};count=sum(len(v) for v in initial.values())
 if result['initial']!={s:summary(v) for s,v in initial.items()}:raise ValueError('Original reference mismatch')
 for branch in plan['branches']:
  training=read(root/branch/'training.json')
  if training['status']!='complete' or len(training['history'])!=2 or training['plan_sha256']!=sha:raise ValueError('Incomplete pilot')
  scores=[initial];means=[initial_mean]
  for epoch in (1,2):
   v,m=values(root/branch/f'epoch{epoch}_validation.json');scores.append(v);means.append(m);count+=sum(len(x) for x in v.values())
   row=training['history'][epoch-1]
   if row['epoch']!=epoch or row['train_images']!=6144 or row['validation']!={s:summary(x) for s,x in v.items()} or abs(row['selection_mean']-m)>1e-14:raise ValueError('Pilot history mismatch')
  selected=int(np.argmax(means));chosen=scores[selected];selected_scores[branch]=chosen
  if training['best_epoch']!=selected or training['best_sha256']!=digest_file(root/branch/'best.pt'):raise ValueError('Wrong pilot selected checkpoint')
  r=result['branches'][branch]
  if r['selected_epoch']!=selected or r['validation']!={s:summary(v) for s,v in chosen.items()} or r['paired_vs_initial']!={s:paired(chosen[s],initial[s]) for s in ('1','5')}:raise ValueError('Pilot comparison mismatch')
 if result['paired_diversity_vs_control']!={s:paired(selected_scores['diversity'][s],selected_scores['control'][s]) for s in ('1','5')}:raise ValueError('Pilot branch contrast mismatch')
 if result.get('not_publishable') is not True:raise ValueError('Pilot must not be called publishable')
 return {'status':'passed','kind':'channel_diversity_pilot','plan_sha256':sha,'raw_method_episode_predictions_recomputed':count,'test_accessed':False,'external_weights':False,'not_publishable':True}


def local(root):
 plan,sha=check_plan(root);tasks=read(root/'tasks.json');result=read(root/'result.json')
 if digest_json(tasks)!=plan['tasks_content_sha256'] or result['plan_sha256']!=sha:raise ValueError('Changed tasks')
 for group in tasks.values():validate_tasks(group)
 count=0;eligibility=[]
 def values(phase,shot,name):
  nonlocal count
  record=read(root/'scores'/f'{phase}_{shot}_{name}.json');v=raw_values(record,tasks[f'{phase}/{shot}'],sha,key='values');count+=len(v);return v
 for shot in ('1','5'):
  reference={p:values(p,shot,'reference') for p in ('search','confirmation')};rows=result['shots'][shot]['candidates'];search={}
  for c in plan['candidates']:
   name=f"{c['normalization']}_{c['ridge']:g}";search[(c['normalization'],c['ridge'])]=values('search',shot,name)
  ranking=sorted(search,key=lambda n:(-np.mean(search[n]),n[0],n[1]))
  if [(r['normalization'],r['ridge']) for r in rows]!=ranking:raise ValueError('Local search ordering changed')
  for row in rows:
   key=(row['normalization'],row['ridge']);v=search[key]
   if row['search']!=summary(v) or row['search_paired']!=paired(v,reference['search']):raise ValueError('Local search summary mismatch')
   if key in ranking[:2]:
    name=f'{key[0]}_{key[1]:g}';v=values('confirmation',shot,name)
    if row['confirmation']!=summary(v) or row['confirmation_paired']!=paired(v,reference['confirmation']):raise ValueError('Local confirmation summary mismatch')
  selected=min(rows[:2],key=lambda r:(-r['confirmation_paired']['low'],r['normalization'],r['ridge']));eligible=selected['confirmation_paired']['low']>0 and selected['confirmation_paired']['mean']>=.01
  recorded=result['shots'][shot]
  if recorded['selected']!={'normalization':selected['normalization'],'ridge':selected['ridge']} or recorded['eligible'] is not eligible or recorded['reference']!={p:summary(v) for p,v in reference.items()}:raise ValueError('Local selection mismatch')
  eligibility.append(eligible)
 if result['eligible'] is not all(eligibility):raise ValueError('Incorrect local gate')
 return {'status':'passed','kind':'local_reconstruction','plan_sha256':sha,'raw_method_episode_predictions_recomputed':count,'test_accessed':False,'external_weights':False,'eligible':result['eligible']}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',required=True,choices=('learned','pilot','local'));p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory)
 report={'learned':learned,'pilot':pilot,'local':local}[a.kind](root)
 report['files']={str(x.relative_to(root)):digest_file(x) for x in sorted(root.rglob('*.json')) if x.name!='audit.json'}
 write(root/'audit.json',report);print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
