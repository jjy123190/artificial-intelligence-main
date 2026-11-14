from pathlib import Path
import sys,json,importlib,statistics,math
root=Path.cwd();sys.path.insert(0,str(root));out=root/'outputs/qa/lab03-refine-20261112'
from experiments.lab03.dataset import load_dataset,unpack
from experiments.lab03.domain import features,path_key
base=importlib.import_module('experiments.lab03.search')
source=(root/'experiments/lab03/search.py').read_text(encoding='utf-8').replace('[(h_value(root),0,root)]','[(h_value(root),0,0,root)]').replace('heapq.heappop(heap)[2]','heapq.heappop(heap)[3]').replace('(child.depth+h_value(child),counter,child)','(child.depth+h_value(child),-child.depth if any(w) else 0,counter,child)')
ns={'__name__':'experiments.lab03._frozen_trial'};exec(compile(source,'<validation>','exec'),ns)
original=base.heuristic
# All non-LLM choices are frozen before test evaluation.
def h(g,s,q,w,model=None,llm=None,*,expert=True,feature_values=None):
 if g.goal(s,q):return 0.
 x=features(g,s,q,expert) if feature_values is None and any(w) else feature_values
 if x is not None and q.hops-s.depth==1 and x[1]==1:return 1.
 return original(g,s,q,w,model,llm,expert=expert,feature_values=x)
ns['heuristic']=h
base.search=ns['search']
from experiments.lab03.genetic import optimize
from experiments.lab03.learning import fit_classifier
records=[r for r in load_dataset() if r['split'] in ('train','validation')]
tasks=[(r['seed'],g,q) for r in records if r['split']=='validation' for g,qs in [unpack(r)] for q in qs]
models={s:fit_classifier(records,s) for s in (7,19,43)}
ga={s:optimize(records,s) for s in (7,19,43)}
results={}
for method in ('astar_fixed','astar_ga','astar_ml','hybrid'):
 results[method]={}
 for budget in (4,8,16):
  runs=[]
  for seed,g,q in tasks:
   w=ga[seed]['weights'][:];w[5]=base.HYBRID_ML_WEIGHT
   runs.append(base.search(g,q,method,budget,w,models[seed],seed,trace=False))
  results[method][budget]={'solved':sum(r['valid'] for r in runs),'expanded':statistics.mean(r['expanded'] for r in runs),'checks':statistics.mean(r['total_edge_checks'] for r in runs)}
row={'variant':'exact_onehop_high_g_tie','n':len(tasks),'weights':{s:ga[s]['weights'] for s in ga},'results':results}
(out/'validation-selected-search.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(row,ensure_ascii=False),flush=True)
# Reuse real validation model calls from previous round; no test/OOD here.
values=json.loads((root/'outputs/lab03/optimization/validation-llm-values.json').read_text(encoding='utf-8'))['records']
selected={r['query'] for r in values}
summary=[]
for alpha in (0,.25,.5,1.,2.):
 def fusion(g,s,q,w,model=None,llm=None,*,expert=True,feature_values=None):
  score=h(g,s,q,w,model,None,expert=expert,feature_values=feature_values)
  x=feature_values
  if g.goal(s,q) or (x is not None and q.hops-s.depth==1 and x[1]==1):return score
  if llm and s.edges:score+=alpha*(1-llm.get(path_key(s.edges),.5))
  return score
 ns['heuristic']=fusion;runs=[]
 for seed,g,q in tasks:
  if q.id not in selected:continue
  scores={path_key(r['edges']):r['value_score']*(1 if r['prediction'] else .25) for r in values if r['query']==q.id}
  w=ga[seed]['weights'][:];w[5]=base.HYBRID_ML_WEIGHT
  res=ns['search'](g,q,'hybrid_llm',8,w,models[seed],seed,llm_scores=scores,trace=False)
  runs.append({'query':q.id,'valid':res['valid'],'expanded':res['expanded']})
 row={'alpha':alpha,'solved':sum(r['valid'] for r in runs),'expanded':sum(r['expanded'] for r in runs),'runs':runs};summary.append(row)
 print('Fusion',alpha,row['solved'],row['expanded'],flush=True)
(out/'validation-exact-fusion.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
