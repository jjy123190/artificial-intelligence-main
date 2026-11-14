from pathlib import Path
import sys,json,statistics,importlib,math
root=Path.cwd();sys.path.insert(0,str(root));out=root/'outputs/qa/lab03-refine-20261112'
from experiments.lab03.dataset import load_dataset,unpack
from experiments.lab03.domain import features,State
S=importlib.import_module('experiments.lab03.search')
from experiments.lab03.learning import fit_classifier,probability
from experiments.lab03.genetic import optimize
records=[r for r in load_dataset() if r['split'] in ('train','validation')]
tasks=[(r['seed'],g,q) for r in records if r['split']=='validation' for g,qs in [unpack(r)] for q in qs]
models={s:fit_classifier(records,s) for s in (7,19,43)}
original=S.heuristic
results=[]
for variant in ['baseline','remaining_scale','sqrt_remaining','proven_dead','dead_and_remaining']:
 def h(g,s,q,weights,model=None,llm=None,*,expert=True,feature_values=None):
  if g.goal(s,q):return 0.
  left=q.hops-s.depth;base=float(left)
  if any(weights):
   x=features(g,s,q,expert) if feature_values is None else feature_values
   vals=[1-x[1],1-x[2],1-x[3],1-x[4],x[5],1-probability(model,x)]
   penalty=sum(a*b for a,b in zip(weights,vals))
   if variant in ('remaining_scale','dead_and_remaining'):penalty*=left/q.hops
   if variant=='sqrt_remaining':penalty*=math.sqrt(left/q.hops)
   if variant in ('proven_dead','dead_and_remaining') and (not left or x[5] or (left==1 and not x[1])):penalty+=10
   base+=penalty
  if llm and s.edges:
   from experiments.lab03.domain import path_key
   base+=2*(1-llm.get(path_key(s.edges),.5))
  return base
 S.heuristic=h
 ga={s:optimize(records,s) for s in (7,19,43)}
 metrics={}
 for method in ['astar_fixed','astar_ga','astar_ml','hybrid']:
  metrics[method]={}
  for budget in (4,8,16):
   runs=[]
   for seed,g,q in tasks:
    w=list(ga[seed]['weights']);w[5]=S.HYBRID_ML_WEIGHT
    runs.append(S.search(g,q,method,budget,w,models[seed],seed,trace=False))
   metrics[method][budget]={'solved':sum(r['valid'] for r in runs),'expanded':statistics.mean(r['expanded'] for r in runs),'edge_checks':statistics.mean(r['total_edge_checks'] for r in runs)}
 row={'variant':variant,'n':len(tasks),'results':metrics,'weights':{s:ga[s]['weights'] for s in ga}}
 results.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
 (out/'validation-search-refinement.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
