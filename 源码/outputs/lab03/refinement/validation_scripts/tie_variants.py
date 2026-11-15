from pathlib import Path
import json,sys,importlib
root=Path.cwd();sys.path.insert(0,str(root));out=root/'outputs/qa/lab03-refine-20261112'
from experiments.lab03.dataset import load_dataset,unpack
from experiments.lab03.domain import features,State,path_key
from experiments.lab03.search import FIXED,HYBRID_ML_WEIGHT
base=importlib.import_module('experiments.lab03.search')
source=(root/'experiments/lab03/search.py').read_text(encoding='utf-8')
source=source.replace('[(h_value(root),0,root)]','[(h_value(root),0,0,root)]').replace('heapq.heappop(heap)[2]','heapq.heappop(heap)[3]').replace('(child.depth+h_value(child),counter,child)','(child.depth+h_value(child),-child.depth,counter,child)')
variant_namespace={'__name__':'experiments.lab03._trial'};exec(compile(source,'<validation high-g tie-break>','exec'),variant_namespace)
trial=variant_namespace['search'];original=base.heuristic
models=json.loads((root/'outputs/lab03/models.json').read_text(encoding='utf-8'));ga=json.loads((root/'outputs/lab03/ga_history.json').read_text(encoding='utf-8'))
tasks=[(r['seed'],g,q) for r in load_dataset() if r['split']=='validation' for g,qs in [unpack(r)] for q in qs]
rows=[]
for mode in ('tie_only','proof_priority','proof_and_dead'):
 def h(g,s,q,w,model=None,llm=None,*,expert=True,feature_values=None):
  if not any(w):return original(g,s,q,w,model,llm,expert=expert,feature_values=feature_values)
  if g.goal(s,q):return 0.
  x=features(g,s,q,expert) if feature_values is None else feature_values
  if mode!='tie_only' and q.hops-s.depth==1 and x[1]==1:return 1.
  if mode=='proof_and_dead' and (x[5] or (q.hops==s.depth)):return 100.
  return original(g,s,q,w,model,llm,expert=expert,feature_values=x)
 variant_namespace['heuristic']=h
 results={}
 for method in ('astar_fixed','astar_ga','astar_ml','hybrid'):
  results[method]={}
  for budget in (4,8,16):
   res=[]
   for seed,g,q in tasks:
    w=ga[str(seed)]['weights'][:];w[5]=HYBRID_ML_WEIGHT
    res.append(trial(g,q,method,budget,w,models[str(seed)],seed,trace=False))
   results[method][budget]={'n':len(res),'solved':sum(r['valid'] for r in res),'expansions':sum(r['expanded'] for r in res),'checks':sum(r['total_edge_checks'] for r in res)}
 row={'mode':mode,'fixed_weights_from_previous_validation':True,'results':results};rows.append(row);print(json.dumps(row),flush=True)
(out/'validation-tie-break.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
