from pathlib import Path
import json,sys,importlib
root=Path.cwd();sys.path.insert(0,str(root));out=root/'outputs/qa/lab03-refine-20261112'
from experiments.lab03.dataset import load_dataset,unpack
from experiments.lab03.domain import features,State,path_key
from experiments.lab03.search import FIXED,HYBRID_ML_WEIGHT
S=importlib.import_module('experiments.lab03.search')
# A proof that a state has exactly one hop left and a satisfying neighbor is not
# an ML score. It can be prioritized without fabricating the returned witness.
models=json.loads((root/'outputs/lab03/models.json').read_text(encoding='utf-8'));ga=json.loads((root/'outputs/lab03/ga_history.json').read_text(encoding='utf-8'))
tasks=[(r['seed'],g,q) for r in load_dataset() if r['split']=='validation' for g,qs in [unpack(r)] for q in qs]
original=S.heuristic
rows=[]
for mode in ('baseline','proof_priority','proof_and_dead','goal_on_generation'):
 if mode=='goal_on_generation':continue # Changes the charging protocol; do not compare as a free gain.
 def h(g,s,q,w,model=None,llm=None,*,expert=True,feature_values=None):
  x=features(g,s,q,expert) if feature_values is None else feature_values
  if mode!='baseline' and any(w) and q.hops-s.depth==1 and x[1]==1:return 1.
  if mode=='proof_and_dead' and any(w) and x[5]:return 100.
  return original(g,s,q,w,model,llm,expert=expert,feature_values=x)
 S.heuristic=h
 results={}
 for method in ('astar_fixed','astar_ga','astar_ml','hybrid'):
  results[method]={}
  for budget in (4,8,16):
   res=[]
   for seed,g,q in tasks:
    w=ga[str(seed)]['weights'][:];w[5]=HYBRID_ML_WEIGHT
    res.append(S.search(g,q,method,budget,w,models[str(seed)],seed,trace=False))
   results[method][budget]={'n':len(res),'solved':sum(r['valid'] for r in res),'expansions':sum(r['expanded'] for r in res),'checks':sum(r['total_edge_checks'] for r in res)}
 row={'mode':mode,'fixed_weights_from_previous_validation':True,'results':results};rows.append(row);print(json.dumps(row),flush=True)
(out/'validation-proof-priority.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
