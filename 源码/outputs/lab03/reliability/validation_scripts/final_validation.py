"""Final implementation parity on unchanged train/validation; never opens held-out data."""
from pathlib import Path
import json,sys,statistics
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.lab03.dataset import load_dataset,unpack
from experiments.lab03.search import search,HYBRID_ML_WEIGHT
from experiments.lab03.learning import fit_classifier
from experiments.lab03.genetic import optimize,warm_start
OUT=ROOT/'outputs/lab03/reliability'
records=[r for r in load_dataset() if r['split'] in ('train','validation')]
expected=next(v for v in json.loads((OUT/'validation-safe-pruning.json').read_text(encoding='utf-8')) if v['variant']=='known_suffix')
models={str(seed):fit_classifier(records,seed) for seed in [7,19,43]}
ga={str(seed):optimize(records,seed,initial_weights=warm_start(records,seed)) for seed in [7,19,43]}
for seed in ('7','19','43'):assert ga[seed]['weights']==expected['ga'][seed]['weights']
rows=[]
for record in records:
 if record['split']!='validation':continue
 g,qs=unpack(record);seed=record['seed'];w=ga[str(seed)]['weights'][:];w[5]=HYBRID_ML_WEIGHT
 for q in qs:
  for method in ('astar_fixed','astar_ga','astar_ml','hybrid'):
   r=search(g,q,method,8,w,models[str(seed)],seed,trace=False)
   e=next(x for x in expected['rows'] if x['query_id']==q.id and x['method']==method)
   assert all(r[k]==e[k] for k in ('valid','expanded','total_edge_checks','adjacency_edge_scans','heuristic_probe_states'))
   rows.append({'query_id':q.id,'method':method,**{k:r[k] for k in ('valid','expanded','total_edge_checks','proof_pruned','proof_checks','peak_frontier')}})
summary=[]
for method in ('astar_fixed','astar_ga','astar_ml','hybrid'):
 rs=[r for r in rows if r['method']==method]
 summary.append({'method':method,'n':len(rs),'valid':sum(r['valid'] for r in rs),**{k:statistics.mean(r[k] for r in rs) for k in ('expanded','proof_pruned','proof_checks','peak_frontier','total_edge_checks')}})
v={'status':'passed','ga_weights':{s:r['weights'] for s,r in ga.items()},'summary':summary,'rows':rows}
(OUT/'validation-final.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
