"""Validation-derived unanswerable queries; report separately from answerable rates."""
from pathlib import Path
import sys,json,types,statistics
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.lab03.dataset import load_dataset,unpack
from experiments.lab03.domain import Query
from experiments.lab03.search import HYBRID_ML_WEIGHT
OUT=ROOT/'outputs/lab03/reliability';SOURCE=__import__('experiments.lab03.history_three',fromlist=['stage_three_bytes']).stage_three_bytes('experiments/lab03/search.py').decode('utf-8-sig')
models=json.loads((OUT/'previous_models.json').read_text(encoding='utf-8'));gas=json.loads((OUT/'previous_ga_history.json').read_text(encoding='utf-8'))
results=[]
for variant in ['baseline','known_suffix']:
 source=SOURCE
 if variant=='known_suffix':
  needle="                child=graph.apply(state,edge);generated+=1;counter+=1;record(child,state,'generated')"
  source=source.replace(needle,"                child=graph.apply(state,edge);generated+=1;counter+=1\n                if method!='bfs' and any(w) and ((child.depth==query.hops and not graph.goal(child,query)) or (child.depth<query.hops and (state_features(child)[5]==1 or (query.hops-child.depth==1 and state_features(child)[1]==0)))):\n                    terminal_mark(child)\n                    record(child,state,'dead_end')\n                    continue\n                record(child,state,'generated')")
 ns={'__name__':'experiments.lab03._unsat_validation','__package__':'experiments.lab03'};exec(compile(source,variant,'exec'),ns)
 rows=[]
 for record in load_dataset():
  if record['split']!='validation':continue
  g,qs=unpack(record);seed=record['seed'];w=gas[str(seed)]['weights'][:];w[5]=HYBRID_ML_WEIGHT
  for q in qs:
   # A sector absent from the graph is an explicitly generated negative. It is
   # not included in answer accuracy tables, nor claimed an unseen benchmark.
   q=Query(q.id+'-unsat',q.start,q.relations,'development-absent-sector')
   for budget in [8,64]:
    for method in ['astar_fixed','hybrid']:
     r=ns['search'](g,q,method,budget,w,models[str(seed)],seed,trace=False)
     rows.append({'query_id':q.id,'method':method,'budget':budget,**{k:r[k] for k in ('valid','status','expanded','total_edge_checks','adjacency_edge_scans','elapsed_ms')}})
 summary=[]
 for budget in [8,64]:
  for method in ['astar_fixed','hybrid']:
   rs=[r for r in rows if r['method']==method and r['budget']==budget]
   summary.append({'method':method,'budget':budget,'n':len(rs),'correct_exhausted':sum(r['status']=='exhausted' for r in rs),'invalid_found':sum(r['valid'] for r in rs),**{k:statistics.mean(r[k] for r in rs) for k in ['expanded','total_edge_checks','adjacency_edge_scans','elapsed_ms']}})
 result={'variant':variant,'scope':'Validation-derived negative development fixtures, NOT held-out answer accuracy','summary':summary,'rows':rows};results.append(result);print(variant,summary,flush=True)
(OUT/'validation-pruning-negative.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
