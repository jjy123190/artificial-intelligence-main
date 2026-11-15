"""Validate the final implementation against selected development candidate."""
from pathlib import Path
import json,sys,statistics,time,types
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.lab03.dataset import load_dataset,unpack
from experiments.lab03.search import search,HYBRID_ML_WEIGHT
from experiments.lab03.learning import fit_classifier
from experiments.lab03.genetic import optimize,warm_start
from experiments.lab03.history import stage_two_search,stage_two_bytes
OUT=ROOT/'outputs/lab03/structure'
records=[r for r in load_dataset() if r['split'] in ('train','validation')]
choice=next(v for v in json.loads((OUT/'validation-search-variants.json').read_text(encoding='utf-8')) if v['name']=='suffix2-cap2')
models={str(seed):fit_classifier(records,seed) for seed in (7,19,43)}
ga={str(seed):optimize(records,seed,initial_weights=warm_start(records,seed)) for seed in (7,19,43)}
# Inject observational counting only into preserved baseline; no ranking changes.
base=stage_two_search();ns={'__name__':base.__name__+'_cost','__package__':base.__package__}
code=stage_two_bytes('experiments/lab03/search.py').decode('utf-8-sig')
code=code.replace('successor_cache_hits = 0','successor_cache_hits = 0; adjacency_edge_scans=0')
code=code.replace('nonlocal checked, heuristic_edge_checks, successor_cache_hits','nonlocal checked, heuristic_edge_checks, successor_cache_hits, adjacency_edge_scans')
code=code.replace('        raw = graph.successors(s, query, False)','        adjacency_edge_scans += len(graph.out[s.node]) if s.depth < query.hops else 0\n        raw = graph.successors(s, query, False)')
code=code.replace("'total_edge_checks':checked+heuristic_edge_checks", "'adjacency_edge_scans':adjacency_edge_scans,'total_edge_checks':checked+heuristic_edge_checks")
exec(compile(code,'instrumented-stage-two','exec'),ns)
old_models=json.loads((OUT/'previous_models.json').read_text(encoding='utf-8'))
old_ga=json.loads((OUT/'previous_ga_history.json').read_text(encoding='utf-8'))
old_domain=sys.modules[base.__package__+'.domain']
rows=[]
for record in records:
    if record['split']!='validation':continue
    g,qs=unpack(record);seed=record['seed']
    for q in qs:
        for method in ('astar_fixed','astar_ga','hybrid'):
            w=ga[str(seed)]['weights'][:];w[5]=HYBRID_ML_WEIGHT
            r=search(g,q,method,8,w,models[str(seed)],seed,trace=False)
            expected=next(x for x in choice['rows'] if x['query_id']==q.id and x['method']==method)
            assert all(r[k]==expected[k] for k in ('valid','expanded','total_edge_checks'))
            w0=old_ga[str(seed)]['weights'][:];w0[5]=HYBRID_ML_WEIGHT
            old=ns['search'](old_domain.Graph.from_dict(g.to_dict()),old_domain.Query(**q.__dict__),method,8,w0,old_models[str(seed)],seed,trace=False)
            rows.append({'query_id':q.id,'method':method,'before_adjacency_edge_scans':old['adjacency_edge_scans'],**{k:r[k] for k in ('valid','expanded','total_edge_checks','adjacency_edge_scans','heuristic_probe_states')}})
for seed in ('7','19','43'):
    assert ga[seed]['weights']==choice['ga'][seed]['weights']
summary=[]
for method in ('astar_fixed','astar_ga','hybrid'):
    rs=[r for r in rows if r['method']==method]
    summary.append({'method':method,'valid':sum(r['valid'] for r in rs),'n':len(rs),**{k:statistics.mean(r[k] for r in rs) for k in ('expanded','total_edge_checks','adjacency_edge_scans','before_adjacency_edge_scans','heuristic_probe_states')}})
result={'status':'passed','data_splits':['train','validation'],'selected_candidate':'suffix2-cap2','ga_weights':{s:r['weights'] for s,r in ga.items()},'summary':summary,'rows':rows}
(OUT/'validation-final.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
