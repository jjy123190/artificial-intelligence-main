"""Validation-only exact-impossibility pruning. No extra answer oracle/lookahead."""
from pathlib import Path
import sys,json,types,statistics,time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.lab03.dataset import load_dataset,unpack
from experiments.lab03 import search as sm,genetic as gm,learning as lm
OUT=ROOT/'outputs/lab03/reliability'
records=[r for r in load_dataset() if r['split'] in ('train','validation')]
SOURCE=__import__('experiments.lab03.history_three',fromlist=['stage_three_bytes']).stage_three_bytes('experiments/lab03/search.py').decode('utf-8-sig')
results=[]
for variant in ('baseline','terminal','known_suffix'):
    start=time.perf_counter();source=SOURCE
    if variant!='baseline':
        needle="                child=graph.apply(state,edge);generated+=1;counter+=1;record(child,state,'generated')"
        check="child.depth==query.hops and not graph.goal(child,query)"
        if variant=='known_suffix':
            check="(child.depth==query.hops and not graph.goal(child,query)) or (child.depth<query.hops and (state_features(child)[5]==1 or (query.hops-child.depth==1 and state_features(child)[1]==0)))"
        replacement="                child=graph.apply(state,edge);generated+=1;counter+=1\n                if method!='bfs' and any(w) and ("+check+"):\n                    terminal_mark(child)\n                    record(child,state,'dead_end')\n                    continue\n                record(child,state,'generated')"
        assert needle in source;source=source.replace(needle,replacement)
    ns=types.ModuleType('experiments.lab03._proof_validation');ns.__package__='experiments.lab03';exec(compile(source,variant,'exec'),ns.__dict__)
    gm.search=ns.search
    models={str(seed):lm.fit_classifier(records,seed) for seed in (7,19,43)}
    ga={str(seed):gm.optimize(records,seed,initial_weights=gm.warm_start(records,seed)) for seed in (7,19,43)}
    rows=[]
    for record in records:
        if record['split']!='validation':continue
        graph,queries=unpack(record);seed=record['seed']
        for query in queries:
            for method in ('astar_fixed','astar_ga','astar_ml','hybrid'):
                w=ga[str(seed)]['weights'][:];w[5]=sm.HYBRID_ML_WEIGHT
                r=ns.search(graph,query,method,8,w,models[str(seed)],seed,trace=False)
                rows.append({'query_id':query.id,'method':method,**{k:r[k] for k in ('valid','expanded','total_edge_checks','adjacency_edge_scans','heuristic_probe_states','elapsed_ms')}})
    summary=[]
    for method in ('astar_fixed','astar_ga','astar_ml','hybrid'):
        rs=[r for r in rows if r['method']==method];summary.append({'method':method,'n':len(rs),'valid':sum(r['valid'] for r in rs),**{k:statistics.mean(r[k] for r in rs) for k in ('expanded','total_edge_checks','adjacency_edge_scans','heuristic_probe_states','elapsed_ms')}})
    value={'variant':variant,'summary':summary,'ga':ga,'rows':rows,'seconds':time.perf_counter()-start};results.append(value)
    (OUT/'validation-safe-pruning.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(variant,summary,round(value['seconds'],1),flush=True)
