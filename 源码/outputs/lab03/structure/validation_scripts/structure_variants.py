"""Train/validation-only structural h candidates. Does not overwrite live code/results."""
from pathlib import Path
import json, sys, types, time, statistics
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.lab03 import search as sm, domain as dm, learning as lm, genetic as gm
from experiments.lab03.dataset import load_dataset, unpack
OUT=ROOT/'outputs/lab03/structure'
from experiments.lab03.history import stage_two_search, stage_two_bytes
preserved=stage_two_search()
BASE_FEATURES=sys.modules[preserved.__package__+'.domain'].features
SOURCE=stage_two_bytes('experiments/lab03/search.py').decode('utf-8-sig')
base_module=types.ModuleType('experiments.lab03._baseline_validation');base_module.__package__='experiments.lab03'
exec(compile(SOURCE,'preserved_baseline','exec'),base_module.__dict__);base_module.features=BASE_FEATURES
BASE_SEARCH=base_module.search
records=[r for r in load_dataset() if r['split'] in ('train','validation')]
assert {r['split'] for r in records}=={'train','validation'}

def suffix_features(graph,state,query,expert=True,account=None,*,candidates=None,prepare=None,cap=4,neutral=False):
    x=BASE_FEATURES(graph,state,query,expert,account,candidates=candidates)
    remaining=query.hops-state.depth
    if neutral and remaining>2:
        x[1]=.5;x[4]=1.
    if remaining!=2:return x
    if candidates is None:candidates=graph.successors(state,query,expert)
    for edge in candidates[:cap]:
        child=graph.apply(state,edge)
        following=prepare(child,True)[0] if prepare else graph.successors(child,query,expert)
        if any(graph.nodes[e.target].kind=='Company' and graph.nodes[e.target].sector==query.sector
               and graph.nodes[e.target].listed==query.listed for e in following):
            x[1]=x[4]=1.;x[5]=0.
            return x
    if len(candidates)<=cap:
        x[1]=x[4]=0.;x[5]=1.
    else:
        # No proof on a truncated scan is unknown, never proof of failure.
        # Reserve x[1]==1 for an actually observed suffix, not industry agreement.
        x[1]=min(.5,x[1])
    return x

results=[]
for name,cap,neutral in [('baseline',0,False),('suffix2-cap2',2,False),('suffix2-cap4',4,False),('suffix2-cap8',8,False),('suffix2-cap4-neutral',4,True),('suffix2-cap8-neutral',8,True)]:
    started=time.perf_counter()
    if cap:
        def feature(*args,**kwargs):return suffix_features(*args,**kwargs,cap=cap,neutral=neutral)
        module=types.ModuleType('experiments.lab03._structure_validation');module.__package__='experiments.lab03'
        code=SOURCE.replace('query, expert, candidates=candidates)','query, expert, candidates=candidates, prepare=prepare_successors)')
        code=code.replace('q.hops-s.depth == 1 and x[1] == 1','q.hops-s.depth in (1,2) and x[1] == 1').replace('return 1.0\n        vals=','return float(base)\n        vals=')
        exec(compile(code,name,'exec'),module.__dict__);module.features=feature
        engine=module.search;lm.features=feature;gm.search=engine
    else:
        engine=BASE_SEARCH;lm.features=BASE_FEATURES;gm.search=BASE_SEARCH
    models={str(seed):lm.fit_classifier(records,seed) for seed in (7,19,43)}
    ga={str(seed):gm.optimize(records,seed,initial_weights=gm.warm_start(records,seed)) for seed in (7,19,43)}
    rows=[]
    for record in records:
        if record['split']!='validation':continue
        graph,qs=unpack(record);seed=record['seed']
        for q in qs:
            for method in ('astar_fixed','astar_ga','hybrid'):
                w=ga[str(seed)]['weights'][:];w[5]=sm.HYBRID_ML_WEIGHT
                r=engine(graph,q,method,8,w,models[str(seed)],seed,trace=False)
                rows.append({'query_id':q.id,'method':method,**{k:r[k] for k in ('valid','expanded','total_edge_checks','elapsed_ms')}})
    summary=[]
    for method in ('astar_fixed','astar_ga','hybrid'):
        rs=[r for r in rows if r['method']==method]
        summary.append({'method':method,'n':len(rs),'valid':sum(r['valid'] for r in rs),**{k:statistics.mean(r[k] for r in rs) for k in ('expanded','total_edge_checks','elapsed_ms')}})
    value={'name':name,'cap':cap,'neutral_early_sector':neutral,'summary':summary,'ga':ga,'models':models,'rows':rows,'seconds':time.perf_counter()-started}
    results.append(value);(OUT/'validation-search-variants.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(name,summary,round(value['seconds'],2),flush=True)
