from pathlib import Path
import sys,json,statistics,importlib,time
root=Path.cwd();sys.path.insert(0,str(root))
from experiments.lab03.dataset import load_dataset,unpack
from experiments.lab03.domain import State,features as original_features
S=importlib.import_module('experiments.lab03.search')
L=importlib.import_module('experiments.lab03.learning')
from experiments.lab03.genetic import optimize
OUT=root/'outputs/qa/lab03-optimize-20261112'
data=[r for r in load_dataset() if r['split'] in ('train','validation')]
tasks=[(r['seed'],g,q) for r in data if r['split']=='validation' for g,qs in [unpack(r)] for q in qs]

def feature_variant(mode):
    def feature(g,s,q):
        x=original_features(g,s,q);remaining=q.hops-s.depth
        if mode=='baseline':return x
        options=g.successors(s,q,True)
        x[2]=min(len(options),6)/6
        x[5]=float(remaining>0 and not options)
        if mode=='legal':return x
        if mode=='stage':
            # Only the endpoint is required to satisfy attributes; proxies weaken earlier.
            x[1]=1-(1-x[1])/max(1,remaining)
            x[4]=1-(1-x[4])/max(1,remaining)
            return x
        neighbors=[g.nodes[e.target] for e in options]
        match=lambda n:float(n.kind=='Company' and n.sector==q.sector and n.listed==q.listed)
        if mode=='onehop':
            if remaining==1:
                x[1]=max([match(n) for n in neighbors] or [0.])
                x[4]=x[1]
            elif remaining>1:
                x[1]=.5*x[1]+.5*max([float(n.sector==q.sector) for n in neighbors] or [0.])
                x[4]=1-(1-x[4])/remaining
            return x
        if mode=='branch_evidence':
            if remaining:
                x[1]=sum(n.sector==q.sector for n in neighbors)/max(1,len(neighbors))
                x[4]=sum(match(n) for n in neighbors)/max(1,len(neighbors)) if remaining==1 else 1-(1-x[4])/remaining
            return x
        raise ValueError(mode)
    return feature

results=[]
for mode in ['baseline','legal','stage','onehop','branch_evidence']:
    f=feature_variant(mode);S.features=f;L.features=f
    models={seed:L.fit_classifier(data,seed) for seed in (7,19,43)}
    for budget_ga in [12,8] if mode in ('baseline','stage','onehop') else [8]:
        ga={seed:optimize(data,seed,budget=budget_ga) for seed in (7,19,43)}
        metrics={}
        for method in ['bfs','astar_fixed','astar_ga','astar_ml','hybrid']:
            rows=[]
            for seed,g,q in tasks:
                w=list(ga[seed]['weights']);w[5]=1.5
                rows.append(S.search(g,q,method,8,w,models[seed],seed,trace=False))
            metrics[method]={'valid':sum(r['valid'] for r in rows),'n':len(rows),'expanded':statistics.mean(r['expanded'] for r in rows),
                             'elapsed_ms':statistics.mean(r['elapsed_ms'] for r in rows)}
        record={'mode':mode,'ga_budget':budget_ga,'validation':metrics,'weights':{seed:ga[seed]['weights'] for seed in ga}}
        results.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
        (OUT/'validation-search-sweep.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
