"""GA selects heuristic weights only on graph-disjoint validation questions."""
import random,statistics
from .dataset import unpack
from .search import search,FIXED


def optimize(records,seed,population=14,generations=10,budget=12):
    rng=random.Random(seed);tasks=[]
    for record in records:
        if record['seed']==seed and record['split']=='validation':
            graph,queries=unpack(record);tasks.extend((graph,q) for q in queries)
    cache={}
    def score(weights):
        key=tuple(round(x,5) for x in weights)
        if key not in cache:
            results=[search(g,q,'astar_ga',budget,weights=weights,trace=False) for g,q in tasks]
            success=statistics.mean(float(r['valid']) for r in results)
            work=statistics.mean(r['expanded']/budget for r in results)
            cache[key]=(success-.12*work,success,work)
        return cache[key]
    pop=[list(FIXED)]+[[rng.uniform(0,4) for _ in range(5)]+[0.] for _ in range(population-1)]
    history=[]
    for generation in range(generations+1):
        ranked=sorted(pop,key=lambda x:score(x)[0],reverse=True)
        best=ranked[0];fit,success,work=score(best)
        history.append({'generation':generation,'fitness':fit,'success_rate':success,'expanded_fraction':work,'weights':best[:]})
        if generation==generations:break
        nextpop=[x[:] for x in ranked[:2]]
        while len(nextpop)<population:
            a=max(rng.sample(ranked,3),key=lambda x:score(x)[0]);b=max(rng.sample(ranked,3),key=lambda x:score(x)[0])
            alpha=rng.random()
            child=[alpha*a[i]+(1-alpha)*b[i] for i in range(5)]+[0.]
            child=[max(0,min(5,x+rng.gauss(0,.65))) if rng.random()<.3 else x for x in child[:5]]+[0.]
            nextpop.append(child)
        pop=nextpop
    return {'seed':seed,'weights':history[-1]['weights'],'history':history,'population':population,'generations':generations,'budget':budget,
            'fitness':'validation valid answer rate - 0.12 * mean(expansions/budget)',
            'validation_graph_ids':sorted({g.id for g,q in tasks}),'validation_queries':len(tasks),'unique_candidates':len(cache),
            'selection_note':'仅验证集选择h参数；测试与OOD结果不参与适应度或代数选择。'}