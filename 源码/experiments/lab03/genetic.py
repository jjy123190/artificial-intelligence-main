"""GA selects heuristic weights only on graph-disjoint validation questions."""
import random,statistics,math,json,hashlib
from .dataset import unpack,ROOT
from .search import search,FIXED
from .artifacts import provenance


def warm_start(records, seed):
    """An explicitly archived validation-only incumbent, not a hidden test hint."""
    config = json.loads((ROOT/'data/lab03/ga_warm_start.json').read_text(encoding='utf-8'))
    if config['dataset_sha256'] != hashlib.sha256((ROOT/'data/lab03/benchmark.json').read_bytes()).hexdigest():
        raise ValueError('GA热启动与当前数据集不一致')
    record = config['seed_records'].get(str(seed))
    graph_ids = sorted(r['graph']['id'] for r in records if r['seed']==seed and r['split']=='validation')
    if record is None or graph_ids != record['validation_graph_ids']:
        raise ValueError('GA热启动只能用于其来源验证图；其他图须显式从头优化')
    return [list(record['weights'])]


def optimize(records,seed,population=14,generations=10,budget=8,initial_weights=()):
    if type(seed) is not int or type(population) is not int or population < 3 or type(generations) is not int or generations < 0:
        raise ValueError('GA需要整数种子、至少3个个体及非负代数')
    if type(budget) is not int or not 1 <= budget <= 256:raise ValueError('GA扩展预算必须是1–256的整数')
    if (not isinstance(initial_weights,(list,tuple)) or len(initial_weights) >= population
            or any(not isinstance(w,(list,tuple)) or len(w)!=6 or w[-1]!=0
                   or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=5 for x in w)
                   for w in initial_weights)):
        raise ValueError('GA初始候选必须是合法的六维非负权重，学习项为0且少于种群大小')
    rng=random.Random(seed);tasks=[]
    for record in records:
        if record['seed']==seed and record['split']=='validation':
            graph,queries=unpack(record);tasks.extend((graph,q) for q in queries)
    if not tasks:raise ValueError('该种子没有独立验证图，不能进行GA调参')
    cache={}
    def score(weights):
        key=tuple(weights)
        if key not in cache:
            results=[search(g,q,'astar_ga',budget,weights=weights,trace=False) for g,q in tasks]
            success=statistics.mean(float(r['valid']) for r in results)
            work=statistics.mean(r['expanded']/budget for r in results)
            cache[key]=(success-.12*work,success,work)
        return cache[key]
    pop=[list(FIXED)]+[list(w) for w in initial_weights]+[[rng.uniform(0,4) for _ in range(5)]+[0.] for _ in range(population-1-len(initial_weights))]
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
    return {'provenance':provenance(),'seed':seed,'weights':history[-1]['weights'],'history':history,'population':population,'generations':generations,'budget':budget,
            'fitness':'validation valid answer rate - 0.12 * mean(expansions/budget)',
            'validation_graph_ids':sorted({g.id for g,q in tasks}),'validation_queries':len(tasks),'unique_candidates':len(cache),'initial_weights':[list(w) for w in initial_weights],
            'selection_note':'保留固定h与显式验证集热启动候选；精确权重作缓存键。验证预算8与主比较一致；仅验证集选择h参数；测试与OOD结果不参与适应度或代数选择。'}