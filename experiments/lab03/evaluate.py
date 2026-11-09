"""Reproducible benchmark, ablations, GA histories, paired comparisons and failure cases."""
from __future__ import annotations
import csv,json,hashlib,statistics,time,random
from pathlib import Path
from .dataset import ROOT,load_dataset,unpack,shared_graph
from .domain import enumerate_paths,State,digest,features
from .learning import fit_classifier,classifier_evaluation
from .genetic import optimize
from .search import search,METHODS,FIXED
OUT=ROOT/'outputs/lab03'

def write(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvwrite(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(rows):
    groups={}
    for r in rows:groups.setdefault((r['split'],r['budget'],r['method']),[]).append(r)
    summary=[]
    for (split,budget,method),rs in groups.items():
        successes=[r for r in rs if r['valid']]
        per_seed=[statistics.mean(float(r['valid']) for r in rs if r['seed']==s) for s in sorted({r['seed'] for r in rs})]
        summary.append({'split':split,'budget':budget,'method':method,'label':METHODS[method],'n':len(rs),
            'success_rate':statistics.mean(float(r['valid']) for r in rs),'seed_success_sd':statistics.stdev(per_seed) if len(per_seed)>1 else 0.,
            'mean_expanded':statistics.mean(r['expanded'] for r in rs),'mean_total_work':statistics.mean(r['total_work'] for r in rs),
            'mean_successor_checks':statistics.mean(r['successor_checks'] for r in rs),'mean_hops_success':statistics.mean(r['hops'] for r in successes) if successes else None,
            'mean_redundant_paths':statistics.mean(r['redundant_path_ratio'] for r in rs),
            'mean_redundancy':statistics.mean(r['redundancy_ratio'] for r in rs),'mean_ms':statistics.mean(r['elapsed_ms'] for r in rs),
            'invalid_answer_rate':statistics.mean(float(r['found'] and not r['valid']) for r in rs),
            'budget_exhaustion_rate':statistics.mean(float(r['status']=='budget_exhausted') for r in rs)})
    return summary


def run_benchmark():
    OUT.mkdir(parents=True,exist_ok=True);data=load_dataset();started=time.perf_counter()
    models={};ga={};learning=[]
    for seed in [7,19,43]:
        models[str(seed)]=fit_classifier(data,seed);print('Fitted classifier',seed,flush=True)
        ga[str(seed)]=optimize(data,seed);print('Optimized GA',seed,ga[str(seed)]['history'][-1],flush=True)
        learning.append({'seed':seed,**classifier_evaluation(data,models[str(seed)])})
    write(OUT/'models.json',models);write(OUT/'ga_history.json',ga);write(OUT/'learning_metrics.json',learning)
    methods=['bfs','astar_admissible','astar_fixed','astar_ga','astar_ml','mcts','hybrid','hybrid_no_expert']
    rows=[];failures=[];traces=[]
    for record in data:
        if record['split'] not in ['test','ood']:continue
        graph,queries=unpack(record);seed=record['seed'];model=models[str(seed)];weights=ga[str(seed)]['weights'][:];weights[5]=1.5
        for q in queries:
            oracle=enumerate_paths(graph,q)
            for budget in [4,8,16]:
                for method in methods:
                    result=search(graph,q,method,budget,weights,model,seed,trace=False)
                    rows.append({'seed':seed,'split':record['split'],'graph_id':graph.id,'query_id':q.id,'hops_requested':q.hops,'answerable':bool(oracle),
                      **{k:result[k] for k in ['method','budget','valid','found','status','expanded','total_work','successor_checks','hops','redundancy_ratio','redundant_path_ratio','elapsed_ms','expert_rejections']}})
                    if budget==4 and method in ['astar_ga','hybrid','hybrid_no_expert'] and not result['valid'] and len(failures)<18:
                        failures.append({'query':q.__dict__,'graph_id':graph.id,'method':method,'status':result['status'],'oracle_paths':len(oracle),
                                         'diagnosis':'证据约束关闭后返回无效路径' if result['found'] else '有限预算未访问到有效目标；不得据此断言图无解', 'result':result})
        if record['split']=='test' and record['seed']==7 and len(traces)<4:
            q=queries[record['seed']%len(queries)]
            traces.append({'graph':graph.to_dict(),'query':q.__dict__,'result':search(graph,q,'hybrid',16,weights,model,seed)})
    summary=summarize(rows);csvwrite(OUT/'runs.csv',rows);csvwrite(OUT/'summary.csv',summary)
    write(OUT/'summary.json',summary);write(OUT/'failure_cases.json',failures);write(OUT/'search_examples.json',traces)
    # Paired question-level differences; seeds and graph split maintained.
    paired=[]
    for split in ['test','ood']:
        for budget in [4,8,16]:
            base={r['query_id']:r for r in rows if r['split']==split and r['budget']==budget and r['method']=='astar_fixed'}
            for method in ['astar_ga','hybrid','mcts']:
                rs=[r for r in rows if r['split']==split and r['budget']==budget and r['method']==method]
                diffs=[int(r['valid'])-int(base[r['query_id']]['valid']) for r in rs]
                # Bootstrap entire graphs, not individual correlated queries.
                bygraph={}
                for r,d in zip(rs,diffs):bygraph.setdefault(r['graph_id'],[]).append(d)
                rng=random.Random(123);clusters=list(bygraph.values())
                samples=sorted(statistics.mean(x for c in rng.choices(clusters,k=len(clusters)) for x in c) for _ in range(1000))
                paired.append({'split':split,'budget':budget,'method':method,'baseline':'astar_fixed','success_delta':statistics.mean(diffs),
                  'graph_bootstrap_low':samples[25],'graph_bootstrap_high':samples[974],'graph_clusters':len(clusters)})
    write(OUT/'paired_effects.json',paired)
    manifest={'created_at':time.strftime('%Y-%m-%dT%H:%M:%S'),'seeds':[7,19,43],'splits':{s:sum(r['split']==s for r in data) for s in ['train','validation','test','ood']},
      'questions':{s:sum(len(r['queries']) for r in data if r['split']==s) for s in ['train','validation','test','ood']},
      'budget_unit':'one popped BFS/A* state or expanded MCTS tree node; MCTS simulation transitions additionally counted in total_work',
      'label':'success = returned endpoint satisfies query AND each returned edge exists, is verified, active and confidence>=0.55',
      'data_note':'教学合成图，非真实企业准确率。shared场景来自前两次实验CSV；benchmark图相互独立。',
      'dataset_sha256':hashlib.sha256((ROOT/'data/lab03/benchmark.json').read_bytes()).hexdigest(),
      'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'experiments/lab03').glob('*.py'))},
      'timing_note':'单次串行测量，不含数据加载/训练；LLM推断耗时另列。机器负载会影响壁钟时间。','runs':len(rows),'elapsed_seconds':time.perf_counter()-started}
    write(OUT/'manifest.json',manifest)
    print('Benchmark complete:',len(rows),'runs',round(manifest['elapsed_seconds'],2),'seconds',flush=True)
    return summary


def run_llm():
    from .llm import score_actions,explain_verified_path,MODEL_ID,REVISION
    data=load_dataset();models=json.loads((OUT/'models.json').read_text(encoding='utf-8'));ga=json.loads((OUT/'ga_history.json').read_text(encoding='utf-8'))
    selected=[]
    for seed in [7,19,43]:
        r=next(r for r in data if r['split']=='test' and r['seed']==seed)
        graph,qs=unpack(r);selected.extend((seed,graph,q) for q in qs[:3])
    audit=[];rows=[];expl=[];cache={}
    for index,(seed,graph,q) in enumerate(selected):
        model=models[str(seed)];w=ga[str(seed)]['weights'][:];w[5]=1.5
        # Explicit bounded look-ahead: only first two layers, no goal traversal.
        edges=[];todo=[State((q.start,))]
        while todo and len(edges)<8:
            s=todo.pop(0)
            for e in graph.successors(s,q,False):
                if e not in edges:edges.append(e)
                if s.depth<1:todo.append(graph.apply(s,e))
                if len(edges)>=8:break
        records=score_actions(graph,q,edges)
        scores={r['edge_id']:r['score'] if r['legal'] else .25*r['score'] for r in records}
        llm_ms=sum(r['inference_ms'] for r in records)
        variants=[('hybrid',None),('hybrid_llm',scores)]
        for method,values in variants:
            result=search(graph,q,method,8,w,model,seed,llm_scores=values)
            rows.append({'seed':seed,'split':'test_llm_subset','graph_id':graph.id,'query_id':q.id,**{k:result[k] for k in ['method','budget','valid','found','status','expanded','total_work','successor_checks','hops','redundancy_ratio','redundant_path_ratio','elapsed_ms']},
                         'llm_ms':llm_ms if method=='hybrid_llm' else 0.,'scored_edges':len(records) if method=='hybrid_llm' else 0})
            if method=='hybrid_llm' and result['valid'] and len(expl)<3:expl.append({'graph_id':graph.id,'query_id':q.id,'path':result['path'],**explain_verified_path(graph,q,result)})
        audit.append({'graph_id':graph.id,'query_id':q.id,'query':q.__dict__,'evaluations':records,'scores':scores})
        cache[q.id]={'scores':scores,'model':MODEL_ID,'revision':REVISION,'graph_sha256':digest(graph.to_dict()),'query_sha256':digest(q.__dict__),'scored_edges':len(records),'llm_ms':llm_ms}
        print('LLM task',index+1,'/',len(selected),q.id,round(llm_ms/1000,2),'s',flush=True)
        write(OUT/'llm_audit.json',audit);write(OUT/'llm_cache.json',cache);write(OUT/'llm_runs.json',rows);write(OUT/'llm_explanations.json',expl)
    write(OUT/'llm_summary.json',{'model':MODEL_ID,'revision':REVISION,'questions':len(selected),'selection':'first test graph per seed; first 3 questions, balanced 2/3/4 hops; fixed before evaluation',
       'lookahead':'score at most 8 existing edges within first two levels; remaining edges use neutral score; lookahead cost = scored_edges, reported separately',
       'results':summarize(rows),'total_inference_ms':sum(r['llm_ms'] for r in rows),
       'legal_checks':sum(len(a['evaluations']) for a in audit),
       'legal_check_accuracy':statistics.mean(r['legal']==r['expert_legal'] for a in audit for r in a['evaluations']),
       'explanation_grounded':sum(e['grounded'] for e in expl),'explanations':len(expl),
       'note':'小模型、9题配对小样本仅验证机制；不可外推到大型模型或全面性能。离线缓存重放不代表再次调用模型。'})
    # Controlled corrupt-score stress test is explicitly simulated, not a claimed model observation.
    stress=[]
    for seed,graph,q in selected:
        w=ga[str(seed)]['weights'][:];w[5]=1.5
        corrupted={e.id:(1. if not e.verified or not e.active else 0.) for e in graph.edges.values()}
        res=search(graph,q,'hybrid_llm',16,w,models[str(seed)],seed,llm_scores=corrupted)
        stress.append({'query_id':q.id,'valid':res['valid'],'found':res['found'],'expert_rejections':res['expert_rejections']})
    write(OUT/'hallucination_stress.json',{'source':'controlled injected wrong scores, NOT real LLM outputs','cases':stress,
       'invalid_answers':sum(r['found'] and not r['valid'] for r in stress)})