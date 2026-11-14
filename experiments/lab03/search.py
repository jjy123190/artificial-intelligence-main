"""Independent BFS, A* and UCT/MCTS engines with common expansion budgets.

All query endpoints satisfy public predicates. No gold answer or exhaustive oracle
is called inside search. Learned/GA h is not claimed admissible. MCTS rollout work
is explicitly counted in addition to tree expansions.
"""
from __future__ import annotations
from collections import deque
import heapq,math,random,time,json
from .domain import State,Query,features,expert_legal,rejection_reason,RELATIONS,path_key
from .learning import probability
FIXED=[1.7,.5,.6,.2,1.5,0.]
# Selected on graph-disjoint validation data, not on test/OOD answers.
HYBRID_ML_WEIGHT = .25
LLM_VALUE_WEIGHT = 2.0
METHODS={'bfs':'BFS','astar_admissible':'A* 下界','astar_fixed':'A* 固定启发式','astar_ga':'A* 遗传优化','astar_ml':'A* 路径分类器','hybrid':'GA + 学习 + 专家','mcts':'MCTS / UCT','hybrid_no_expert':'混合（无专家）','hybrid_llm':'混合 + LLM'}

GA_METHODS = {'astar_ga', 'hybrid', 'hybrid_no_expert', 'hybrid_llm'}
ML_METHODS = {'astar_ml', 'hybrid', 'hybrid_no_expert', 'hybrid_llm'}

def heuristic(g,s,q,weights,model=None,llm=None,*,expert=True,feature_values=None):
    if g.goal(s,q):return 0.0
    base=q.hops-s.depth
    if any(weights):
        x=features(g,s,q,expert) if feature_values is None else feature_values
        # An observed one/two-edge completion is stronger evidence than a noisy
        # estimate. Still expand this prefix and its child; never return the
        # neighbor without paying the normal search expansion budget.
        if q.hops-s.depth in (1,2) and x[1] == 1:
            return float(base)
        vals=[1-x[1],1-x[2],1-x[3],1-x[4],x[5],1-probability(model,x)]
        base+=sum(w*v for w,v in zip(weights,vals))
    # A missing score must be genuinely neutral, not a +1 depth penalty.
    # This bounded signed adjustment is preference only, never a hard prune.
    if llm and s.edges:base = max(0.,base + LLM_VALUE_WEIGHT*(.5-llm.get(path_key(s.edges),.5)))
    return base


def search(graph,query,method='bfs',budget=24,weights=None,model=None,seed=7,expert=True,llm_scores=None,trace=True):
    if (not isinstance(query,Query) or not isinstance(query.start,str)
            or not isinstance(query.relations,tuple) or any(not isinstance(r,str) for r in query.relations)
            or not isinstance(query.id,str) or not query.id or not isinstance(query.sector,str)
            or not isinstance(query.text,str)):
        raise ValueError('查询必须包含有效编号、起点、关系元组与终点属性')
    if query.start not in graph.nodes:raise ValueError('起始实体不存在')
    if not 2<=query.hops<=4:raise ValueError('只允许2–4跳查询')
    if not isinstance(method,str) or method not in METHODS:raise ValueError('未知搜索方法')
    if type(budget) is not int or not 1<=budget<=256:raise ValueError('扩展预算必须是1–256的整数')
    if type(seed) is not int:raise ValueError('随机种子必须是整数')
    if type(expert) is not bool:raise ValueError('专家约束开关必须为布尔值')
    if method in GA_METHODS and weights is None:
        raise ValueError('该方法必须提供GA参数，不能静默使用固定启发式')
    if method in ML_METHODS and model is None:
        raise ValueError('该方法必须提供路径分类器，不能静默使用中性概率')
    if method == 'hybrid_llm' and (llm_scores is None or (not llm_scores and graph.successors(State((query.start,)), query, False))):
        raise ValueError('LLM方法必须提供当前问题的真实路径前缀评分')
    if any(r not in RELATIONS for r in query.relations):raise ValueError('查询包含未知关系')
    if type(query.listed) is not bool:raise ValueError('终点上市属性必须是布尔值')
    if weights is not None and (not isinstance(weights,(list,tuple)) or len(weights)!=6 or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in weights)):
        raise ValueError('启发式权重必须是6个有限非负数')
    if model is not None and (not isinstance(model,dict) or not isinstance(model.get('coef'),list) or len(model['coef'])!=6 or any(type(v) not in (int,float) or not math.isfinite(v) for v in model['coef']+[model.get('intercept')])):
        raise ValueError('路径分类器参数不完整或包含无效数值')
    if model is not None and (type(model.get('seed')) is not int or model['seed'] != seed):
        raise ValueError('路径分类器种子与本次搜索不一致')
    if llm_scores is not None:
        if not isinstance(llm_scores, dict):
            raise ValueError('LLM评分必须是路径前缀到有限数值的映射')
        for key, value in llm_scores.items():
            try:
                ids = json.loads(key)
                if not ids or path_key(ids) != key:
                    raise ValueError('noncanonical path')
                graph.state_for_edges(query, ids)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError('LLM评分必须绑定当前图和问题的真实路径前缀') from exc
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('LLM评分必须是0–1的有限数值')
    start=time.perf_counter();rng=random.Random(seed);root=State((query.start,))
    steps=[];trace_events=0;generated=1;expanded=0;checked=0;pruned=0;rollouts=0;simulations=0;repeated=0
    signatures=set();solution=None;counter=0;terminal_paths=set();terminal_endpoints=set();redundant_paths=0
    if method=='hybrid_no_expert':expert=False
    use_model=model if method in {'astar_ml','hybrid','hybrid_no_expert','hybrid_llm'} else None
    w=list(FIXED if weights is None else weights)
    if method in {'astar_fixed','mcts'}:w=list(FIXED)
    if method=='astar_admissible':w=[0.]*6
    if method=='astar_ml':w=[0,0,0,0,0,3.]
    if method=='astar_ga':w[5]=0.
    llm=llm_scores if method=='hybrid_llm' else None
    feature_cache = {}
    h_cache = {}
    successor_cache = {}
    heuristic_evaluations = 0
    heuristic_edge_checks = 0
    successor_cache_hits = 0
    heuristic_probe_states = 0
    adjacency_edge_scans = 0
    proof_pruned = 0
    proof_checks = 0
    peak_frontier = 1
    def probe_mark():
        nonlocal heuristic_probe_states
        heuristic_probe_states += 1
    def prepare_successors(s, for_heuristic=False):
        nonlocal checked, heuristic_edge_checks, successor_cache_hits, adjacency_edge_scans
        if s.edges in successor_cache:
            successor_cache_hits += 1
            return successor_cache[s.edges]
        # Include edges filtered out by relation/cycle checks, not only evidence candidates.
        if s.depth < query.hops:
            adjacency_edge_scans += len(graph.out[s.node])
        raw = graph.successors(s, query, False)
        if for_heuristic:
            heuristic_edge_checks += len(raw)
        else:
            checked += len(raw)
        allowed, rejected = [], []
        for edge in raw:
            (rejected if expert and not expert_legal(edge) else allowed).append(edge)
        # Query, graph and expert mode are constant for this search invocation;
        # no cache survives into another task or merges distinct full prefixes.
        successor_cache[s.edges] = (allowed, rejected)
        return allowed, rejected
    def state_features(s):
        nonlocal heuristic_evaluations
        if s.edges not in feature_cache:
            heuristic_evaluations += 1
            candidates, _ = prepare_successors(s, True)
            feature_cache[s.edges] = features(graph, s, query, expert, candidates=candidates,
                                               prepare=prepare_successors, probe=probe_mark)
        return feature_cache[s.edges]
    def h_value(s):
        if s.edges not in h_cache:
            # The admissible h0 needs no features. Cache per full prefix, not node.
            x = state_features(s) if any(w) and not graph.goal(s,query) else None
            h_cache[s.edges] = heuristic(graph, s, query, w, use_model, llm,
                                        expert=expert, feature_values=x)
        return h_cache[s.edges]
    def impossibility_reason(s):
        """Only exact negative facts already observable to this weighted A*.

        No LLM/classifier probability can delete a path. Unknown capped suffixes
        remain in the frontier. BFS/h0/MCTS retain their original blind semantics.
        """
        nonlocal proof_checks
        if method in {'bfs','mcts'} or not any(w):
            return ''
        proof_checks += 1
        if s.depth == query.hops:
            return '' if graph.goal(s,query) else '完整路径的终点不满足问题属性'
        x=state_features(s)
        if x[5] == 1:
            return '当前无合法后继，或有界后缀已完整核验且无法满足目标'
        if query.hops-s.depth == 1 and x[1] == 0:
            return '全部一步后继均不满足终点谓词'
        return ''

    def terminal_mark(s):
        nonlocal redundant_paths
        if s.depth==query.hops and s.edges not in terminal_paths:
            terminal_paths.add(s.edges)
            if s.node in terminal_endpoints:redundant_paths+=1
            terminal_endpoints.add(s.node)
    def record(s,parent,status,**extra):
        nonlocal trace_events
        if not trace:return
        event_id=trace_events;trace_events+=1
        if len(steps)>=1600 and status!='solution':return
        h=round(h_value(s),5) if method not in {'bfs','mcts'} and status not in {'rejected','proof_pruned'} else None
        event={'id':event_id,'node':s.node,'depth':s.depth,'path':list(s.nodes),'edge_ids':list(s.edges),'parent_path':list(s.edges[:-1]),'status':status,'g':s.depth,'h':h,'f':round(s.depth+h,5) if h is not None else None,'p':round(probability(use_model,feature_cache[s.edges]),5) if use_model and s.edges in feature_cache else None,**extra}
        if len(steps)>=1600:steps[-1]=event
        else:steps.append(event)
    def actions(s):
        nonlocal pruned
        allowed, rejected = prepare_successors(s)
        for edge in rejected:
            pruned += 1
            record(graph.apply(s,edge),s,'rejected',reason=rejection_reason(edge))
        return allowed
    def expand_mark(s):
        nonlocal expanded,repeated
        expanded+=1;signature=(s.node,s.depth)
        if signature in signatures:repeated+=1
        signatures.add(signature);record(s,None,'expanded')
    record(root,None,'start')
    if method=='mcts':
        class Tree:
            def __init__(self,state,parent=None):self.state=state;self.parent=parent;self.children=[];self.n=0;self.value=0.;self.expanded=False;self.closed=False
        tree=Tree(root)
        while expanded<budget and not tree.closed:
            simulations+=1;node=tree
            while node.expanded and node.children and not node.closed:
                available=[c for c in node.children if not c.closed]
                if not available:node.closed=True;break
                node=max(available,key=lambda c:(float('inf') if c.n==0 else c.value/c.n+1.414*math.sqrt(math.log(max(1,node.n))/c.n),-len(c.state.edges)))
            state=node.state
            terminal_mark(state)
            if graph.goal(state,query):solution=state;break
            if node.closed:
                if node.parent and all(c.closed for c in node.parent.children):node.parent.closed=True
                continue
            expand_mark(state);node.expanded=True
            options=actions(state)
            node.children=[Tree(graph.apply(state,e),node) for e in options];generated+=len(options)
            for child in node.children:record(child.state,state,'generated')
            if not options:node.closed=True
            simulation=state
            while simulation.depth<query.hops:
                choices=actions(simulation)
                if not choices:break
                rollouts+=1
                vals=[.5 for _ in choices]  # Pure MCTS: uniform rollout, no learned/heuristic look-ahead.
                e=rng.choices(choices,weights=vals)[0];parent=simulation;simulation=graph.apply(simulation,e);record(simulation,parent,'rollout')
            terminal_mark(simulation)
            reward=1. if graph.goal(simulation,query) else .125
            if graph.goal(simulation,query):solution=simulation
            while node:
                node.n+=1;node.value+=reward
                if node.expanded and (not node.children or all(c.closed for c in node.children)):node.closed=True
                node=node.parent
            if solution:break
        frontier_remaining=not tree.closed
    else:
        queue=deque([root]) if method=='bfs' else deque()
        heap=[] if method=='bfs' else [(h_value(root),0,0,root)]
        while (queue if method=='bfs' else heap) and expanded<budget:
            state=queue.popleft() if method=='bfs' else heapq.heappop(heap)[3]
            expand_mark(state)
            terminal_mark(state)
            if graph.goal(state,query):solution=state;break
            for edge in actions(state):
                child=graph.apply(state,edge);generated+=1;counter+=1
                reason=impossibility_reason(child)
                if reason:
                    proof_pruned+=1
                    terminal_mark(child)
                    record(child,state,'proof_pruned',reason=reason)
                    continue
                record(child,state,'generated')
                if method=='bfs':queue.append(child)
                else:heapq.heappush(heap,(child.depth+h_value(child),-child.depth if any(w) else 0,counter,child))
                peak_frontier=max(peak_frontier,len(queue if method=='bfs' else heap))
        frontier_remaining=bool(queue if method=='bfs' else heap)
    if solution:record(solution,None,'solution')
    elapsed=(time.perf_counter()-start)*1000
    valid=bool(solution and graph.valid_path(solution,query))
    return {'method':method,'label':METHODS[method],'status':'found' if solution else ('budget_exhausted' if frontier_remaining else 'exhausted'),
       'found':solution is not None,'valid':valid,'answer':solution.node if solution else None,
       'path':list(solution.nodes) if solution else [],'edge_ids':list(solution.edges) if solution else [],
       'expanded':expanded,'generated':generated,'successor_checks':checked,'expert_rejections':pruned,
       'rollout_steps':rollouts,'simulations':simulations,'total_work':expanded+rollouts,'repeated_states':repeated,
       'heuristic_evaluations':heuristic_evaluations,'heuristic_edge_checks':heuristic_edge_checks,
       'heuristic_probe_states':heuristic_probe_states,'adjacency_edge_scans':adjacency_edge_scans,
       'proof_pruned':proof_pruned,'proof_checks':proof_checks,
       'peak_frontier':peak_frontier if method!='mcts' else None,
       'total_edge_checks':checked+heuristic_edge_checks,'successor_cache_hits':successor_cache_hits,
       'successor_cache_entries':len(successor_cache),
       'completed_paths':len(terminal_paths),'redundant_paths':redundant_paths,'redundant_path_ratio':redundant_paths/max(1,len(terminal_paths)),
       'redundancy_ratio':repeated/max(1,expanded),'hops':len(solution.edges) if solution else None,'elapsed_ms':elapsed,
       'budget':budget,'expert_enabled':expert,'trace':steps,'trace_events':trace_events,'trace_truncated':trace_events>len(steps),
       'classifier_used':use_model is not None,'heuristic_used':method not in {'bfs','mcts'},
       'note':'目标仅按问题谓词和图中实际路径验证；预算耗尽不等于无答案。GA/学习启发式不保证可采纳性。一/两步已知目标延伸优先于估值，仍按出队付费；两步仅探测最多2个子前缀，heuristic_probe_states与边检查另计；有估值A*同分优先更深前缀，下界A*保留FIFO。候选与规则判断按完整前缀缓存；total_edge_checks仅计候选证据检查；adjacency_edge_scans另计含关系/环过滤的出边扫描；模型推断另计。total_work仅计扩展+模拟。未评分LLM前缀为零调整，真实评分最多±1且不能覆盖已知合法后缀。有估值A*可剪去已完整核验的无解后缀，截断未知分支不剪；proof_pruned不冒充专家拦截，证据检查/探测仍计费。'}