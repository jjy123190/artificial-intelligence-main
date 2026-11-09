"""Independent BFS, A* and UCT/MCTS engines with common expansion budgets.

All query endpoints satisfy public predicates. No gold answer or exhaustive oracle
is called inside search. Learned/GA h is not claimed admissible. MCTS rollout work
is explicitly counted in addition to tree expansions.
"""
from __future__ import annotations
from collections import deque
import heapq,math,random,time
from .domain import State,features,expert_legal,rejection_reason,RELATIONS
from .learning import probability
FIXED=[1.7,.5,.6,.2,1.5,0.]
METHODS={'bfs':'BFS','astar_admissible':'A* 下界','astar_fixed':'A* 固定启发式','astar_ga':'A* 遗传优化','astar_ml':'A* 路径分类器','hybrid':'GA + 学习 + 专家','mcts':'MCTS / UCT','hybrid_no_expert':'混合（无专家）','hybrid_llm':'混合 + LLM'}

def heuristic(g,s,q,weights,model=None,llm=None):
    if g.goal(s,q):return 0.0
    x=features(g,s,q)
    vals=[1-x[1],1-x[2],1-x[3],1-x[4],x[5],1-probability(model,x)]
    base=q.hops-s.depth+sum(w*v for w,v in zip(weights,vals))
    if llm and s.edges:base += 2.0*(1-llm.get(s.edges[-1],.5))
    return base


def search(graph,query,method='bfs',budget=24,weights=None,model=None,seed=7,expert=True,llm_scores=None,trace=True):
    if query.start not in graph.nodes:raise ValueError('起始实体不存在')
    if not 2<=query.hops<=4:raise ValueError('只允许2–4跳查询')
    if method not in METHODS:raise ValueError('未知搜索方法')
    if type(budget) is not int or not 1<=budget<=256:raise ValueError('扩展预算必须是1–256的整数')
    if type(seed) is not int:raise ValueError('随机种子必须是整数')
    if any(r not in RELATIONS for r in query.relations):raise ValueError('查询包含未知关系')
    if type(query.listed) is not bool:raise ValueError('终点上市属性必须是布尔值')
    if weights is not None and (not isinstance(weights,(list,tuple)) or len(weights)!=6 or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in weights)):
        raise ValueError('启发式权重必须是6个有限非负数')
    if model is not None and (not isinstance(model,dict) or not isinstance(model.get('coef'),list) or len(model['coef'])!=6 or any(type(v) not in (int,float) or not math.isfinite(v) for v in model['coef']+[model.get('intercept')])):
        raise ValueError('路径分类器参数不完整或包含无效数值')
    if llm_scores is not None and (not isinstance(llm_scores,dict) or any(e not in graph.edges or type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for e,v in llm_scores.items())):
        raise ValueError('LLM评分必须绑定当前图中的边，且为0–1的有限数值')
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
        h=round(heuristic(graph,s,query,w,use_model,llm),5) if method not in {'bfs','mcts'} else None
        event={'id':event_id,'node':s.node,'depth':s.depth,'path':list(s.nodes),'edge_ids':list(s.edges),'parent_path':list(s.edges[:-1]),'status':status,'g':s.depth,'h':h,'f':round(s.depth+h,5) if h is not None else None,'p':round(probability(use_model,features(graph,s,query)),5) if use_model else None,**extra}
        if len(steps)>=1600:steps[-1]=event
        else:steps.append(event)
    def actions(s):
        nonlocal checked,pruned
        raw=graph.successors(s,query,False);checked+=len(raw)
        out=[]
        for e in raw:
            if expert and not expert_legal(e):
                pruned+=1;record(graph.apply(s,e),s,'rejected',reason=rejection_reason(e))
            else:out.append(e)
        return out
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
                vals=[max(.05,probability(use_model,features(graph,graph.apply(simulation,e),query))) for e in choices]
                e=rng.choices(choices,weights=vals)[0];parent=simulation;simulation=graph.apply(simulation,e);record(simulation,parent,'rollout')
            terminal_mark(simulation)
            reward=1. if graph.goal(simulation,query) else .25*probability(use_model,features(graph,simulation,query))
            if graph.goal(simulation,query):solution=simulation
            while node:
                node.n+=1;node.value+=reward
                if node.expanded and (not node.children or all(c.closed for c in node.children)):node.closed=True
                node=node.parent
            if solution:break
        frontier_remaining=not tree.closed
    else:
        queue=deque([root]);heap=[(heuristic(graph,root,query,w,use_model,llm),0,root)]
        while (queue if method=='bfs' else heap) and expanded<budget:
            state=queue.popleft() if method=='bfs' else heapq.heappop(heap)[2]
            expand_mark(state)
            terminal_mark(state)
            if graph.goal(state,query):solution=state;break
            for edge in actions(state):
                child=graph.apply(state,edge);generated+=1;counter+=1;record(child,state,'generated')
                if method=='bfs':queue.append(child)
                else:heapq.heappush(heap,(child.depth+heuristic(graph,child,query,w,use_model,llm),counter,child))
        frontier_remaining=bool(queue if method=='bfs' else heap)
    if solution:record(solution,None,'solution')
    elapsed=(time.perf_counter()-start)*1000
    valid=bool(solution and graph.valid_path(solution,query))
    return {'method':method,'label':METHODS[method],'status':'found' if solution else ('budget_exhausted' if frontier_remaining else 'exhausted'),
       'found':solution is not None,'valid':valid,'answer':solution.node if solution else None,
       'path':list(solution.nodes) if solution else [],'edge_ids':list(solution.edges) if solution else [],
       'expanded':expanded,'generated':generated,'successor_checks':checked,'expert_rejections':pruned,
       'rollout_steps':rollouts,'simulations':simulations,'total_work':expanded+rollouts,'repeated_states':repeated,
       'completed_paths':len(terminal_paths),'redundant_paths':redundant_paths,'redundant_path_ratio':redundant_paths/max(1,len(terminal_paths)),
       'redundancy_ratio':repeated/max(1,expanded),'hops':len(solution.edges) if solution else None,'elapsed_ms':elapsed,
       'budget':budget,'expert_enabled':expert,'trace':steps,'trace_events':trace_events,'trace_truncated':trace_events>len(steps),
       'classifier_used':use_model is not None,'heuristic_used':method not in {'bfs','mcts'},
       'note':'目标仅按问题谓词和图中实际路径验证；预算耗尽不等于无答案。GA/学习启发式不保证可采纳性。MCTS另计rollout开销。'}