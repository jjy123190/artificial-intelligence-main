"""Serializable graph/query/state models. Gold answers are never search inputs."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib, json

RELATIONS = {"HOLDS": "持股", "GUARANTEES": "担保", "TRADES": "交易"}
FEATURE_NAMES = ["剩余跳数", "主题相关度", "下一步可选性", "路径证据可信度", "终点类型匹配", "死路提示"]

@dataclass(frozen=True)
class Node:
    id: str
    name: str
    kind: str = "Company"
    sector: str = "综合"
    listed: bool = False

@dataclass(frozen=True)
class Edge:
    id: str
    source: str
    target: str
    relation: str
    confidence: float = 1.0
    verified: bool = True
    active: bool = True
    value: float = 0.0

@dataclass(frozen=True)
class Query:
    id: str
    start: str
    relations: tuple[str, ...]
    sector: str
    listed: bool = True
    text: str = ""

    @property
    def hops(self): return len(self.relations)

@dataclass(frozen=True)
class State:
    nodes: tuple[str, ...]
    edges: tuple[str, ...] = ()

    @property
    def depth(self): return len(self.edges)
    @property
    def node(self): return self.nodes[-1]

class Graph:
    def __init__(self, nodes, edges, graph_id="", split=""):
        self.nodes = {n.id:n for n in nodes}
        self.edges = {e.id:e for e in edges}
        if len(self.nodes)!=len(nodes) or len(self.edges)!=len(edges):raise ValueError("重复节点/边编号")
        self.id, self.split = graph_id, split
        self.out = defaultdict(list)
        for e in edges:
            if e.source not in self.nodes or e.target not in self.nodes:raise ValueError("关系引用缺失实体")
            if e.relation not in RELATIONS:raise ValueError("未知关系")
            if not 0 <= e.confidence <= 1:raise ValueError("证据置信度越界")
            self.out[e.source].append(e)
        for edgeset in self.out.values():edgeset.sort(key=lambda e:e.id)

    def successors(self, state, query, expert=True):
        if state.depth >= query.hops:return []
        required=query.relations[state.depth]
        return [e for e in self.out[state.node] if e.relation==required and e.target not in state.nodes
                and (not expert or expert_legal(e))]

    def apply(self,state,edge):return State(state.nodes+(edge.target,), state.edges+(edge.id,))

    def goal(self,state,query):
        if not state.nodes or state.depth != query.hops:
            return False
        n = self.nodes.get(state.node)
        return bool(n and n.kind == "Company" and n.sector == query.sector and n.listed == query.listed)

    def valid_path(self,state,query):
        if not state.nodes or state.nodes[0]!=query.start or len(state.nodes)!=len(state.edges)+1:return False
        if state.depth != query.hops or any(n not in self.nodes for n in state.nodes):return False
        if len(set(state.nodes))!=len(state.nodes) or not self.goal(state,query):return False
        for i,eid in enumerate(state.edges):
            e=self.edges.get(eid)
            if not e or (e.source,e.target,e.relation)!=(state.nodes[i],state.nodes[i+1],query.relations[i]) or not expert_legal(e):return False
        return True

    def to_dict(self):return {"id":self.id,"split":self.split,"nodes":[asdict(n) for n in self.nodes.values()],"edges":[asdict(e) for e in self.edges.values()]}
    @classmethod
    def from_dict(cls,d):return cls([Node(**n) for n in d['nodes']],[Edge(**e) for e in d['edges']],d.get('id',''),d.get('split',''))


def expert_legal(edge):
    """Facts are accepted only when independently verified, in-date, and >=0.55 confidence."""
    return edge.verified and edge.active and edge.confidence >= .55


def rejection_reason(edge):
    return "；".join(s for yes,s in [(not edge.verified,"未经核验"),(not edge.active,"已失效"),(edge.confidence<.55,"置信度不足")] if yes)


def features(graph,state,query):
    n=graph.nodes[state.node]
    remaining=query.hops-state.depth
    candidates=graph.successors(state,query,False)
    confidence=min([graph.edges[e].confidence for e in state.edges] or [1.0])
    return [remaining/4.0, float(n.sector==query.sector), min(len(candidates),6)/6.0,
            confidence, float(n.kind=='Company' and n.listed==query.listed),float(remaining>0 and not candidates)]


def enumerate_paths(graph,query,expert=True):
    """Independent exhaustive oracle, used for labels/evaluation only, never by search."""
    found=[]
    def visit(state):
        if state.depth==query.hops:
            if graph.goal(state,query):found.append(state)
            return
        for edge in graph.successors(state,query,expert):visit(graph.apply(state,edge))
    visit(State((query.start,)))
    return found


def digest(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()