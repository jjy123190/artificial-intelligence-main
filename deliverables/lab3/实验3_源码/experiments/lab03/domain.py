"""Serializable graph/query/state models. Gold answers are never search inputs."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib, json, math

RELATIONS = {"HOLDS": "持股", "GUARANTEES": "担保", "TRADES": "交易"}
FEATURE_NAMES = ["剩余跳数", "局部后缀潜力", "有效后继可选性", "路径证据可信度", "阶段化终点匹配", "有限后缀死路"]

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
        nodes, edges = list(nodes), list(edges)
        if any(not isinstance(n, Node) or not isinstance(n.id, str) or not n.id
               or not isinstance(n.name, str) or not n.name or not isinstance(n.kind, str) or n.kind not in {'Company', 'Person'}
               or not isinstance(n.sector, str) or type(n.listed) is not bool for n in nodes):
            raise ValueError("实体编号、类型、名称或上市属性不合法")
        if any(not isinstance(e, Edge) or not isinstance(e.id, str) or not e.id
               or not isinstance(e.source, str) or not isinstance(e.target, str)
               or not isinstance(e.relation, str) or type(e.verified) is not bool or type(e.active) is not bool
               or type(e.confidence) not in (int, float) or not math.isfinite(e.confidence)
               or type(e.value) not in (int, float) or not math.isfinite(e.value) for e in edges):
            raise ValueError("关系编号、证据标记或数值不合法")
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

    def valid_state(self, state):
        """Check the whole witness, including callers outside the search engines."""
        if (not isinstance(state, State) or not isinstance(state.nodes, tuple)
                or not isinstance(state.edges, tuple) or not state.nodes
                or len(state.nodes) != len(state.edges) + 1
                or any(not isinstance(n, str) or n not in self.nodes for n in state.nodes)
                or any(not isinstance(e, str) or e not in self.edges for e in state.edges)):
            return False
        if len(set(state.nodes)) != len(state.nodes):
            return False
        return all((self.edges[e].source, self.edges[e].target) == (state.nodes[i], state.nodes[i+1])
                   for i, e in enumerate(state.edges))

    def apply(self, state, edge):
        if (not self.valid_state(state) or not isinstance(edge, Edge)
                or not isinstance(edge.id, str) or self.edges.get(edge.id) != edge or edge.source != state.node):
            raise ValueError("转移必须使用完整有效历史和当前实体在图中的真实出边")
        if edge.target in state.nodes:
            raise ValueError("证据路径不能重复访问实体")
        return State(state.nodes + (edge.target,), state.edges + (edge.id,))

    def goal(self,state,query):
        if not self.valid_prefix(state, query, expert=False) or state.depth != query.hops:
            return False
        n = self.nodes.get(state.node)
        return bool(n and n.kind == "Company" and n.sector == query.sector and n.listed == query.listed)

    def valid_prefix(self, state, query, expert=True):
        if (not isinstance(query,Query) or not isinstance(query.relations,tuple)
                or any(not isinstance(r,str) or r not in RELATIONS for r in query.relations)
                or not isinstance(query.start,str) or type(query.listed) is not bool
                or not isinstance(query.sector,str) or type(expert) is not bool):
            return False
        if not self.valid_state(state) or state.nodes[0] != query.start or state.depth > query.hops:
            return False
        for i, eid in enumerate(state.edges):
            edge = self.edges.get(eid)
            if (not edge or (edge.source, edge.target, edge.relation) !=
                    (state.nodes[i], state.nodes[i + 1], query.relations[i])
                    or (expert and not expert_legal(edge))):
                return False
        return True

    def state_for_edges(self, query, edge_ids):
        if not isinstance(edge_ids, (list, tuple)) or any(not isinstance(e, str) for e in edge_ids):
            raise ValueError("路径必须是实际边编号序列")
        state = State((query.start,))
        for eid in edge_ids:
            edge = self.edges.get(eid)
            if not edge:
                raise ValueError("路径包含图中不存在的边")
            state = self.apply(state, edge)
        if not self.valid_prefix(state, query, expert=False):
            raise ValueError("路径前缀与当前问题不一致")
        return state

    def valid_path(self, state, query):
        return self.valid_prefix(state, query) and self.goal(state, query)

    def to_dict(self):return {"id":self.id,"split":self.split,"nodes":[asdict(n) for n in self.nodes.values()],"edges":[asdict(e) for e in self.edges.values()]}
    @classmethod
    def from_dict(cls,d):return cls([Node(**n) for n in d['nodes']],[Edge(**e) for e in d['edges']],d.get('id',''),d.get('split',''))


def expert_legal(edge):
    """Facts are accepted only when independently verified, in-date, and >=0.55 confidence."""
    return edge.verified and edge.active and edge.confidence >= .55


def rejection_reason(edge):
    return "；".join(s for yes,s in [(not edge.verified,"未经核验"),(not edge.active,"已失效"),(edge.confidence<.55,"置信度不足")] if yes)


# Strictly bounded query-local suffix evidence, selected only on validation.
SUFFIX_PROBE_CAP = 2


def features(graph, state, query, expert=True, account=None, *, candidates=None, prepare=None, probe=None):
    """Local evidence features with at most TWO child prefixes for a two-hop suffix.

    Only states with exactly two remaining relations use the extra probe. It never
    traverses an entire 3/4-hop query or consults answer labels. Observed suffix
    success is evidence, not a returned solution; queue pops remain mandatory.
    A capped/unfinished scan is UNKNOWN, not proof that a branch has no answer.
    ``prepare`` shares the search's full-prefix successor cache. All edge reads
    go through it; ``probe`` records the additional heuristic-only state visits.
    """
    node = graph.nodes[state.node]
    remaining = query.hops - state.depth
    def local_candidates(s):
        if prepare is not None:
            return prepare(s, True)[0]
        raw = graph.successors(s, query, False)
        if account is not None:
            account(len(raw))
        return [e for e in raw if not expert or expert_legal(e)]
    if candidates is None:
        candidates = local_candidates(state)
    neighbors = [graph.nodes[e.target] for e in candidates]
    topic = float(node.sector == query.sector)
    endpoint = float(node.kind == 'Company' and node.listed == query.listed)
    dead = float(remaining > 0 and not candidates)
    if remaining == 1:
        topic = endpoint = float(any(n.kind == 'Company' and n.sector == query.sector
                                     and n.listed == query.listed for n in neighbors))
    elif remaining > 1:
        topic = .5 * topic + .5 * float(any(n.sector == query.sector for n in neighbors))
        endpoint = 1 - (1 - endpoint) / remaining
    if remaining == 2:
        observed = False
        for edge in candidates[:SUFFIX_PROBE_CAP]:
            child = graph.apply(state, edge)
            if probe is not None:
                probe()
            following = local_candidates(child)
            if any(graph.nodes[e.target].kind == 'Company' and graph.nodes[e.target].sector == query.sector
                   and graph.nodes[e.target].listed == query.listed for e in following):
                observed = True
                break
        if observed:
            topic = endpoint = 1.; dead = 0.
        elif len(candidates) <= SUFFIX_PROBE_CAP:
            topic = endpoint = 0.; dead = 1.
        else:
            # topic==1 at remaining==2 is reserved for an observed valid suffix.
            topic = min(.5, topic)
    confidence = min([graph.edges[e].confidence for e in state.edges] or [1.0])
    return [remaining / 4.0, topic, min(len(candidates), 6) / 6.0,
            confidence, endpoint, dead]


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

def path_key(edge_ids):
    """A score belongs to a whole path prefix, not an edge in a different context."""
    return json.dumps(list(edge_ids), ensure_ascii=False, separators=(',', ':'))
