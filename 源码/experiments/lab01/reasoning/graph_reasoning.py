"""基于知识图谱的推理：股权穿透、受益所有人识别、环路检测、关联路径查询。

数值累乘类推理（间接持股 = 路径上各层比例之积，再对多条路径求和）用产生式规则很难表达，
在图上做路径遍历则非常自然 —— 这正是不同表示方法各有所长的体现。
"""
from collections import defaultdict

import networkx as nx


def penetrate(kg, company, max_depth=10, threshold=0.25):
    """股权穿透：向上遍历所有持股路径，计算每个最终股东的穿透持股比例。
    threshold=25% 仅为教学筛查阈值，不构成受益所有人的完整法律认定。
    仅自然人可成为筛查候选；深度截断的企业仍返回但不标记为候选。
    环路按简单路径截断，不求解交叉持股下的无限级数。"""
    H = kg.relation_graph("HOLDS")
    owners = defaultdict(lambda: {"ratio": 0.0, "paths": []})

    def dfs(node, ratio, path):
        parents = list(H.predecessors(node))
        if node != company and (not parents or len(path) > max_depth):
            owners[node]["ratio"] += ratio
            owners[node]["paths"].append((list(reversed(path)), ratio))
            return
        for p in parents:
            if p not in path:                      # 遇到环路即截断
                dfs(p, ratio * H[p][node]["ratio"], path + [p])

    dfs(company, 1.0, [company])
    result = [dict(owner=o, name=kg.name(o), ratio=v["ratio"], paths=v["paths"],
                   beneficial=kg.G.nodes[o]["type"] == "Person" and v["ratio"] >= threshold) for o, v in owners.items()]
    return sorted(result, key=lambda x: -x["ratio"])


def upstream_paths(H, company, max_depth):
    """列出 max_depth 层以内的所有上游持股路径（与 SQL 查询语义一致，供效率对比）。"""
    rows = []

    def dfs(node, ratio, depth, path):
        if depth == max_depth:
            return
        for p in H.predecessors(node):
            if p in path:
                continue
            r = ratio * H[p][node]["ratio"]
            rows.append((p, r, depth + 1))
            dfs(p, r, depth + 1, path | {p})

    dfs(company, 1.0, 0, {company})
    return rows


def find_cycles(kg, rel):
    """在指定关系子图上找环：HOLDS → 循环持股，GUARANTEES → 担保圈。"""
    return [c for c in nx.simple_cycles(kg.relation_graph(rel)) if len(c) > 1]


def relation_path(kg, a, b):
    """任意两个实体之间的最短关联路径（忽略方向、跨关系类型）。"""
    U = nx.Graph(kg.G)
    try:
        nodes = nx.shortest_path(U, a, b)
    except nx.NetworkXNoPath:
        return None
    steps = []
    for u, v in zip(nodes, nodes[1:]):
        if kg.G.has_edge(u, v):
            rel = next(iter(kg.G[u][v].values()))["rel"]
            steps.append((u, rel, v, "→"))
        else:
            rel = next(iter(kg.G[v][u].values()))["rel"]
            steps.append((u, rel, v, "←"))
    return steps
