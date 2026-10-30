"""知识图谱表示：模式层（Schema）+ 数据层（三元组），底层用 networkx 有向多重图存储。

同时负责把图谱"翻译"成一阶谓词事实，供产生式系统使用，
例如三元组 (王建国, 持股, 华鑫控股) 对应谓词 Holds(P1, C1, 0.8)。
"""
import networkx as nx

# ---------------- 模式层：定义实体类型、关系类型及其约束 ----------------
SCHEMA = {
    "entity_types": {"Person": "自然人", "Company": "企业"},
    # 关系名: (头实体类型, 尾实体类型, 中文名, 属性名)
    "relations": {
        "HOLDS": (("Person", "Company"), ("Company",), "持股", "ratio"),
        "GUARANTEES": (("Company",), ("Company",), "担保", "amount"),
        "TRADES": (("Company",), ("Company",), "交易", "amount"),
        "KIN": (("Person",), ("Person",), "亲属", "relation"),
        "CONTROLS": (("Person",), ("Company",), "实际控制", "rule"),
    },
}
INFERRED = {"CONTROLS"}     # 不来自原始数据，由推理得出后回写图谱（知识增强），属性记录推出它的规则


class EquityKG:
    def __init__(self, dataset):
        self.G = nx.MultiDiGraph()
        for pid, name in dataset["persons"]:
            self.G.add_node(pid, name=name, type="Person", category="自然人")
        for cid, name, category, capital in dataset["companies"]:
            self.G.add_node(cid, name=name, type="Company", category=category, capital=capital)
        for h, c, r in dataset["holdings"]:
            self.add_relation(h, "HOLDS", c, r)
        for a, b, m in dataset["guarantees"]:
            self.add_relation(a, "GUARANTEES", b, m)
        for a, b, m in dataset["trades"]:
            self.add_relation(a, "TRADES", b, m)
        for a, b, rel in dataset["kinships"]:
            self.add_relation(a, "KIN", b, rel)

    def add_relation(self, head, rel, tail, value):
        """插入一条三元组，并用模式层做类型校验（自顶向下构建）。"""
        head_types, tail_types, _, attr = SCHEMA["relations"][rel]
        if self.G.nodes[head]["type"] not in head_types or self.G.nodes[tail]["type"] not in tail_types:
            raise ValueError(f"违反模式层约束: {head} -{rel}-> {tail}")
        self.G.add_edge(head, tail, key=rel, rel=rel, **{attr: value})

    def name(self, node):
        return self.G.nodes[node]["name"] if node in self.G else str(node)

    def triples(self):
        """数据层：以 (实体1, 关系, 实体2/属性值) 三元组形式导出。"""
        out = []
        for u, v, d in self.G.edges(data=True):
            _, _, cn, attr = SCHEMA["relations"][d["rel"]]
            out.append((self.name(u), cn, self.name(v), d[attr]))
        return out

    def relation_graph(self, rel):
        """抽取单一关系的子图（DiGraph），便于做路径/环路分析。"""
        attr = SCHEMA["relations"][rel][3]
        H = nx.DiGraph()
        H.add_nodes_from(self.G.nodes(data=True))
        for u, v, d in self.G.edges(data=True):
            if d["rel"] == rel:
                H.add_edge(u, v, **{attr: d[attr]})
        return H

    def to_predicates(self):
        """知识图谱 → 一阶谓词事实集合（产生式系统的初始综合数据库）。"""
        facts = set()
        for n, d in self.G.nodes(data=True):
            facts.add((d["type"], n))                      # Person(P1) / Company(C1)
            if d["type"] == "Company":
                facts.add(("IsA", n, d["category"]))       # IsA(C3, 上市公司)
        pred = {"HOLDS": "Holds", "GUARANTEES": "Guarantee", "TRADES": "Trade", "KIN": "Kin",
                "CONTROLS": "Controls"}
        for u, v, d in self.G.edges(data=True):
            if d["rel"] in INFERRED:
                facts.add((pred[d["rel"]], u, v))          # Controls(P1, C3)，来源规则不进入谓词
            else:
                attr = SCHEMA["relations"][d["rel"]][3]
                facts.add((pred[d["rel"]], u, v, d[attr]))  # Holds(P1, C1, 0.8)
        return facts
