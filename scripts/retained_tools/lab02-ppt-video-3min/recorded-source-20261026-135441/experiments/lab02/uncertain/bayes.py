"""二值贝叶斯网络：完整 CPT、变量消元、穷举核验与星形网络解析推理。

股权模型为 Risk→四个观测指标的朴素贝叶斯网络，明确假设指标在 Risk 给定时独立。
这是教学简化，不把模糊隶属度当概率。连续指标先按阈值二值化，缺测节点边缘化。
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from core.kb.features import FEATURES, RiskObservation


class ImpossibleEvidence(ValueError):
    pass


def _bits(n):
    return itertools.product((0, 1), repeat=n)


@dataclass(frozen=True)
class Factor:
    variables: tuple[str, ...]
    values: dict[tuple[int, ...], float]

    def restrict(self, evidence: dict[str, int]) -> "Factor":
        keep = [i for i, v in enumerate(self.variables) if v not in evidence]
        values = {}
        for key, value in self.values.items():
            if all(v not in evidence or evidence[v] == key[i] for i, v in enumerate(self.variables)):
                values[tuple(key[i] for i in keep)] = value
        return Factor(tuple(self.variables[i] for i in keep), values)

    def multiply(self, other: "Factor") -> "Factor":
        variables = self.variables + tuple(v for v in other.variables if v not in self.variables)
        a = [variables.index(v) for v in self.variables]
        b = [variables.index(v) for v in other.variables]
        return Factor(variables, {bits: self.values.get(tuple(bits[i] for i in a), 0.0)
                                 * other.values.get(tuple(bits[i] for i in b), 0.0)
                                 for bits in _bits(len(variables))})

    def sum_out(self, variable: str) -> "Factor":
        if variable not in self.variables:
            return self
        i = self.variables.index(variable)
        out = {}
        for key, value in self.values.items():
            small = key[:i] + key[i + 1:]
            out[small] = out.get(small, 0.0) + value
        return Factor(self.variables[:i] + self.variables[i + 1:], out)


@dataclass(frozen=True)
class BinaryNode:
    name: str
    parents: tuple[str, ...]
    p_true: dict[tuple[int, ...], float]

    def __post_init__(self):
        if not self.name or len(set(self.parents)) != len(self.parents):
            raise ValueError("节点名无效或父节点重复")
        if set(self.p_true) != set(_bits(len(self.parents))):
            raise ValueError(f"{self.name} 的 CPT 不完整")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in self.p_true.values()):
            raise ValueError("CPT 概率须在 [0,1]")

    def factor(self) -> Factor:
        return Factor((*self.parents, self.name),
                      {(*parents, state): p if state else 1 - p
                       for parents, p in self.p_true.items() for state in (0, 1)})


class BinaryBayesNet:
    def __init__(self, nodes: list[BinaryNode]):
        lookup = {n.name: n for n in nodes}
        if len(lookup) != len(nodes):
            raise ValueError("贝叶斯节点重名")
        for n in nodes:
            if any(p not in lookup for p in n.parents):
                raise ValueError(f"节点 {n.name} 引用未知父节点")
        ordered, todo = [], list(nodes)
        while todo:
            ready = [n for n in todo if all(p in {v.name for v in ordered} for p in n.parents)]
            if not ready:
                raise ValueError("贝叶斯网络必须是有向无环图")
            ordered.extend(ready)
            todo = [n for n in todo if n not in ready]
        self.nodes, self.names = ordered, tuple(n.name for n in ordered)

    def _check(self, query, evidence):
        if query not in self.names or set(evidence) - set(self.names):
            raise ValueError("查询或证据中含未知节点")
        if any(v not in (0, 1) for v in evidence.values()):
            raise ValueError("二值证据只能为 0/1")

    def query(self, query: str, evidence: dict[str, int] | None = None, *, method: str = "ve") -> float:
        evidence = dict(evidence or {})
        self._check(query, evidence)
        if method == "enumeration":
            totals = [0.0, 0.0]
            for states in _bits(len(self.nodes)):
                assignment = dict(zip(self.names, states))
                if any(assignment[k] != v for k, v in evidence.items()):
                    continue
                prob = 1.0
                for node in self.nodes:
                    p = node.p_true[tuple(assignment[p] for p in node.parents)]
                    prob *= p if assignment[node.name] else 1 - p
                totals[assignment[query]] += prob
        elif method == "ve":
            factors = [n.factor().restrict(evidence) for n in self.nodes]
            hidden = set(self.names) - {query} - set(evidence)
            while hidden:
                # 最小合并作用域启发式，只改变消元顺序，不改变概率语义。
                v = min(hidden, key=lambda h: (len({x for f in factors if h in f.variables for x in f.variables}), h))
                involved = [f for f in factors if v in f.variables]
                rest = [f for f in factors if v not in f.variables]
                product = Factor((), {(): 1.0})
                for f in involved:
                    product = product.multiply(f)
                factors = rest + [product.sum_out(v)]
                hidden.remove(v)
            product = Factor((), {(): 1.0})
            for f in factors:
                product = product.multiply(f)
            if query in evidence:
                total = sum(product.values.values())
                totals = [total if evidence[query] == 0 else 0.0, total if evidence[query] == 1 else 0.0]
            else:
                totals = [product.values.get((i,), 0.0) for i in (0, 1)]
        else:
            raise ValueError("method 只能为 ve 或 enumeration")
        total = sum(totals)
        if total <= 0:
            raise ImpossibleEvidence("该证据在模型下的概率为零")
        return totals[1] / total


class RiskBayes:
    """手工先验/CPT 基线；fit 用训练集做 Laplace 平滑估计，不读取验证/测试标签。"""
    DEFAULT_CPT = {"opacity": (.28, .65), "related_trade": (.22, .70),
                   "guarantee": (.24, .62), "cycle": (.12, .60)}

    def __init__(self, prior: float = .25, cpt: dict | None = None, threshold: float = .5):
        self.prior = float(prior)
        self.cpt = {k: tuple(v) for k, v in (self.DEFAULT_CPT if cpt is None else cpt).items()}
        self.threshold = threshold
        self.training_count = 0
        if not 0 < self.prior < 1 or not 0 < threshold < 1 or set(self.cpt) != set(FEATURES):
            raise ValueError("无效先验、阈值或 CPT 特征")
        if any(len(v) != 2 or any(not math.isfinite(p) or not 0 < p < 1 for p in v) for v in self.cpt.values()):
            raise ValueError("风险 CPT 需要两个严格位于 (0,1) 的概率")
        self.network = BinaryBayesNet([BinaryNode("Risk", (), {(): self.prior})] + [
            BinaryNode(f, ("Risk",), {(0,): self.cpt[f][0], (1,): self.cpt[f][1]}) for f in FEATURES])

    def evidence(self, observation: RiskObservation) -> dict[str, int]:
        return {f: int(v >= self.threshold) for f, v in observation.observed().items()}

    def predict(self, observation: RiskObservation, *, method: str = "analytic") -> float:
        evidence = self.evidence(observation)
        if method != "analytic":
            return self.network.query("Risk", evidence, method=method)
        log_odds = math.log(self.prior / (1 - self.prior))
        for f, state in evidence.items():
            p0, p1 = self.cpt[f]
            log_odds += math.log(p1 / p0) if state else math.log((1 - p1) / (1 - p0))
        if log_odds >= 0:
            return 1 / (1 + math.exp(-log_odds))
        ex = math.exp(log_odds)
        return ex / (1 + ex)

    @classmethod
    def fit(cls, observations: list[RiskObservation], labels: list[int], *, alpha: float = 1.0,
            threshold: float = .5) -> "RiskBayes":
        if not observations or len(observations) != len(labels) or alpha <= 0:
            raise ValueError("训练集须非空、长度一致且 alpha>0")
        if any(y not in (0, 1) for y in labels):
            raise ValueError("标签必须为 0/1")
        counts = [labels.count(y) for y in (0, 1)]
        cpt = {}
        for f in FEATURES:
            probabilities = []
            for y in (0, 1):
                values = [o.values[f] for o, target in zip(observations, labels)
                          if target == y and o.values.get(f) is not None]
                probabilities.append((sum(v >= threshold for v in values) + alpha) / (len(values) + 2 * alpha))
            cpt[f] = tuple(probabilities)
        model = cls((counts[1] + alpha) / (len(labels) + 2 * alpha), cpt, threshold)
        model.training_count = len(labels)
        return model

    def as_dict(self) -> dict:
        return {"kind": "binary_star_bayes", "prior": self.prior, "cpt": self.cpt,
                "threshold": self.threshold, "training_count": self.training_count,
                "assumption": "features conditionally independent given Risk"}

    @classmethod
    def from_dict(cls, data: dict) -> "RiskBayes":
        if data.get("kind") != "binary_star_bayes":
            raise ValueError("模型文件类型错误")
        model = cls(data["prior"], data["cpt"], data["threshold"])
        model.training_count = int(data.get("training_count", 0))
        return model
