"""二元 D-S 证据融合：普通 Dempster、可靠性折扣及冲突转无知集合。

识别框架 Θ={review, normal}。来源可靠性必须由来源质量给出，不能用测试标签倒推。
Dempster 组合要求合理的来源独立性；同一份指标算出的 Bayes/Fuzzy 不应直接当独立证据。
"""
from __future__ import annotations
import math
from dataclasses import dataclass


class TotalConflict(ValueError):
    """K=1 时标准 Dempster 组合未定义。"""


@dataclass(frozen=True)
class Mass:
    review: float
    normal: float
    ignorance: float

    def __post_init__(self):
        values = (self.review, self.normal, self.ignorance)
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in values) or abs(sum(values) - 1) > 1e-9:
            raise ValueError("BPA 各项须非负且和为 1")

    @classmethod
    def from_score(cls, score: float, reliability: float = 1.0) -> "Mass":
        if not math.isfinite(score) or not 0 <= score <= 1 or not math.isfinite(reliability) or not 0 <= reliability <= 1:
            raise ValueError("评分和可靠性须在 [0,1]")
        return cls(reliability * score, reliability * (1 - score), 1 - reliability)

    def discount(self, reliability: float) -> "Mass":
        if not math.isfinite(reliability) or not 0 <= reliability <= 1:
            raise ValueError("可靠性须在 [0,1]")
        return Mass(self.review * reliability, self.normal * reliability,
                    1 - reliability + reliability * self.ignorance)

    @property
    def pignistic(self) -> float:
        return self.review + .5 * self.ignorance

    @property
    def interval(self) -> tuple[float, float]:
        return self.review, self.review + self.ignorance

    def as_dict(self):
        return {"review": self.review, "normal": self.normal, "ignorance": self.ignorance}


@dataclass(frozen=True)
class FusionResult:
    mass: Mass
    conflict: float
    method: str


def _conjunctive(a: Mass, b: Mass):
    """二元识别框架：相反单点集合的乘积归入冲突，其余质量分配给交集。"""
    k = a.review * b.normal + a.normal * b.review
    pos = a.review * b.review + a.review * b.ignorance + a.ignorance * b.review
    neg = a.normal * b.normal + a.normal * b.ignorance + a.ignorance * b.normal
    return pos, neg, a.ignorance * b.ignorance, k


def dempster(a: Mass, b: Mass) -> FusionResult:
    pos, neg, unknown, k = _conjunctive(a, b)
    if 1 - k <= 1e-12:
        raise TotalConflict("K=1（或数值上近乎1），标准 Dempster 组合未定义")
    z = pos + neg + unknown
    return FusionResult(Mass(pos / z, neg / z, unknown / z), k, "dempster")


def yager(a: Mass, b: Mass) -> FusionResult:
    """把冲突质量保留为无知，避免高冲突时归一化放大微小证据。"""
    pos, neg, unknown, k = _conjunctive(a, b)
    z = pos + neg + unknown + k
    return FusionResult(Mass(pos / z, neg / z, (unknown + k) / z), k, "yager")


def fuse(sources: list[Mass], *, reliabilities: list[float] | None = None,
         robust: bool = True, conflict_threshold: float = .6) -> FusionResult:
    """固定来源顺序逐对融合；Yager 不保证结合律，顺序必须记录在实验配置中。"""
    if not 0 <= conflict_threshold <= 1:
        raise ValueError("冲突阈值须在 [0,1]")
    if reliabilities is not None and len(reliabilities) != len(sources):
        raise ValueError("可靠性数量与来源不一致")
    if reliabilities is not None:
        sources = [m.discount(r) for m, r in zip(sources, reliabilities)]
    current = Mass(0, 0, 1)
    maximum_conflict = 0.0
    methods = []
    for source in sources:
        k = _conjunctive(current, source)[3]
        result = yager(current, source) if robust and k >= conflict_threshold else dempster(current, source)
        maximum_conflict = max(maximum_conflict, result.conflict)
        methods.append(result.method)
        current = result.mass
    return FusionResult(current, maximum_conflict, "+".join(methods) or "vacuous")
