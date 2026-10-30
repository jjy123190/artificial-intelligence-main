"""Mamdani 模糊推理：三角/肩形隶属度、min 合取、max 聚合、质心解模糊。

输出是 0–1 风险评分，不是经过频率校准的概率。缺失指标不按零处理，评分向
中性 0.5 收缩，并报告 coverage。隶属度参数可在实验中独立扰动。
"""
from __future__ import annotations
import math
from core.kb.features import FEATURES, RiskObservation


def triangular(x: float, a: float, b: float, c: float) -> float:
    if not all(math.isfinite(v) for v in (x, a, b, c)) or not a <= b <= c or a == c:
        raise ValueError("三角隶属度需要有限的 a≤b≤c 且 a<c")
    if x == b:
        return 1.0
    if x <= a or x >= c:
        return 0.0
    return (x - a) / (b - a) if x < b else (c - x) / (c - b)


class MamdaniFuzzy:
    def __init__(self, *, high_start: float = .45, grid_points: int = 101, medium_weight: float = 1.0):
        if (isinstance(high_start, bool) or not isinstance(high_start, (int, float))
                or not math.isfinite(high_start) or not .05 <= high_start <= .9
                or isinstance(grid_points, bool) or not isinstance(grid_points, int) or grid_points < 11
                or isinstance(medium_weight, bool) or not isinstance(medium_weight, (int, float))
                or not math.isfinite(medium_weight) or not 0 < medium_weight <= 1):
            raise ValueError("high_start须在[.05,.9]，grid_points为至少11的整数，medium_weight须在(0,1]")
        self.high_start, self.grid_points = high_start, grid_points
        self.medium_weight = medium_weight
        self.training_count = 0
        self.training_brier = None
        self.fit_history = []
        self.grid = [i / (grid_points - 1) for i in range(grid_points)]
        self.outputs = {
            "low": [triangular(x, 0, 0, .45) for x in self.grid],
            "medium": [triangular(x, .2, .5, .8) for x in self.grid],
            "high": [triangular(x, .55, 1, 1) for x in self.grid],
        }
        self.weights = {"opacity": .65, "related_trade": .90, "guarantee": .85, "cycle": 1.0}

    def memberships(self, value: float) -> dict:
        if not 0 <= value <= 1:
            raise ValueError("输入指标须在 [0,1]")
        return {"low": triangular(value, 0, 0, 1 - self.high_start),
                "medium": triangular(value, 0, .5, 1),
                "high": triangular(value, self.high_start, 1, 1)}

    def explain(self, observation: RiskObservation) -> dict:
        inputs = {f: self.memberships(v) for f, v in observation.observed().items()}
        if not inputs:
            return {"score": .5, "coverage": 0.0, "memberships": {}, "activations": {},
                    "note": "全部缺测，返回中性评分，不认定低风险"}
        high = max(self.weights[f] * m["high"] for f, m in inputs.items())
        medium = self.medium_weight * max(m["medium"] for m in inputs.values())
        low = min(m["low"] for m in inputs.values())
        highs = sorted((m["high"] for m in inputs.values()), reverse=True)
        # 两个高隶属度线索相互支持；中等线索只激活中等输出，避免无依据升级。
        high = max(high, min(highs[:2]) if len(highs) >= 2 else 0.0)
        activations = {"low": low, "medium": medium, "high": high}
        aggregate = [max(min(activations[level], self.outputs[level][i]) for level in activations)
                     for i in range(self.grid_points)]
        # 梯形积分；端点各半权，避免把肩形集合的端点重复计入。
        weights = [.5 if i in (0, self.grid_points - 1) else 1.0 for i in range(self.grid_points)]
        area = sum(a * w for a, w in zip(aggregate, weights))
        crisp = sum(x * a * w for x, a, w in zip(self.grid, aggregate, weights)) / area if area else .5
        score = observation.coverage * crisp + (1 - observation.coverage) * .5
        return {"score": score, "coverage": observation.coverage,
                "memberships": inputs, "activations": activations,
                "note": "质心风险评分，不是校准概率"}

    def predict(self, observation: RiskObservation) -> float:
        return self.explain(observation)["score"]

    def as_dict(self) -> dict:
        """完整保存预测所需参数及训练审计记录，不需要原训练数据即可重载。"""
        return {"high_start": self.high_start, "grid_points": self.grid_points,
                "medium_weight": self.medium_weight, "training_count": self.training_count,
                "training_brier": self.training_brier, "fit_history": self.fit_history}

    @classmethod
    def from_dict(cls, data: dict) -> "MamdaniFuzzy":
        model = cls(high_start=data["high_start"], grid_points=data.get("grid_points", 101),
                    medium_weight=data.get("medium_weight", 1.0))
        model.training_count = data.get("training_count", 0)
        model.training_brier = data.get("training_brier")
        model.fit_history = list(data.get("fit_history", []))
        return model

    @classmethod
    def fit(cls, observations: list[RiskObservation], labels: list[int]) -> "MamdaniFuzzy":
        """训练集监督网格拟合。候选集预先固定，包含原手工基线；测试数据不参与。

        拟合共享的高隶属度起点及中等规则激活系数，仍为min/max/质心Mamdani。
        训练损失最小不保证测试改善；相同损失优先选择更接近手工参数的模型。
        """
        if (not observations or len(observations) != len(labels)
                or any(type(y) is not int or y not in (0, 1) for y in labels)):
            raise ValueError("训练集须非空，观测与0/1整数标签等长")
        history = []
        for start in (.15, .30, .45, .60, .75):
            for medium in (.35, .65, 1.0):
                model = cls(high_start=start, medium_weight=medium)
                loss = math.fsum((model.predict(o) - y) ** 2 for o, y in zip(observations, labels)) / len(labels)
                history.append({"high_start": start, "medium_weight": medium, "training_brier": loss})
        selected = min(history, key=lambda r: (r["training_brier"],
                       abs(r["high_start"] - .45) + abs(r["medium_weight"] - 1)))
        model = cls(high_start=selected["high_start"], medium_weight=selected["medium_weight"])
        model.training_count = len(labels)
        model.training_brier = selected["training_brier"]
        model.fit_history = history
        return model
