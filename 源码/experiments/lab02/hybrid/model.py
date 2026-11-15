"""结构事实锁定 + 可校准的线性意见池。

同一份特征产生的规则/Bayes/Fuzzy 输出彼此相关，因此不把三者当成独立 D-S
证据相乘。概率和模糊评分在线性池中加权，权重只在验证集上按 Brier 选择。
结构规则推导的事实保留不动；概率评分只能调整复核排序，不能推翻控制关系。
"""
from __future__ import annotations
import copy
import math
from core.kb.features import RiskObservation
from core.kb.screening import ScreeningRules
from experiments.lab02.uncertain.bayes import RiskBayes
from experiments.lab02.uncertain.fuzzy import MamdaniFuzzy
from .decision import review_decision
from .calibration import ReviewPolicy


class HybridModel:
    NAMES = ("bayes", "fuzzy", "rule")
    DEFAULT_WEIGHTS = (.60, .25, .15)

    def __init__(self, bayes: RiskBayes | None = None, fuzzy: MamdaniFuzzy | None = None,
                 rules: ScreeningRules | None = None, weights=DEFAULT_WEIGHTS):
        if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("混合权重需要三个非负有限值且和为正数")
        total = sum(weights)
        self.weights = tuple(w / total for w in weights)
        self.bayes = RiskBayes() if bayes is None else bayes
        self.fuzzy = MamdaniFuzzy() if fuzzy is None else fuzzy
        self.rules = ScreeningRules() if rules is None else rules
        self.review_policy = ReviewPolicy()
        self.validation_count = 0
        self.validation_brier = None
        self.ablation = None

    def components(self, observation: RiskObservation) -> dict:
        screening = self.rules.explain(observation)
        # 0.85/0.15 是规则意见评分，不声称是规则命中的经验概率。
        rule_score = (.85 if screening["needs_review"] else .15) if observation.coverage else .5
        return {"bayes": self.bayes.predict(observation), "fuzzy": self.fuzzy.predict(observation),
                "rule": rule_score}

    def pool(self, scores: dict) -> float:
        return math.fsum(w * scores[name] for name, w in zip(self.NAMES, self.weights) if w)

    def predict(self, observation: RiskObservation) -> float:
        # 评分时跳过零权重组件；explain 仍可按需计算全部诊断信息。
        bayes_w, fuzzy_w, rule_w = self.weights
        parts = [bayes_w * self.bayes.predict(observation)] if bayes_w else []
        if fuzzy_w:
            parts.append(fuzzy_w * self.fuzzy.predict(observation))
        if rule_w:
            detail = self.rules.explain(observation)
            rule_score = (.85 if detail["needs_review"] else .15) if observation.coverage else .5
            parts.append(rule_w * rule_score)
        return math.fsum(parts)

    def explain(self, observation: RiskObservation, *, established_facts=(),
                lock_structure: bool = True, threshold: float = .5) -> dict:
        scores = self.components(observation)
        decision = review_decision(self.pool(scores), observation.company_id,
            established_facts=established_facts, coverage=observation.coverage,
            lock_structure=lock_structure, threshold=threshold)
        decision.update({"score_kind": "线性融合风险评分（非法律结论）", "components": scores,
                         "weights": dict(zip(self.NAMES, self.weights))})
        if self.ablation is not None:
            decision["ablation"] = copy.deepcopy(self.ablation)
        return decision

    def fit_weights(self, observations: list[RiskObservation], labels: list[int]):
        if not observations or len(observations) != len(labels) or any(y not in (0, 1) for y in labels):
            raise ValueError("验证集须非空且为等长二分类标签")
        components = [self.components(o) for o in observations]
        candidates = [(a / 4, b / 4, (4 - a - b) / 4) for a in range(5) for b in range(5 - a)]
        def loss(weights):
            return sum((math.fsum(w * s[n] for w, n in zip(weights, self.NAMES) if w) - y) ** 2
                       for s, y in zip(components, labels)) / len(labels)
        # 并列时更偏向较简单的 Bayes 基线，不强迫实验得出“融合更好”。
        self.weights = min(candidates, key=lambda w: (loss(w), -w[0], -w[1]))
        self.validation_count = len(labels)
        self.validation_brier = loss(self.weights)
        self.ablation = None
        return self

    def without(self, component: str, *, fallback: str | None = None) -> "HybridModel":
        """严格删除并归一化；全零时只有显式指定 fallback 才构造替代基线。"""
        if component not in self.NAMES:
            raise ValueError("未知消融组件")
        if fallback is not None and (fallback not in self.NAMES or fallback == component):
            raise ValueError("回退组件必须是未被移除的已知组件")
        weights = tuple(0.0 if n == component else w for n, w in zip(self.NAMES, self.weights))
        kind = "renormalized" if self.weights[self.NAMES.index(component)] else "unchanged_zero_weight"
        used_fallback = None
        if not sum(weights):
            if fallback is None:
                raise ValueError("移除后没有剩余正权重组件；基线对照须显式指定 fallback")
            weights = tuple(float(n == fallback) for n in self.NAMES)
            kind, used_fallback = "baseline_fallback", fallback
        model = HybridModel(self.bayes, self.fuzzy, self.rules, weights)
        model.review_policy = ReviewPolicy.from_dict(self.review_policy.as_dict())
        model.ablation = {"removed_component": component, "kind": kind,
                          "base_weights": dict(zip(self.NAMES, self.weights)),
                          "result_weights": dict(zip(self.NAMES, model.weights)),
                          "fallback_if_empty": fallback, "fallback_component": used_fallback}
        return model

    def as_dict(self) -> dict:
        return {"weights": dict(zip(self.NAMES, self.weights)), "validation_count": self.validation_count,
                "validation_brier": self.validation_brier,
                "fuzzy_high_start": self.fuzzy.high_start,
                "fuzzy": self.fuzzy.as_dict(),
                "review_policy": self.review_policy.as_dict(),
                "ablation": copy.deepcopy(self.ablation),
                "screening_thresholds": self.rules.thresholds,
                "bayes": self.bayes.as_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "HybridModel":
        fuzzy = (MamdaniFuzzy.from_dict(data["fuzzy"]) if "fuzzy" in data
                 else MamdaniFuzzy(high_start=data["fuzzy_high_start"]))
        model = cls(RiskBayes.from_dict(data["bayes"]), fuzzy,
                    ScreeningRules(data["screening_thresholds"]), [data["weights"][n] for n in cls.NAMES])
        if "review_policy" in data:
            model.review_policy = ReviewPolicy.from_dict(data["review_policy"])
        model.validation_count = data.get("validation_count", 0)
        model.validation_brier = data.get("validation_brier")
        model.ablation = copy.deepcopy(data.get("ablation"))
        return model
