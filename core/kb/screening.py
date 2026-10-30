"""指标层的确定性风险筛查基线，复用真正的规则引擎。

与 27 条结构规则分层：前者推控制/可达事实，本层将同一组 0–1 指标映射为
NeedsReview，用于与概率/模糊模型在同一二分类目标下作量化对照。
阈值为教学设定，规则触发不等于合成标签或真实违法认定。
"""
from core.kb.features import FEATURES, FEATURE_CN, RiskObservation
from core.kb.rules_dsl import Pattern, Rule, RuleSet, gt
from experiments.lab02.deterministic.forward import ForwardEngine


class ScreeningRules:
    DEFAULTS = {"opacity": .70, "related_trade": .60, "guarantee": .65, "cycle": .50}

    def __init__(self, thresholds: dict | None = None):
        self.thresholds = dict(self.DEFAULTS if thresholds is None else thresholds)
        if set(self.thresholds) != set(FEATURES) or any(not 0 <= v <= 1 for v in self.thresholds.values()):
            raise ValueError("筛查阈值须覆盖四个特征且位于 [0,1]")
        rules = [Rule(f"S{i}", f"{FEATURE_CN[f]}超过阈值 {self.thresholds[f]}",
                 (Pattern.of("Metric", "?c", f, "?v"),), Pattern.of("ScreenSignal", "?c", f),
                 gt("?v", self.thresholds[f])) for i, f in enumerate(FEATURES, 1)]
        rules.append(Rule("S5", "有任一指标线索则建议人工复核",
                     (Pattern.of("ScreenSignal", "?c", "?f"),), Pattern.of("NeedsReview", "?c")))
        self.rules = RuleSet(rules)

    def explain(self, observation: RiskObservation) -> dict:
        facts = [("Metric", observation.company_id, f, v) for f, v in observation.observed().items()]
        engine = ForwardEngine(self.rules, facts)
        engine.run()
        signals = [f[2] for f in engine.query("ScreenSignal", observation.company_id, "?f")]
        needs_review = bool(signals)
        return {"needs_review": needs_review, "signals": signals, "coverage": observation.coverage,
                "score": float(needs_review) if observation.coverage else .5,
                "trace": [str(s) for s in engine.steps]}

    def predict(self, observation: RiskObservation) -> float:
        return self.explain(observation)["score"]
