"""股权图→27条结构规则→特征→风险评分→复核行动的共用生产入口。

CLI、演示及端到端实验共用本模块；所有方法使用同一张观测图和同一事实闭包。
训练标签/干净图真值不进入本接口。不完整资料中“未证明”不能自动当“无风险”。
"""
from __future__ import annotations
import copy
import time
from dataclasses import asdict, dataclass
from core.kb.domain import EquityCase
from core.kb.facts import FactBase, build_facts
from core.kb.features import RiskObservation, from_case
from core.kb.rules import build_rules
from experiments.lab02.deterministic.forward import ForwardEngine
from .decision import review_decision
from .model import HybridModel

SOURCE_FIELDS = ("holdings", "trades", "guarantees", "registry")
METHODS = ("rules_only", "bayes_only", "fuzzy_only", "hybrid_unlocked", "hybrid_locked", "hybrid_calibrated")


@dataclass
class ReasonedCase:
    case: EquityCase
    facts: FactBase
    engine: ForwardEngine
    observations: dict[str, RiskObservation]
    completeness: dict[str, bool]
    preparation_ms: float

    def established_for(self, company_id: str) -> list[tuple]:
        if company_id not in self.observations:
            raise ValueError(f"不是案例中的企业: {company_id}")
        return sorted((f for f in self.engine.wm.facts
            if (f[0] in ("Alert", "RiskLevel", "ActualController", "BeneficialOwner") and f[1] == company_id)
            or (f[0] == "Controls" and f[2] == company_id)), key=str)

    def as_dict(self, *, include_trace: bool = True) -> dict:
        out = {"case_label": self.case.label, "source": self.case.source,
               "size": self.case.summary(), "completeness": self.completeness,
               "preparation_ms": self.preparation_ms, "forward": self.engine.stats(),
               "observations": [o.as_dict() for o in self.observations.values()]}
        if include_trace:
            out["input_facts"] = [asdict(f) for f in self.facts]
            out["derived_facts"] = [{"fact": f, "rule": r, "premises": p}
                                    for f, (r, p) in self.engine.why.items()]
            out["steps"] = [asdict(s) for s in self.engine.steps]
        return out


def prepare_case(case: EquityCase, *, completeness: dict[str, bool] | None = None) -> ReasonedCase:
    start = time.perf_counter()
    flags = {f: True for f in SOURCE_FIELDS}
    if completeness is not None:
        if not isinstance(completeness, dict) or set(completeness) - set(flags):
            raise ValueError("未知的资料完整性字段")
        if any(not isinstance(v, bool) for v in completeness.values()):
            raise ValueError("资料完整性必须是布尔值")
        flags.update(completeness)
    snapshot = copy.deepcopy(case)
    facts = build_facts(snapshot)
    if not flags["trades"] or not flags["registry"]:
        # 资料不完整时，未见交易不能作为“没有经营记录”的壳结构证据。
        facts.facts = [f for f in facts.facts if f.pred != "Shell"]
    engine = ForwardEngine(build_rules(), facts.to_tuples())
    engine.run()
    observations = {}
    for original in from_case(snapshot, engine):
        values = dict(original.values)
        if not flags["registry"]:
            values["opacity"] = None
        if not flags["holdings"] or not flags["trades"]:
            # 不把不完整交易总额/未证共同控制当作精确的0比例。
            values["related_trade"] = None
        if not flags["guarantees"]:
            values["guarantee"] = None
        if (not flags["holdings"] or not flags["guarantees"]) and not values["cycle"]:
            values["cycle"] = None
        observations[original.company_id] = RiskObservation(original.company_id, values,
            source=original.source + "; 未完整采集的指标用None表示，已有正向证明仍保留", quality=original.quality)
    return ReasonedCase(snapshot, facts, engine, observations, flags,
                        (time.perf_counter() - start) * 1000)


def assess_company(context: ReasonedCase, company_id: str, model: HybridModel | None = None,
                   *, method: str = "hybrid_locked", threshold: float = .5,
                   include_explanation: bool = False) -> dict:
    start = time.perf_counter()
    if method not in METHODS:
        raise ValueError(f"未知方法: {method}")
    if company_id not in context.observations:
        raise ValueError(f"未知企业: {company_id}")
    model = HybridModel() if model is None else model
    observation = context.observations[company_id]
    established = context.established_for(company_id)
    components = None
    if method == "rules_only":
        score = float(any(f[0] == "Alert" or (f[0] == "RiskLevel" and f[2] == "高")
                          for f in established))
    elif method == "bayes_only":
        score = model.bayes.predict(observation)
    elif method == "fuzzy_only":
        score = model.fuzzy.predict(observation)
    else:
        score = model.predict(observation)
    if method == "hybrid_calibrated":
        decision = model.review_policy.decide(score, company_id, established_facts=established,
                                               coverage=observation.coverage, threshold=threshold)
    else:
        decision = review_decision(score, company_id, established_facts=established,
                                  coverage=observation.coverage, lock_structure=method in ("hybrid_locked", "rules_only"),
                                  threshold=threshold)
    decision_ms = (time.perf_counter() - start) * 1000
    active_weights = (dict(zip(model.NAMES, model.weights)) if method.startswith("hybrid_")
                      else {"structural_rules" if method == "rules_only" else method.removesuffix("_only"): 1.0})
    decision.update({"company_id": company_id, "company_name": context.case.name(company_id),
                     "method": method, "observation": observation.as_dict(),
                     "preparation_ms": context.preparation_ms, "decision_ms": decision_ms,
                     "end_to_end_ms": context.preparation_ms + decision_ms,
                     "weights": active_weights})
    if include_explanation:
        components = (model.components(observation) if method.startswith("hybrid_")
                      else {next(iter(active_weights)): score})
        decision["components"] = components
        decision["proofs"] = [{"fact": f, "lines": context.engine.explain(f)} for f in established]
    return decision
