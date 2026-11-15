"""统一的可审计复核策略。数值评分不被锁定规则改写，最终行动单独记录。"""
from __future__ import annotations
import math

LEVELS = {"低": 0, "中": 1, "高": 2}


def review_decision(score: float, company_id: str, *, established_facts=(),
                    coverage: float = 1.0, lock_structure: bool = True,
                    threshold: float = .5) -> dict:
    """所有对照使用相同阈值；锁定只提高行动优先级，不伪造概率。

    constraint_violation 只衡量与已知输入规则的相容性，不是实际风险预测错误率。
    若输入事实本身错误，强制保留其预警仍可能增加误报，应由实验单独检验。
    """
    for name, value in [("score", score), ("coverage", coverage), ("threshold", threshold)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} 必须为有限数值")
        if not 0 <= value <= 1:
            raise ValueError(f"{name} 必须在 [0,1]")
    if not 0 < threshold < 1:
        raise ValueError("threshold 必须在 (0,1)")
    if not isinstance(lock_structure, bool):
        raise ValueError("lock_structure 必须是布尔值")
    facts = []
    alerts = []
    floor = 0
    for raw in established_facts:
        if not isinstance(raw, (list, tuple)) or not raw or not isinstance(raw[0], str):
            raise ValueError("结构事实格式错误")
        fact = tuple(raw)
        facts.append(fact)
        if fact[0] in ("Alert", "RiskLevel"):
            if len(fact) != 3:
                raise ValueError(f"结构事实元数错误: {fact}")
            if fact[1] != company_id:
                continue
            if fact[0] == "Alert":
                alerts.append(fact[2])
                floor = 2
            else:
                if fact[2] not in LEVELS:
                    raise ValueError(f"未知风险等级: {fact[2]}")
                floor = max(floor, LEVELS[fact[2]])
    risk_class = int(score >= threshold)
    observed_required = floor == 2
    review_required = bool(risk_class or (lock_structure and observed_required))
    active_floor = floor if lock_structure else 0
    if review_required:
        priority = "高"
    elif active_floor >= 1 or score >= min(.35, threshold) or coverage < 1:
        priority = "中"
    else:
        priority = "低"
    return {"score": float(score), "risk_class": risk_class,
            "review_required": review_required, "review_priority": priority,
            "decision_threshold": threshold, "lock_structure": lock_structure,
            "structural_alert_requires_review": observed_required,
            "rule_priority_floor": ("低", "中", "高")[floor],
            "applied_priority_floor": ("低", "中", "高")[active_floor],
            "lock_changed_action": bool(lock_structure and observed_required and not risk_class),
            "constraint_violation": bool(observed_required and not review_required),
            "priority_floor_violation": LEVELS[priority] < floor,
            "provided_facts": facts, "locked_facts": facts if lock_structure else [],
            "alerts": sorted(set(alerts)), "coverage": coverage,
            "note": "score始终是原始评分；review_required才是包含结构约束的最终复核行动。"}
