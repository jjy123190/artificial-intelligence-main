"""可校准的复核行动门控，结构证明永不删除；并非宣称输入记录可靠。

对照策略区别：硬锁定所有已证预警；门控只保留得到最低评分支持的额外行动。
阈值仅按预设验证扰动上的代价选择，允许退化为不锁定，并如实记录漏报风险。
"""
from __future__ import annotations
import math
from .decision import review_decision


class ReviewPolicy:
    CANDIDATES = (0.0, .15, .25, .35, .45, .5)

    def __init__(self, minimum_support: float = .25, false_negative_cost: float = 2.0):
        if (isinstance(minimum_support, bool) or not isinstance(minimum_support, (int, float))
                or not math.isfinite(minimum_support) or not 0 <= minimum_support <= 1):
            raise ValueError("minimum_support须为[0,1]内有限数")
        if (isinstance(false_negative_cost, bool) or not isinstance(false_negative_cost, (int, float))
                or not math.isfinite(false_negative_cost) or false_negative_cost <= 0):
            raise ValueError("漏报代价须为有限正数")
        self.minimum_support = minimum_support
        self.false_negative_cost = false_negative_cost
        self.validation_count = 0
        self.selection_history = []

    def decide(self, score, company_id, *, established_facts=(), coverage=1.0, threshold=.5):
        facts = tuple(established_facts)
        decision = review_decision(score, company_id, established_facts=facts, coverage=coverage,
                                   threshold=threshold, lock_structure=False)
        observed = decision["structural_alert_requires_review"]
        supported = observed and score >= self.minimum_support
        if supported:
            decision = review_decision(score, company_id, established_facts=facts, coverage=coverage,
                                       threshold=threshold, lock_structure=True)
        # 证明与最终行动分离：被门控的预警仍在provided_facts/retained_proofs中，不能改称未证。
        decision.update({"review_policy": "validation_calibrated_support",
                         "minimum_support": self.minimum_support,
                         "retained_proofs": list(facts),
                         "deferred_structural_alert": bool(observed and not decision["review_required"]),
                         "policy_note": "门控允许已证线索不触发高复核；不删除证明，不保证零结构违反或零漏报"})
        return decision

    def fit(self, rows: list[dict], *, threshold=.5):
        if not rows or not 0 < threshold < 1:
            raise ValueError("验证数据须非空，threshold须在(0,1)")
        for row in rows:
            if (type(row.get("label")) is not int or row["label"] not in (0, 1)
                    or type(row.get("observed_required")) is not bool
                    or not isinstance(row.get("score"), (int, float))
                    or isinstance(row["score"], bool) or not math.isfinite(row["score"])
                    or not 0 <= row["score"] <= 1):
                raise ValueError("验证行须含score、布尔observed_required及0/1整数label")
        history = []
        for support in self.CANDIDATES:
            actions = [r["score"] >= threshold or (r["observed_required"] and r["score"] >= support) for r in rows]
            fp = sum(a and not r["label"] for a, r in zip(actions, rows))
            fn = sum(not a and r["label"] for a, r in zip(actions, rows))
            history.append({"minimum_support": support, "fp": fp, "fn": fn,
                            "cost": (fp + self.false_negative_cost * fn) / len(rows)})
        choice = min(history, key=lambda r: (r["cost"], r["minimum_support"]))
        self.minimum_support = choice["minimum_support"]
        self.validation_count = len(rows)
        self.selection_history = history
        return self

    def as_dict(self):
        return {"minimum_support": self.minimum_support, "false_negative_cost": self.false_negative_cost,
                "validation_count": self.validation_count, "selection_history": self.selection_history}

    @classmethod
    def from_dict(cls, data):
        model = cls(data["minimum_support"], data["false_negative_cost"])
        model.validation_count = data.get("validation_count", 0)
        model.selection_history = list(data.get("selection_history", []))
        return model
