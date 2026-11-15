"""同一股权案例到四个风险指标的可审计映射。

opacity 为明示的教学设定；related_trade/guarantee 的分母是注册资本，不是营收。
cycle 来自逻辑可达闭包。缺测用 None，不能当作零风险。
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Mapping

FEATURES = ("opacity", "related_trade", "guarantee", "cycle")
FEATURE_CN = {"opacity": "股权不透明度代理", "related_trade": "关联交易额/注册资本",
              "guarantee": "对外担保额/注册资本", "cycle": "持股或担保回路线索"}


@dataclass(frozen=True)
class RiskObservation:
    company_id: str
    values: Mapping[str, float | None]
    source: str = ""
    quality: float = 1.0

    def __post_init__(self):
        if not self.company_id:
            raise ValueError("需要样本/企业 ID")
        if set(self.values) - set(FEATURES):
            raise ValueError(f"未知特征 {set(self.values) - set(FEATURES)}")
        for key, value in self.values.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))
                                      or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{key} 必须为 [0,1] 内的数值或 None")
        if not math.isfinite(self.quality) or not 0 <= self.quality <= 1:
            raise ValueError("quality 必须在 [0,1]")

    @property
    def coverage(self) -> float:
        return sum(self.values.get(f) is not None for f in FEATURES) / len(FEATURES)

    def observed(self) -> dict[str, float]:
        return {f: float(self.values[f]) for f in FEATURES if self.values.get(f) is not None}

    def as_dict(self) -> dict:
        return {"company_id": self.company_id, "values": {f: self.values.get(f) for f in FEATURES},
                "source": self.source, "quality": self.quality}


def from_case(case, forward_engine) -> list[RiskObservation]:
    """把图结构映射为0到1指标；交易额除以注册资本，截断只用于评分。"""
    related = {(a, b) for _, a, b in forward_engine.query("RelatedTrade", "?a", "?b")}
    cycles = {c for _, c, tag in forward_engine.query("Alert", "?c", "?t")
              if tag in ("循环持股", "担保圈")}
    out = []
    for e in case.companies:
        amount = sum(t.amount for t in case.trades
                     if (t.seller, t.buyer) in related and e.eid in (t.seller, t.buyer))
        out.append(RiskObservation(e.eid, {
            "opacity": e.opacity,
            "related_trade": min(amount / e.capital, 1.0),
            "guarantee": case.guarantee_load(e.eid),
            "cycle": float(e.eid in cycles),
        }, source="同源 CSV + 规则派生 + 明示的 opacity 教学参数", quality=e.disclosure))
    return out
