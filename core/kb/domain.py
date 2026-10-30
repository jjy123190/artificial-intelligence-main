"""企业股权教学案例。第一次 CSV 为唯一基础数据，扩展数据不改写基础关系。

金额单位均为万元。opacity/disclosure 是教学设定，不是对真实企业的测量。
全部数据虚构，任何风险输出均为人工复核线索，不能用于法律认定。
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Entity:
    eid: str
    name: str
    kind: str
    category: str = ""
    capital: float = 0.0
    opacity: float = 0.0
    disclosure: float = 1.0

    @property
    def is_person(self) -> bool:
        return self.kind == "Person"

    @property
    def is_listed(self) -> bool:
        return self.category == "上市公司"


@dataclass(frozen=True)
class Holding:
    holder: str
    target: str
    ratio: float
    since: str = ""
    layered: bool = False


@dataclass(frozen=True)
class Guarantee:
    guarantor: str
    beneficiary: str
    amount: float


@dataclass(frozen=True)
class Trade:
    seller: str
    buyer: str
    amount: float


@dataclass(frozen=True)
class Kinship:
    a: str
    b: str
    relation: str
    evidence: bool = False


@dataclass
class EquityCase:
    entities: list[Entity] = field(default_factory=list)
    holdings: list[Holding] = field(default_factory=list)
    guarantees: list[Guarantee] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    kinships: list[Kinship] = field(default_factory=list)
    source: str = ""
    label: str = ""

    def entity(self, eid: str) -> Entity:
        for e in self.entities:
            if e.eid == eid:
                return e
        raise KeyError(f"未知实体 {eid}")

    def name(self, eid: str) -> str:
        return self.by_id().get(eid, Entity(eid, eid, "Person")).name

    def by_id(self) -> dict[str, Entity]:
        return {e.eid: e for e in self.entities}

    @property
    def persons(self) -> list[Entity]:
        return [e for e in self.entities if e.is_person]

    @property
    def companies(self) -> list[Entity]:
        return [e for e in self.entities if not e.is_person]

    def capital_of(self, eid: str) -> float:
        return self.entity(eid).capital

    def guarantee_load(self, cid: str) -> float:
        total = sum(g.amount for g in self.guarantees if g.guarantor == cid)
        return min(total / self.capital_of(cid), 1.0)

    def trade_share(self, cid: str) -> float:
        """总交易额/注册资本的截断代理，不是关联交易占比或营业收入占比。"""
        total = sum(t.amount for t in self.trades if cid in (t.seller, t.buyer))
        return min(total / self.capital_of(cid), 1.0)

    def in_degree(self, cid: str) -> int:
        return sum(h.target == cid for h in self.holdings)

    def out_degree(self, cid: str) -> int:
        return sum(h.holder == cid for h in self.holdings)

    def summary(self) -> dict:
        return {"实体": len(self.entities), "自然人": len(self.persons),
                "企业": len(self.companies), "持股关系": len(self.holdings),
                "担保关系": len(self.guarantees), "交易关系": len(self.trades),
                "亲属关系": len(self.kinships),
                "上市公司": sum(e.is_listed for e in self.companies)}

    def validate(self) -> "EquityCase":
        """拒绝缺失引用、重复持股、负金额、NaN 和同一公司总持股超过 100%。"""
        ids = self.by_id()
        if len(ids) != len(self.entities):
            raise ValueError("实体 ID 重复")
        for e in self.entities:
            if not e.eid or e.eid.startswith("?") or e.kind not in ("Person", "Company"):
                raise ValueError(f"无效实体: {e}")
            for v in (e.opacity, e.disclosure):
                if not math.isfinite(v) or not 0 <= v <= 1:
                    raise ValueError(f"不确定性属性必须在 [0,1]: {e.eid}")
            if not e.is_person and (not math.isfinite(e.capital) or e.capital <= 0):
                raise ValueError(f"注册资本须为正数: {e.eid}")
        totals: dict[str, float] = {}
        pairs = set()
        for h in self.holdings:
            if h.holder not in ids or h.target not in ids or ids[h.target].is_person:
                raise ValueError(f"持股引用无效: {h}")
            if not math.isfinite(h.ratio) or not 0 < h.ratio <= 1:
                raise ValueError(f"持股比例须在 (0,1]: {h}")
            if (h.holder, h.target) in pairs:
                raise ValueError(f"持股记录重复: {h}")
            pairs.add((h.holder, h.target))
            totals[h.target] = totals.get(h.target, 0.0) + h.ratio
        for cid, ratio in totals.items():
            if ratio > 1.0 + 1e-9:
                raise ValueError(f"{cid} 已知股权合计 {ratio:.2%} 超过 100%")
        for rel in [*self.guarantees, *self.trades]:
            a, b = ((rel.guarantor, rel.beneficiary) if isinstance(rel, Guarantee)
                    else (rel.seller, rel.buyer))
            if any(x not in ids or ids[x].is_person for x in (a, b)):
                raise ValueError(f"企业关系引用无效: {rel}")
            if not math.isfinite(rel.amount) or rel.amount < 0:
                raise ValueError(f"金额须为非负有限数: {rel}")
        for k in self.kinships:
            if k.a == k.b or any(x not in ids or not ids[x].is_person for x in (k.a, k.b)):
                raise ValueError(f"亲属关系引用无效: {k}")
        return self


LAB_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DATA = LAB_ROOT / "data" / "raw"
# 第一次目录是只读存档，仅供可选的数据一致性检查，不是默认运行依赖。
ARCHIVE_DATA = LAB_ROOT / "docs" / "lab01" / "original-data-reference"
SIBLING_DATA = ARCHIVE_DATA  # 保留旧接口名，避免破坏既有校验代码。
CSV_NAMES = ("persons", "companies", "holdings", "guarantees", "trades", "kinships")


def load_from_equity_kg(data_dir: str | Path | None = None) -> EquityCase:
    """默认只读取统一项目内的 data/raw 快照，不依赖或改写第一次存档。

    名称保留历史兼容。指定 data_dir 后只读该目录；缺文件立即报错，不回退。
    """
    snapshot = data_dir is None
    d = SNAPSHOT_DATA if snapshot else Path(data_dir)
    raw = {}
    for name in CSV_NAMES:
        with (d / f"{name}.csv").open(encoding="utf-8-sig", newline="") as fh:
            raw[name] = list(csv.DictReader(fh))
    case = EquityCase(source=f"{'本项目内的第一次 CSV 快照' if snapshot else '用户指定 CSV'}: {d.resolve()}",
                      label="第一次汇报同源案例")
    for r in raw["persons"]:
        case.entities.append(Entity(r["id"], r["姓名"], "Person"))
    for r in raw["companies"]:
        cap = float(r["注册资本万元"])
        cat = r["类型"]
        opacity = 0.05 if cat == "上市公司" else 0.35 if cat == "担保公司" else 0.55 if cap <= 5000 else 0.15
        case.entities.append(Entity(r["id"], r["名称"], "Company", cat, cap,
                                    opacity, 1.0 - opacity))
    case.holdings = [Holding(r["股东"], r["被投企业"], float(r["持股比例"])) for r in raw["holdings"]]
    case.guarantees = [Guarantee(r["担保方"], r["被担保方"], float(r["金额万元"])) for r in raw["guarantees"]]
    case.trades = [Trade(r["卖方"], r["买方"], float(r["金额万元"])) for r in raw["trades"]]
    case.kinships = [Kinship(r["自然人1"], r["自然人2"], r["关系"], True) for r in raw["kinships"]]
    return case.validate()


def demo_case() -> EquityCase:
    """同源扩展：9 名自然人 + 20 家企业；保留第一次全部关系和金额。

    P5→C13→C14→C15 是控制链；C15 仅持 C3 的 8%，不能推出 P5 控制 C3。
    P8 是 C20 的登记股东；没有证据时不推断其代持或真实出资人。
    """
    case = load_from_equity_kg()
    for eid, name in [("P5", "孙德海"), ("P6", "周文斌"), ("P7", "周文杰"),
                      ("P8", "吴国强"), ("P9", "郑晓雯")]:
        case.entities.append(Entity(eid, name, "Person"))
    for eid, name, cat, cap, op in [
        ("C13", "北岸投资控股有限公司", "企业", 40000, .50),
        ("C14", "北岸产业基金合伙企业", "企业", 25000, .60),
        ("C15", "启元科技发展有限公司", "企业", 12000, .65),
        ("C16", "联信置业有限公司", "企业", 18000, .50),
        ("C17", "联信装饰工程有限公司", "企业", 4000, .60),
        ("C18", "宏泰融资租赁有限公司", "企业", 22000, .45),
        ("C19", "宏泰供应链管理有限公司", "企业", 5000, .60),
        ("C20", "名义持股平台一号", "企业", 100, .85),
    ]:
        case.entities.append(Entity(eid, name, "Company", cat, float(cap), op, 1 - op))
    for a, b, r in [
        ("P5", "C13", .62), ("C13", "C14", .58), ("C14", "C15", .55), ("C15", "C3", .08),
        ("C16", "C17", .55), ("C17", "C16", .35), ("P6", "C16", .33), ("P7", "C16", .22),
        ("P9", "C18", .55), ("C18", "C19", .60), ("C19", "C20", .70), ("P8", "C20", .30),
        ("C20", "C18", .15), ("C16", "C19", .20), ("C18", "C17", .10), ("P6", "C12", .08),
        ("P9", "C9", .15), ("C13", "C16", .10), ("C2", "C15", .10), ("C5", "C4", .10),
        ("C12", "C19", .05),
    ]:
        case.holdings.append(Holding(a, b, r, layered=a.startswith("C")))
    for a, b, m in [("C18", "C19", 8000), ("C19", "C20", 5000), ("C20", "C18", 3000),
                    ("C2", "C15", 4000), ("C3", "C1", 6000), ("C9", "C4", 1500),
                    ("C13", "C18", 2000), ("C6", "C7", 1800)]:
        case.guarantees.append(Guarantee(a, b, float(m)))
    for a, b, m in [("C15", "C3", 9000), ("C17", "C16", 3000), ("C3", "C15", 4500),
                    ("C12", "C19", 2500), ("C8", "C6", 1200), ("C11", "C10", 900),
                    ("C19", "C12", 3200)]:
        case.trades.append(Trade(a, b, float(m)))
    case.kinships.extend([Kinship("P6", "P7", "兄弟", True), Kinship("P3", "P8", "表亲", False),
                          Kinship("P4", "P9", "配偶", True), Kinship("P2", "P9", "姐妹", False)])
    case.source += "；追加明示的虚构扩展数据（未改变第一次记录）"
    case.label = "第二次汇报扩展案例"
    return case.validate()


def scaled_case(n_groups: int = 20, seed: int = 42) -> EquityCase:
    """固定种子的结构规模实验。结果仅反映此类控股链/担保圈分布。"""
    if n_groups < 1:
        raise ValueError("n_groups 必须为正整数")
    rng = random.Random(seed)
    case = EquityCase(source=f"合成股权网络 seed={seed}", label=f"规模实验 {n_groups} 组")
    for g in range(1, n_groups + 1):
        p = f"P{g}"
        case.entities.append(Entity(p, f"自然人{g}", "Person"))
        chain = []
        prev = p
        for lv in range(rng.randint(2, 4)):
            cid = f"C{g}_{lv}"
            opacity = rng.uniform(.05, .85)
            case.entities.append(Entity(cid, f"企业{g}-{lv}", "Company",
                rng.choice(["企业", "企业", "上市公司"]), float(rng.randint(1000, 80000)),
                opacity, 1 - opacity))
            # 根节点给下一组的交叉持股预留至少 35%，避免总和超过 100%。
            case.holdings.append(Holding(prev, cid, round(rng.uniform(.35, .65 if lv == 0 else .95), 2)))
            chain.append(cid)
            prev = cid
        if len(chain) >= 3:
            for a, b in zip(chain[:3], chain[1:3] + chain[:1]):
                case.guarantees.append(Guarantee(a, b, float(rng.randint(500, 9000))))
        if g > 1:
            case.holdings.append(Holding(chain[0], f"C{g-1}_0", round(rng.uniform(.05, .30), 2)))
        case.trades.append(Trade(chain[-1], chain[0], float(rng.randint(200, 8000))))
    return case.validate()


def describe(case: EquityCase) -> str:
    lines = [case.label, case.source, "，".join(f"{k}: {v}" for k, v in case.summary().items())]
    lines.extend(f"{e.eid:4s} {e.name} [{e.kind}]" for e in case.entities)
    return "\n".join(lines)
