"""事实库：把领域数据（EquityCase）转换为逻辑事实。

同一份事实支撑三种推理，因此这里提供两种视图：

    to_atoms()   → 一阶原子公式集合，供归结反演使用（kb/fol.py 的 Atom）
    to_tuples()  → 元组形式 (pred, 参数…)，供正向/反向演绎推理使用
                   （kb/rules_dsl.py 的 Pattern 直接匹配元组）

之所以不做"一个视图转另一个"，是因为归结需要 Num/Const 的严格类型区分，
而演绎推理只需要可比较的值；各留一份，代价是要保证两处转换规则一致，
因此二者共用同一个 _raw_facts() 中间表示（列表 + 来源标注），不重复写逻辑。

事实规模如实统计：每个谓词各有多少条、哪些是输入、哪些是派生的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from .domain import EquityCase
from .fol import Atom, Const, Num, Var, atom

# 空壳公司判定阈值（教学口径，见 rules.R18 的 note）
SHELL_CAPITAL_MAX = 500.0


@dataclass(frozen=True)
class RawFact:
    """一条事实：谓词 + 参数（Python 原生值）+ 来源说明。"""

    pred: str
    args: tuple
    source: str = "输入数据"

    def as_tuple(self) -> tuple:
        return (self.pred,) + self.args

    def as_atom(self) -> Atom:
        return atom(self.pred, *self.args)

    def __str__(self) -> str:
        if not self.args:
            return self.pred
        return f"{self.pred}({','.join(_fmt(a) for a in self.args)})"


def _fmt(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, float):
        return str(int(x)) if x.is_integer() else str(x)
    return str(x)


@dataclass
class FactBase:
    """事实集合，带按谓词索引与来源统计。"""

    facts: list[RawFact] = field(default_factory=list)

    def add(self, pred: str, *args, source: str = "输入数据") -> None:
        self.facts.append(RawFact(pred, tuple(args), source))

    def by_pred(self, pred: str) -> list[RawFact]:
        return [f for f in self.facts if f.pred == pred]

    def preds(self) -> list[str]:
        out: list[str] = []
        for f in self.facts:
            if f.pred not in out:
                out.append(f.pred)
        return out

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.facts:
            out[f.pred] = out.get(f.pred, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def to_atoms(self) -> list[Atom]:
        """转为一阶原子公式（去重，保持顺序稳定）。"""
        seen: set[Atom] = set()
        out: list[Atom] = []
        for f in self.facts:
            a = f.as_atom()
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out

    def to_tuples(self) -> list[tuple]:
        """转为元组形式（去重，保持顺序稳定）。"""
        seen: set[tuple] = set()
        out: list[tuple] = []
        for f in self.facts:
            t = f.as_tuple()
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def __len__(self) -> int:
        return len(self.facts)

    def __iter__(self) -> Iterator[RawFact]:
        return iter(self.facts)

    def describe(self) -> str:
        lines = [f"事实总数 {len(self.facts)}（去重后 {len(self.to_tuples())}）"]
        for pred, n in self.counts().items():
            lines.append(f"  {pred:<18s} {n:>4d} 条")
        return "\n".join(lines)


# ===========================================================================
# 案例 → 事实
# ===========================================================================


def build_facts(case: EquityCase, *, shell_threshold: float = SHELL_CAPITAL_MAX) -> FactBase:
    """把股权案例转换为事实集合。

    转换规则
    --------
    实体      Person(P1) / Company(C1) / Listed(C3)            类型事实
    空壳      注册资本 ≤ 500 万元且无实质经营记录 → Shell(C20)  教学口径判定
    持股      Holds(P1, C1, 0.8)                              三元数值谓词
    担保      Guarantee(C7, C10, 3000)
    交易      Trade(C4, C3, 5000)
    亲属      Kin(P1, P2, 配偶)                                原始事实，保留关系名
    亲属佐证  KinEvidenced(P6, P7)                             供 R03 使用的规范化事实
    """
    case.validate()
    fb = FactBase()

    for e in case.entities:
        if e.is_person:
            fb.add("Person", e.eid)
        else:
            fb.add("Company", e.eid)
            if e.is_listed:
                fb.add("Listed", e.eid)
            if e.category == "国有企业":
                fb.add("StateOwned", e.eid)
            if _is_shell(case, e.eid, shell_threshold):
                fb.add("Shell", e.eid, source="空壳判定")

    for h in case.holdings:
        fb.add("Holds", h.holder, h.target, h.ratio)

    for g in case.guarantees:
        fb.add("Guarantee", g.guarantor, g.beneficiary, g.amount)

    for t in case.trades:
        fb.add("Trade", t.seller, t.buyer, t.amount)

    for k in case.kinships:
        fb.add("Kin", k.a, k.b, k.relation)
        fb.add("Kin", k.b, k.a, k.relation, source="亲属关系对称化")
        # 非配偶亲属需要公开佐证才进入一致行动推断（规则 R03）
        if k.relation != "配偶" and k.evidence:
            fb.add("KinEvidenced", k.a, k.b, source="亲属佐证")
            fb.add("KinEvidenced", k.b, k.a, source="亲属佐证对称化")

    for a, b, depth in elevated_reach(case):
        fb.add("ElevatedReach", a, b, source=f"路径预处理: 存在长度{depth}的简单路径, 搜索上限8")
    return fb


def _is_shell(case: EquityCase, cid: str, threshold: float) -> bool:
    """空壳公司判定：注册资本不超过阈值且样本未见交易（只是教学代理，非真实空壳认定）。"""
    e = case.entity(cid)
    if e.is_person or e.capital > threshold:
        return False
    has_trade = any(t.seller == cid or t.buyer == cid for t in case.trades)
    return not has_trade


# ===========================================================================
# 派生事实：多层可达（供 R17 使用）
# ===========================================================================


def elevated_reach(case: EquityCase, min_depth: int = 3, max_depth: int = 8) -> list[tuple]:
    """找出持股链深度 ≥ min_depth 的 (股东, 目标) 对。

    这一趟计算放在知识库侧而非规则侧，是因为"深度"是路径结构属性，
    在一阶逻辑里表达需要引入列表结构或函数项递归，成本远高于其价值。
    报告中将此作为"分层知识表示"的例子说明：结构特征用算法抽取，
    业务结论用逻辑规则推导。
    """
    adj: dict[str, list[str]] = {}
    for h in case.holdings:
        adj.setdefault(h.holder, []).append(h.target)

    out: list[tuple] = []

    def walk(node: str, depth: int, seen: tuple[str, ...], origin: str) -> None:
        if depth > max_depth:
            return
        for nxt in adj.get(node, []):
            if nxt in seen:
                continue
            if depth >= min_depth:
                out.append((origin, nxt, depth))
            walk(nxt, depth + 1, seen + (nxt,), origin)

    for e in case.entities:
        walk(e.eid, 1, (e.eid,), e.eid)
    # 去重（同一对可能有多条路径达到不同深度），保留最短深度
    best: dict[tuple[str, str], int] = {}
    for a, b, d in out:
        key = (a, b)
        if key not in best or d < best[key]:
            best[key] = d
    return [(a, b, d) for (a, b), d in sorted(best.items())]


def fact_predicates_used_by(rules) -> set[str]:
    """规则库中出现的全部谓词（前件+后件），用于核对事实是否齐备。"""
    used: set[str] = set()
    for r in rules:
        for p in r.antecedents:
            used.add(p.pred)
        used.add(r.consequent.pred)
    return used


def check_fact_coverage(case: EquityCase, rules) -> dict:
    """检查事实库是否覆盖规则库所需的输入谓词。

    返回 {'provided': [...], 'missing': [...]}。
    missing 里的谓词若属于派生谓词（如 Controls），是正常的；
    若属于输入谓词而缺失，说明数据层没有提供对应事实，规则永远不会触发。
    """
    from .rules import BASE_PREDS, DERIVED_PREDS

    fb = build_facts(case)
    provided = set(fb.preds())
    needed = fact_predicates_used_by(rules)
    missing = sorted(needed - provided - DERIVED_PREDS)
    return {
        "provided": sorted(provided),
        "missing": missing,
        "missing_base": sorted(set(missing) & BASE_PREDS),
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
    sys.stdout.reconfigure(encoding="utf-8")
    from core.kb.domain import load_from_equity_kg
    from core.kb.rules import build_rules

    c = load_from_equity_kg()
    fb = build_facts(c)
    print(fb.describe())
    print()
    cov = check_fact_coverage(c, build_rules())
    print("规则所需输入谓词缺失情况：", cov["missing_base"] or "无")
    print("多层可达（深度≥3）：", elevated_reach(c)[:8])
