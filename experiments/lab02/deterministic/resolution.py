"""带证明 DAG 的教学归结反演器：given-clause、单元优先、谓词索引、包摄剪枝。

一般一阶归结可能不终止；资源耗尽明确返回 limit，不将其包装成反证失败。
保留所有一步因子与原子句。这里只处理逻辑子句，算术必须以显式事实输入。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from core.kb.fol import Atom, Clause, Const, Func, Literal, Num, Var, atom, facts_to_clauses, rule_formula, symbol_key, to_clauses
from core.kb.unify import factors, resolve, subsumes


def alpha_key(clause: Clause) -> tuple:
    """规范化首次出现的变量名；相等键一定为变量改名等价（可能漏去重，不误合并）。"""
    names = {}
    def term(t):
        if isinstance(t, Var):
            names.setdefault(t.name, len(names))
            return ("v", names[t.name])
        if isinstance(t, Num):
            return ("n", t.value)
        if isinstance(t, Const):
            return ("c", *symbol_key(t))
        return ("f", *symbol_key(t), tuple(term(a) for a in t.args))
    return tuple((l.negated, l.atom.pred, tuple(term(t) for t in l.atom.args)) for l in clause)


@dataclass
class ResolutionStep:
    cid: int
    clause: Clause
    parents: tuple[int, ...] = ()
    kind: str = "input"


@dataclass
class ResolutionResult:
    status: str
    steps: dict[int, ResolutionStep]
    final_id: int | None
    stats: dict = field(default_factory=dict)
    reason: str = ""

    @property
    def proved(self) -> bool:
        return self.status == "proved"

    def proof_steps(self) -> list[ResolutionStep]:
        if self.final_id is None:
            return []
        used = set()
        def visit(cid):
            if cid in used:
                return
            used.add(cid)
            for p in self.steps[cid].parents:
                visit(p)
        visit(self.final_id)
        return [self.steps[cid] for cid in sorted(used)]

    def lines(self) -> list[str]:
        return [f"[{s.cid}] {s.clause} ← {s.kind} {','.join(map(str, s.parents)) or s.clause.origin}"
                for s in self.proof_steps()]


class ResolutionEngine:
    def __init__(self, *, optimized: bool = True, max_clauses: int = 3000,
                 max_pairs: int = 100000, max_seconds: float = 10.0):
        self.optimized = optimized
        self.max_clauses, self.max_pairs, self.max_seconds = max_clauses, max_pairs, max_seconds

    def prove(self, clauses: list[Clause], goal: Atom) -> ResolutionResult:
        """把目标的否定加入知识库；导出空子句时沿父节点恢复完整反证路径。"""
        if self.max_clauses < 1 or self.max_pairs < 1 or self.max_seconds <= 0:
            raise ValueError("归结预算须为正数")
        t0 = time.perf_counter()
        steps: dict[int, ResolutionStep] = {}
        pending, active, seen = [], [], set()
        index: dict[tuple, set[int]] = {}
        pairs = generated = pruned = selected = 0
        final_id = None
        def add(c, parents=(), kind="input"):
            nonlocal generated, pruned, final_id
            generated += 1
            key = alpha_key(c)
            if c.is_tautology or key in seen:
                pruned += 1
                return
            # 因子不能因被其原子句包摄就丢弃，否则二元归结可能失去所需的因子。
            if self.optimized and kind not in ("factor", "input", "goal") and not c.is_empty:
                if any(subsumes(s.clause, c) for s in steps.values()):
                    pruned += 1
                    return
            if len(steps) >= self.max_clauses:
                raise OverflowError("max_clauses")
            cid = len(steps) + 1
            c = Clause.make(c.literals, c.origin, cid)
            steps[cid] = ResolutionStep(cid, c, parents, kind)
            seen.add(key)
            pending.append(cid)
            if c.is_empty:
                final_id = cid
        reason, status = "", "saturated"
        try:
            for c in clauses:
                add(c)
            add(Clause.make([Literal(goal, True)], "否定待证目标"), kind="goal")
            while pending and final_id is None:
                if time.perf_counter() - t0 > self.max_seconds:
                    raise OverflowError("max_seconds")
                selected += 1
                # 每五步取最早子句，避免纯最短优先饿死长子句。
                if self.optimized and selected % 5:
                    cid = min(pending, key=lambda i: (len(steps[i].clause), steps[i].clause.max_term_depth, i))
                    pending.remove(cid)
                else:
                    cid = pending.pop(0)
                given = steps[cid].clause
                active.append(cid)
                for literal in given:
                    key = (literal.atom.pred, literal.negated, literal.atom.arity)
                    index.setdefault(key, set()).add(cid)
                for c in factors(given):
                    add(c, (cid,), "factor")
                if self.optimized:
                    candidates = set()
                    for literal in given:
                        candidates |= index.get((literal.atom.pred, not literal.negated, literal.atom.arity), set())
                else:
                    candidates = set(active)
                for other in sorted(candidates):
                    pairs += 1
                    if pairs > self.max_pairs or time.perf_counter() - t0 > self.max_seconds:
                        raise OverflowError("max_pairs/max_seconds")
                    for c in resolve(given, steps[other].clause, do_factor=False):
                        add(c, (cid, other), "resolution")
                        if final_id is not None:
                            break
                    if final_id is not None:
                        break
            if final_id is not None:
                status = "proved"
        except OverflowError as exc:
            status, reason = "limit", str(exc)
        return ResolutionResult(status, steps, final_id,
            {"pairs": pairs, "generated": generated, "retained": len(steps), "pruned": pruned,
             "selected": selected, "elapsed_ms": (time.perf_counter() - t0) * 1000}, reason)


def equity_resolution_example(case) -> tuple[list[Clause], Atom]:
    """把第一次 P1/C1/C2/C3 子案例的数值守卫显式化后做独立归结。

    此处只编译教学子案例，不声称把含算术的全库直接变成纯一阶子句。
    比例取自输入 CSV，不从正向/反向推理的结论生成证明前提。
    """
    holdings = {(h.holder, h.target): h.ratio for h in case.holdings}
    required = [("P1", "C1"), ("C1", "C2"), ("C2", "C3"), ("C1", "C3")]
    missing = [p for p in required if p not in holdings]
    if missing:
        raise ValueError(f"归结示例缺少持股输入: {missing}")
    values = [holdings[p] for p in required]
    inputs = [atom("Holds", a, b, holdings[a, b]) for a, b in required]
    for value in dict.fromkeys(values):
        if value > .5:
            inputs.append(atom("AboveHalf", value))
    if values[2] + values[3] > .5:
        inputs.append(atom("SumAboveHalf", values[2], values[3]))
    clauses = facts_to_clauses(inputs, "CSV/显式算术比较")
    rules = [
        ([atom("Holds", "?x", "?y", "?r"), atom("AboveHalf", "?r")], atom("DirectControls", "?x", "?y")),
        ([atom("DirectControls", "C1", "C2"), atom("Holds", "C2", "C3", values[2]),
          atom("Holds", "C1", "C3", values[3]), atom("SumAboveHalf", values[2], values[3])],
         atom("DirectControls", "C1", "C3")),
        ([atom("DirectControls", "?x", "?y")], atom("Controls", "?x", "?y")),
        ([atom("Controls", "P1", "C1"), atom("Controls", "C1", "C3")], atom("Controls", "P1", "C3")),
    ]
    for i, (body, head) in enumerate(rules, 1):
        clauses.extend(to_clauses(rule_formula(body, head), origin=f"股权子案例规则{i}"))
    return clauses, atom("Controls", "P1", "C3")
