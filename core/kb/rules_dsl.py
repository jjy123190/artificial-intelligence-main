"""安全的、无函数项的正 Horn 规则 DSL。

前后向引擎共享模式、守卫和绑定语义。一般一阶项与归结在 fol/unify 中独立实现。
数值守卫不能在导出子句时被忽略；带守卫规则必须先显式构造算术事实。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence
from .fol import Atom, Clause, atom, rule_formula, to_clauses

Binding = dict[str, object]
Guard = Callable[[Binding], bool]


def is_var(x) -> bool:
    return isinstance(x, str) and x.startswith("?")


def var_name(x: str) -> str:
    return x[1:] if is_var(x) else x


def validate_value(x) -> None:
    if isinstance(x, bool) or not isinstance(x, (str, int, float)):
        raise TypeError("规则 DSL 只支持字符串和有限数值；函数项请使用 fol 模块")
    if isinstance(x, (int, float)) and not math.isfinite(x):
        raise ValueError("事实/模式不能包含 NaN 或无穷大")


@dataclass(frozen=True)
class Pattern:
    pred: str
    args: tuple = ()

    def __post_init__(self):
        if not self.pred or not isinstance(self.pred, str):
            raise ValueError("谓词名须为非空字符串")
        for arg in self.args:
            validate_value(arg)
            if arg == "?":
                raise ValueError("变量需要名称，不能仅写 ?")

    @staticmethod
    def of(pred: str, *args) -> "Pattern":
        return Pattern(pred, tuple(args))

    def to_atom(self) -> Atom:
        return atom(self.pred, *self.args)

    def variables(self) -> list[str]:
        return [a for a in self.args if is_var(a)]

    def key(self) -> tuple:
        return self.pred, self.args

    def __str__(self) -> str:
        return f"{self.pred}({','.join(map(str, self.args))})" if self.args else self.pred


@dataclass(frozen=True)
class GuardSpec:
    """显式声明依赖变量，避免通过闭包/函数 id 猜测守卫依赖。"""
    required: frozenset[str]
    predicate: Guard
    display_conditions: tuple[str, ...] | None = None

    def __call__(self, binding: Binding) -> bool:
        return bool(self.predicate(binding))


def guard_vars(guard: Guard | None) -> frozenset[str]:
    if guard is None:
        return frozenset()
    if isinstance(guard, GuardSpec):
        return guard.required
    # 兼容简单教学 lambda；正式规则使用 GuardSpec。不缓存 id，避免对象 id 复用。
    names, seen = set(), set()
    def scan(v):
        if id(v) in seen:
            return
        seen.add(id(v))
        if isinstance(v, str) and is_var(v):
            names.add(v)
        elif isinstance(v, (tuple, list, set, frozenset)):
            for x in v:
                scan(x)
        elif hasattr(v, "co_consts"):
            scan(v.co_consts)
        elif callable(v):
            scan(getattr(v, "__code__", None))
            scan(getattr(v, "__defaults__", None))
            for cell in getattr(v, "__closure__", None) or ():
                scan(cell.cell_contents)
    scan(guard)
    return frozenset(names)


def guard_ready(guard: Guard | None, binding: Binding) -> bool:
    return all(v in binding and not is_var(binding[v]) for v in guard_vars(guard))


def eval_guard(guard: Guard | None, binding: Binding) -> bool | None:
    """None 仅表示未绑定；守卫实现错误抛出异常，不能悄悄变成否定结论。"""
    if guard is None:
        return True
    if not guard_ready(guard, binding):
        return None
    try:
        return bool(guard(binding))
    except Exception as exc:
        raise ValueError(f"守卫求值失败，绑定={binding!r}") from exc


def _num(x) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"守卫需要有限数值，收到 {x!r}")
    return float(x)


def _display_value(value) -> str:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, str) else str(value)


def gt(var: str, thresh: float) -> GuardSpec:
    return GuardSpec(frozenset([var]), lambda b: _num(b[var]) > thresh,
                     (f"{var} > {_display_value(thresh)}",))


def ge(var: str, thresh: float) -> GuardSpec:
    return GuardSpec(frozenset([var]), lambda b: _num(b[var]) >= thresh,
                     (f"{var} ≥ {_display_value(thresh)}",))


def ge_var(var: str, other: str) -> GuardSpec:
    return GuardSpec(frozenset([var, other]), lambda b: _num(b[var]) >= _num(b[other]),
                     (f"{var} ≥ {other}",))


def gt_var(var: str, other: str) -> GuardSpec:
    return GuardSpec(frozenset([var, other]), lambda b: _num(b[var]) > _num(b[other]),
                     (f"{var} > {other}",))


def ratio_sum_gt(vars_: Sequence[str], thresh: float) -> GuardSpec:
    return GuardSpec(frozenset(vars_), lambda b: sum(_num(b[v]) for v in vars_) > thresh,
                     (f"{' + '.join(vars_) or '0'} > {_display_value(thresh)}",))


def ratio_sum_le(vars_: Sequence[str], thresh: float) -> GuardSpec:
    return GuardSpec(frozenset(vars_), lambda b: sum(_num(b[v]) for v in vars_) <= thresh,
                     (f"{' + '.join(vars_) or '0'} ≤ {_display_value(thresh)}",))


def neq(a: str, b: str) -> GuardSpec:
    return GuardSpec(frozenset([a, b]), lambda bd: bd[a] != bd[b], (f"{a} ≠ {b}",))


def eq_var(a: str, b: str) -> GuardSpec:
    return GuardSpec(frozenset([a, b]), lambda bd: bd[a] == bd[b], (f"{a} = {b}",))


def equals(var: str, value) -> GuardSpec:
    return GuardSpec(frozenset([var]), lambda b: b[var] == value,
                     (f"{var} = {_display_value(value)}",))


def approximately(var: str, value: float, tolerance: float = 1e-9) -> GuardSpec:
    return GuardSpec(frozenset([var]), lambda b: abs(_num(b[var]) - value) <= tolerance,
                     (f"abs({var} - {_display_value(value)}) ≤ {_display_value(tolerance)}",))


def all_of(*guards: Guard) -> GuardSpec:
    required = frozenset().union(*(guard_vars(g) for g in guards))
    # Display metadata travels with the same helper parameters as the evaluator.
    # An opaque callback must not make a partially described guard look complete.
    complete = all(isinstance(g, GuardSpec) and g.display_conditions is not None for g in guards)
    conditions = tuple(c for g in guards for c in (g.display_conditions or ())) if complete else None
    return GuardSpec(required, lambda b: all(g(b) for g in guards), conditions)


@dataclass
class Rule:
    name: str
    desc: str
    antecedents: tuple[Pattern, ...]
    consequent: Pattern
    guard: Guard | None = None
    priority: int = 0
    group: str = ""
    note: str = ""

    def __post_init__(self):
        body_vars = {v for p in self.antecedents for v in p.variables()}
        required = set(self.consequent.variables()) | set(guard_vars(self.guard))
        if not required <= body_vars:
            raise ValueError(f"不安全规则 {self.name}: {sorted(required - body_vars)} 未在前件绑定")

    def to_formula(self):
        if self.guard is not None:
            raise ValueError(f"{self.name} 含数值/比较守卫，不能静默丢弃后转为无条件子句")
        return rule_formula([p.to_atom() for p in self.antecedents], self.consequent.to_atom())

    def to_clauses(self) -> list[Clause]:
        return to_clauses(self.to_formula(), origin=self.name)

    def head_pred(self) -> str:
        return self.consequent.pred

    def __str__(self) -> str:
        guard = " [含守卫]" if self.guard else ""
        return f"{self.name}: {' ∧ '.join(map(str, self.antecedents))} → {self.consequent}{guard}"


class RuleSet:
    def __init__(self, rules: Sequence[Rule] = ()):
        self.rules: list[Rule] = []
        self._by_head: dict[str, list[Rule]] = {}
        self._arity: dict[str, int] = {}
        for r in rules:
            self.add(r)

    def add(self, rule: Rule) -> None:
        if any(r.name == rule.name for r in self.rules):
            raise ValueError(f"重复规则编号 {rule.name}")
        arities = dict(self._arity)
        for p in (*rule.antecedents, rule.consequent):
            if p.pred in arities and arities[p.pred] != len(p.args):
                raise ValueError(f"同一谓词元数不一致: {p.pred}")
            arities[p.pred] = len(p.args)
        self._arity = arities
        self.rules.append(rule)
        self._by_head.setdefault(rule.head_pred(), []).append(rule)

    def reindex(self) -> None:
        current = list(self.rules)
        self.__init__(current)

    def for_head(self, pred: str) -> list[Rule]:
        return self._by_head.get(pred, [])

    def by_group(self, group: str) -> list[Rule]:
        return [r for r in self.rules if r.group == group]

    def clause_base(self) -> list[Clause]:
        return list(dict.fromkeys(c for r in self.rules for c in r.to_clauses()))

    @property
    def groups(self) -> list[str]:
        return list(dict.fromkeys(r.group for r in self.rules if r.group))

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self) -> Iterator[Rule]:
        return iter(self.rules)

    def describe(self) -> str:
        lines = []
        for r in self.rules:
            lines.extend([str(r), f"  {r.desc}"])
            if r.note:
                lines.append(f"  边界：{r.note}")
        return "\n".join(lines)
