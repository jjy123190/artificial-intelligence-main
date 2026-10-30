"""安全正 Horn 规则的正向不动点求值：朴素/真正 delta 驱动的半朴素连接。

支持无函数项、有限常量域、正前件、显式守卫；不支持默认否定或撤销结论。
优先级只决定同轮顺序，不饿死低优先级规则。达到预算时抛错，不能假称收敛。
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable
from core.kb.rules_dsl import Binding, Pattern, Rule, RuleSet, eval_guard, is_var, validate_value

Fact = tuple


class InferenceLimitError(RuntimeError):
    """资源预算耗尽：结果不完整，不能当作不可证明。"""


class WorkingMemory:
    """保序去重并按谓词/任意已绑定参数索引。不把数值转为字符串。"""
    def __init__(self, facts: Iterable[Fact] = ()):
        self.facts: set[Fact] = set()
        self.order: list[Fact] = []
        self._by_pred: dict[str, list[Fact]] = defaultdict(list)
        self._by_pos: dict[tuple, list[Fact]] = defaultdict(list)
        for fact in facts:
            self.add(tuple(fact))

    def add(self, fact: Fact) -> bool:
        if not fact or not isinstance(fact[0], str) or not fact[0]:
            raise ValueError("事实须包含非空谓词名")
        for arg in fact[1:]:
            validate_value(arg)
            if is_var(arg):
                raise ValueError(f"事实必须是基事实，不能含变量: {fact}")
        if fact in self.facts:
            return False
        self.facts.add(fact)
        self.order.append(fact)
        self._by_pred[fact[0]].append(fact)
        for i, arg in enumerate(fact[1:]):
            self._by_pos[(fact[0], i, arg)].append(fact)
        return True

    def candidates(self, pat: Pattern, binding: Binding, *, indexed: bool = True):
        if not indexed:
            return self.order
        best = self._by_pred.get(pat.pred, [])
        for i, arg in enumerate(pat.args):
            value = binding.get(arg, arg) if is_var(arg) else arg
            if not is_var(value):
                hit = self._by_pos.get((pat.pred, i, value), [])
                if len(hit) < len(best):
                    best = hit
        return best

    def of_pred(self, pred: str) -> list[Fact]:
        return sorted(self._by_pred.get(pred, []), key=str)

    def __contains__(self, fact: Fact) -> bool:
        return fact in self.facts

    def __len__(self) -> int:
        return len(self.facts)


def match_pattern(pat: Pattern, fact: Fact, binding: Binding) -> Binding | None:
    """单向模式匹配，严格校验重复变量及来自其他前件的已有绑定。"""
    if pat.pred != fact[0] or len(pat.args) != len(fact) - 1:
        return None
    out = dict(binding)
    for arg, value in zip(pat.args, fact[1:]):
        if is_var(arg):
            if arg in out and out[arg] != value:
                return None
            out[arg] = value
        elif arg != value:
            return None
    return out


def _instantiate(pat: Pattern, binding: Binding) -> Fact:
    args = tuple(binding.get(a, a) if is_var(a) else a for a in pat.args)
    if any(is_var(a) for a in args):
        raise ValueError(f"未绑定后件/前提: {pat}, {binding}")
    return (pat.pred, *args)


def _fmt_fact(f: Fact) -> str:
    return f"{f[0]}({','.join(map(str, f[1:]))})" if len(f) > 1 else f[0]


@dataclass
class Step:
    cycle: int
    rule: str
    facts: list[Fact]
    binding: Binding

    def __str__(self):
        return f"[周期{self.cycle}] {self.rule} ⇒ {', '.join(map(_fmt_fact, self.facts))}"


class ForwardEngine:
    def __init__(self, rules: RuleSet, facts: Iterable[Fact], *, max_cycles: int = 10000,
                 max_facts: int = 100000, use_index: bool = True):
        self.rules = rules
        self.input_facts = list(dict.fromkeys(tuple(f) for f in facts))
        self.wm = WorkingMemory(self.input_facts)
        for f in self.input_facts:
            if f[0] in rules._arity and len(f) - 1 != rules._arity[f[0]]:
                raise ValueError(f"事实元数与规则不符: {f}")
        self.max_cycles, self.max_facts, self.use_index = max_cycles, max_facts, use_index
        self.why: dict[Fact, tuple[str, list[Fact]]] = {}
        self.steps: list[Step] = []
        self.rounds: list[dict] = []
        self.cycles = self.candidate_checks = self.rule_instances = 0
        self.elapsed = 0.0
        self.converged = False

    def _join(self, rule: Rule, delta_wm: WorkingMemory | None = None, anchor: int | None = None):
        """delta 直接参与连接，而不是先做全量连接再过滤。"""
        pats = rule.antecedents
        def visit(todo: tuple[int, ...], binding: Binding):
            verdict = eval_guard(rule.guard, binding)
            if verdict is False:
                return
            if not todo:
                if verdict is None:
                    raise ValueError(f"规则 {rule.name} 完成连接后守卫仍未绑定")
                yield binding
                return
            def candidates(i):
                source = delta_wm if i == anchor else self.wm
                return source.candidates(pats[i], binding, indexed=self.use_index)
            # delta 位置优先；随后用当前绑定下最小的候选集，减少中间连接。
            i = anchor if anchor in todo else min(todo, key=lambda j: (len(candidates(j)), j))
            rest = tuple(j for j in todo if j != i)
            for fact in candidates(i):
                self.candidate_checks += 1
                nxt = match_pattern(pats[i], fact, binding)
                if nxt is not None:
                    yield from visit(rest, nxt)
        yield from visit(tuple(range(len(pats))), {})

    def match_rule(self, rule: Rule, delta: set[Fact] | WorkingMemory | None = None) -> list[Binding]:
        if delta is None or not rule.antecedents:
            streams = [self._join(rule)]
        else:
            dwm = delta if isinstance(delta, WorkingMemory) else WorkingMemory(sorted(delta, key=str))
            streams = [self._join(rule, dwm, i) for i, p in enumerate(rule.antecedents)
                       if dwm._by_pred.get(p.pred)]
        seen, out = set(), []
        for stream in streams:
            for b in stream:
                key = tuple(sorted(b.items()))
                if key not in seen:
                    seen.add(key)
                    out.append(b)
        self.rule_instances += len(out)
        return out

    def run(self, *, semi_naive: bool = True, max_cycles: int | None = None) -> int:
        """每次从同一输入重新运行，便于公平对比；返回新增事实的轮数。"""
        limit = self.max_cycles if max_cycles is None else max_cycles
        if limit < 1 or self.max_facts < 1:
            raise ValueError("推理预算须为正数")
        self.wm = WorkingMemory(self.input_facts)
        if len(self.wm) > self.max_facts:
            raise InferenceLimitError("输入事实已超过 max_facts")
        self.why.clear()
        self.steps.clear()
        self.rounds.clear()
        self.cycles = self.candidate_checks = self.rule_instances = 0
        self.converged = False
        delta = WorkingMemory(self.input_facts)
        t0 = time.perf_counter()
        try:
            while True:
                pending: dict[Fact, tuple[Rule, Binding]] = {}
                for rule in sorted(self.rules, key=lambda r: (-r.priority, r.name)):
                    for b in self.match_rule(rule, delta if semi_naive else None):
                        fact = _instantiate(rule.consequent, b)
                        if fact not in self.wm:
                            pending.setdefault(fact, (rule, b))
                if not pending:
                    self.converged = True
                    break
                if self.cycles >= limit:
                    raise InferenceLimitError(f"超过 {limit} 轮，尚未到达不动点")
                if len(self.wm) + len(pending) > self.max_facts:
                    raise InferenceLimitError("超过 max_facts，结果不完整")
                self.cycles += 1
                delta = WorkingMemory()
                fired = set()
                for fact, (rule, b) in pending.items():
                    self.wm.add(fact)
                    delta.add(fact)
                    self.why[fact] = rule.name, [_instantiate(p, b) for p in rule.antecedents]
                    self.steps.append(Step(self.cycles, rule.name, [fact], b))
                    fired.add(rule.name)
                self.rounds.append({"cycle": self.cycles, "rules_fired": sorted(fired),
                                    "new_facts": len(delta), "total_facts": len(self.wm)})
        finally:
            self.elapsed = time.perf_counter() - t0
        return self.cycles

    def query(self, pred: str, *args) -> list[Fact]:
        pat = Pattern.of(pred, *args)
        return [f for f in self.wm.of_pred(pred) if match_pattern(pat, f, {}) is not None]

    def holds(self, fact: Fact) -> bool:
        return fact in self.wm

    def explain(self, fact: Fact, *, depth: int | None = None, rules: RuleSet | None = None) -> list[str]:
        if fact not in self.wm:
            return [f"{_fmt_fact(fact)} [知识库无法证明；不等于现实中为假]"]
        desc = {r.name: r.desc for r in (self.rules if rules is None else rules)}
        def rec(f, level):
            pad = "    " * level
            if f not in self.why:
                return [f"{pad}• {_fmt_fact(f)} [输入/预处理事实]"]
            rule, premises = self.why[f]
            lines = [f"{pad}• {_fmt_fact(f)} ⇐ {rule} {desc.get(rule, '')}"]
            if depth is not None and level >= depth:
                return lines + [pad + "    [展示深度限制，完整证明在 traces.json]"]
            for premise in premises:
                lines.extend(rec(premise, level + 1))
            return lines
        return rec(fact, 0)

    def conclude(self, pred: str) -> list[Fact]:
        return sorted((f for f in self.why if f[0] == pred), key=str)

    def stats(self) -> dict:
        return {"cycles": self.cycles, "facts": len(self.wm), "derived_facts": len(self.why),
                "candidate_checks": self.candidate_checks, "rule_instances": self.rule_instances,
                "elapsed_ms": self.elapsed * 1000, "converged": self.converged,
                "peak_new_facts": max((r["new_facts"] for r in self.rounds), default=0)}

    def trace_text(self, limit: int | None = None) -> str:
        rounds = self.rounds if limit is None else self.rounds[:limit]
        return "\n".join(f"周期 {r['cycle']:3d} 触发 {','.join(r['rules_fired'])} "
                         f"新增 {r['new_facts']} 累计 {r['total_facts']}" for r in rounds)


def run_forward(rules: RuleSet, facts: Iterable[Fact], **kw) -> ForwardEngine:
    eng = ForwardEngine(rules, facts)
    eng.run(**kw)
    return eng
