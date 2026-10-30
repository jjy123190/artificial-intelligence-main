"""目标驱动的表格化反向推理（tabled goal evaluation）。

每个变体子目标拥有答案表，按规则做 SLD 式头合一与完整前件连接。
递归依赖先登记为表，不在 Python 调用栈中反复展开；迭代传播新答案至不动点。
这不是调用正向引擎，也不是只返回第一个子目标候选的深度优先搜索。

对本项目的有限、无函数项、安全正 Horn 规则，表和基答案均有限，饱和时得到
所请求目标的最小模型答案。预算耗尽抛 InferenceLimitError，不返回假阴性。
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Iterable
from core.kb.rules_dsl import Binding, Pattern, RuleSet, eval_guard, is_var
from .forward import Fact, InferenceLimitError, WorkingMemory, _fmt_fact, match_pattern


def _walk(value, subst):
    seen = set()
    while is_var(value) and value in subst:
        if value in seen:
            raise ValueError("循环变量替换")
        seen.add(value)
        value = subst[value]
    return value


def apply_subst_to_fact(fact: Fact, subst: Binding) -> Fact:
    return (fact[0], *(_walk(a, subst) for a in fact[1:]))


def unify_facts(left: Fact, right: Fact, subst: Binding | None = None) -> Binding | None:
    if left[0] != right[0] or len(left) != len(right):
        return None
    out = dict(subst or {})
    for a, b in zip(left[1:], right[1:]):
        a, b = _walk(a, out), _walk(b, out)
        if a == b:
            continue
        if is_var(a):
            out[a] = b
        elif is_var(b):
            out[b] = a
        else:
            return None
    return out


def merge_subst(a: Binding, b: Binding) -> Binding | None:
    out = dict(a)
    for key, value in b.items():
        out = unify_facts(("=", key), ("=", value), out)
        if out is None:
            return None
    return out


def _canonical(goal: Fact) -> Fact:
    """忽略变量名字，但保留重复变量的约束，让等价查询共用答案表。"""
    names = {}
    args = []
    for a in goal[1:]:
        if is_var(a):
            names.setdefault(a, f"?v{len(names)}")
            a = names[a]
        args.append(a)
    return (goal[0], *args)


@dataclass
class ProofNode:
    goal: Fact
    rule: str | None = None
    binding: Binding = field(default_factory=dict)
    children: list["ProofNode"] = field(default_factory=list)
    note: str = ""

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def lines(self, desc: dict | None = None, indent: int = 0) -> list[str]:
        pad = "    " * indent
        suffix = f"⇐ {self.rule} {(desc or {}).get(self.rule, '')}" if self.rule else "[输入/预处理事实]"
        out = [f"{pad}• {_fmt_fact(self.goal)} {suffix}"]
        for child in self.children:
            out.extend(child.lines(desc, indent + 1))
        return out

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def height(self) -> int:
        return 1 + max((c.height() for c in self.children), default=0)

    def rules_used(self) -> list[str]:
        return ([self.rule] if self.rule else []) + [r for c in self.children for r in c.rules_used()]


@dataclass
class _Table:
    goal: Fact
    answers: dict[Fact, ProofNode] = field(default_factory=dict)


class BackwardEngine:
    def __init__(self, rules: RuleSet, facts: Iterable[Fact], *, use_memo: bool = True,
                 max_rounds: int = 1000, max_tables: int = 50000,
                 depth_cap: int | None = None, use_iterative_deepening: bool | None = None):
        self.rules = rules
        self.wm = WorkingMemory(facts)
        for f in self.wm.order:
            if f[0] in rules._arity and len(f) - 1 != rules._arity[f[0]]:
                raise ValueError(f"事实元数与规则不符: {f}")
        self.facts = self.wm.order
        self.use_memo = use_memo
        # 保留旧调用的参数名，但现在是表格化不动点轮预算，不声称是证明深度。
        self.max_rounds = max_rounds if depth_cap is None else depth_cap
        self.max_tables = max_tables
        self.tables: dict[Fact, _Table] = {}
        self._revision = 0
        self._complete = False
        self._fresh = itertools.count()
        self.nodes_visited = self.rules_tried = self.cache_hits = self.rounds = 0
        self.elapsed = 0.0

    def _table(self, goal: Fact) -> _Table:
        key = _canonical(goal)
        if key in self.tables:
            self.cache_hits += 1
            return self.tables[key]
        if len(self.tables) >= self.max_tables:
            raise InferenceLimitError("反向推理超过子目标表预算")
        table = _Table(key)
        self.tables[key] = table
        self._revision += 1
        self.nodes_visited += 1
        pat = Pattern.of(key[0], *key[1:])
        for fact in self.wm.candidates(pat, {}):
            if match_pattern(pat, fact, {}) is not None:
                table.answers[fact] = ProofNode(fact)
        return table

    def _evaluate(self, table: _Table):
        for rule in self.rules.for_head(table.goal[0]):
            self.rules_tried += 1
            tag = next(self._fresh)
            names = {v: f"?r{tag}_{v[1:]}" for p in (*rule.antecedents, rule.consequent)
                     for v in p.variables()}
            def renamed(p):
                return (p.pred, *(names.get(a, a) for a in p.args))
            head = renamed(rule.consequent)
            subst = unify_facts(head, table.goal)
            if subst is None:
                continue
            body = [renamed(p) for p in rule.antecedents]
            def local_binding(s):
                return {v: val for v, fresh in names.items()
                        if not is_var(val := _walk(fresh, s))}
            def join(todo, current, proofs):
                binding = local_binding(current)
                verdict = eval_guard(rule.guard, binding)
                if verdict is False:
                    return
                if not todo:
                    if verdict is None:
                        raise ValueError(f"{rule.name} 完成连接后守卫仍未绑定")
                    answer = apply_subst_to_fact(head, current)
                    if any(is_var(a) for a in answer[1:]):
                        raise ValueError(f"不安全规则产生非基答案 {answer}")
                    if answer not in table.answers:
                        table.answers[answer] = ProofNode(answer, rule.name, binding,
                                                         [proofs[i] for i in range(len(body))])
                        self._revision += 1
                    return
                # 先做可直接读取的外延关系，再按已绑定参数数量选择子目标。
                def cost(i):
                    g = apply_subst_to_fact(body[i], current)
                    return (bool(self.rules.for_head(g[0])), sum(is_var(a) for a in g[1:]), i)
                i = min(todo, key=cost)
                goal = apply_subst_to_fact(body[i], current)
                child = self._table(goal)
                rest = tuple(j for j in todo if j != i)
                # 遍历全部答案：前一个前提的候选失败时必须回溯到其他候选。
                for answer, proof in list(child.answers.items()):
                    extension = unify_facts(goal, answer, current)
                    if extension is not None:
                        yield_proofs = dict(proofs)
                        yield_proofs[i] = proof
                        join(rest, extension, yield_proofs)
            join(tuple(range(len(body))), subst, {})

    def _query(self, goal: Fact, budget: int | None = None) -> _Table:
        Pattern.of(goal[0], *goal[1:])
        if goal[0] in self.rules._arity and len(goal) - 1 != self.rules._arity[goal[0]]:
            raise ValueError(f"查询元数与规则不符: {goal}")
        cap = self.max_rounds if budget is None else budget
        if cap < 1 or self.max_tables < 1:
            raise ValueError("推理预算须为正数")
        t0 = time.perf_counter()
        self.nodes_visited = self.rules_tried = self.cache_hits = self.rounds = 0
        if not self.use_memo:
            self.tables.clear()
            self._complete = False
        known = _canonical(goal) in self.tables
        try:
            root = self._table(goal)
            if known and self._complete:
                return root
            self._complete = False
            for round_ in range(1, cap + 1):
                # 新建子目标表和新答案都会增加revision；两者都停止变化才算完成。
                old_revision = self._revision
                for table in list(self.tables.values()):
                    self._evaluate(table)
                self.rounds = round_
                if self._revision == old_revision:
                    self._complete = True
                    return root
            raise InferenceLimitError(f"反向推理超过 {cap} 轮，答案尚未饱和")
        finally:
            self.elapsed = time.perf_counter() - t0

    def solve(self, goal: Fact, *, max_depth: int | None = None) -> ProofNode | None:
        table = self._query(goal, max_depth)
        answers = sorted(table.answers, key=str)
        return table.answers[answers[0]] if answers else None

    def ask(self, pred: str, *args) -> list[Fact]:
        return sorted(self._query((pred, *args)).answers, key=str)

    def explain(self, goal: Fact) -> list[str]:
        proof = self.solve(goal)
        return proof.lines({r.name: r.desc for r in self.rules}) if proof else [
            f"{_fmt_fact(goal)} [有限知识库无法证明，不是现实中的否定结论]"]

    def stats(self) -> dict:
        return {"rounds": self.rounds, "new_tables": self.nodes_visited,
                "table_count": len(self.tables), "rule_attempts": self.rules_tried,
                "cache_hits": self.cache_hits, "elapsed_ms": self.elapsed * 1000,
                "answer_count": sum(len(t.answers) for t in self.tables.values()),
                "converged": self._complete}


def run_backward(rules: RuleSet, facts: Iterable[Fact], goal: Fact, **kw) -> ProofNode | None:
    return BackwardEngine(rules, facts, **kw).solve(goal)
