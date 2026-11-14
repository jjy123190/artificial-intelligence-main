"""产生式系统：规则库 + 综合数据库 + 推理机（正向推理）。

事实用一阶谓词的元组形式表示，如 ("Holds", "P1", "C1", 0.8) 即 Holds(P1, C1, 0.8)；
规则中以 "?" 开头的字符串为变元，如 ("Controls", "?x", "?y")。
推理机每个周期执行：匹配 → 冲突消解 → 执行规则 → 检查终止条件。
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional


def is_var(x):
    return isinstance(x, str) and x.startswith("?")


def substitute(pattern, binding):
    return tuple(binding.get(p, p) if is_var(p) else p for p in pattern)


@dataclass
class Rule:
    name: str
    desc: str
    conditions: list                   # 前提：谓词模式（AND 连接）
    conclusions: list                  # 结论：谓词模式
    test: Optional[Callable] = None    # 附加条件，如 lambda b: b["?r"] > 0.5
    priority: int = 0                  # 冲突消解用：数值大的优先


class WorkingMemory:
    """综合数据库：存放已知事实与推理得到的中间结论。"""

    def __init__(self, facts=()):
        self.facts = set()
        self.by_pred = defaultdict(set)          # 谓词名 -> 事实
        self.by_first = defaultdict(set)         # (谓词名, 第一个参数) -> 事实，加速匹配
        for f in facts:
            self.add(f)

    def add(self, fact):
        if fact in self.facts:
            return False
        self.facts.add(fact)
        self.by_pred[fact[0]].add(fact)
        if len(fact) > 1:
            self.by_first[(fact[0], fact[1])].add(fact)
        return True

    def candidates(self, pattern, binding):
        first = pattern[1] if len(pattern) > 1 else None
        if first is not None and is_var(first):
            first = binding.get(first)
        if first is not None:
            return self.by_first.get((pattern[0], first), ())
        return self.by_pred.get(pattern[0], ())

    def __contains__(self, fact):
        return fact in self.facts

    def __len__(self):
        return len(self.facts)


def unify(pattern, fact, binding):
    if len(pattern) != len(fact):
        return None
    b = dict(binding)
    for p, f in zip(pattern, fact):
        if is_var(p):
            if p in b and b[p] != f:
                return None
            b[p] = f
        elif p != f:
            return None
    return b


class InferenceEngine:
    def __init__(self, rules, facts):
        self.rules = rules
        self.wm = WorkingMemory(facts)
        self.why = {}          # 推出的事实 -> (规则名, 前提事实列表)，用于解释
        self.trace = []        # (周期, 规则名, 新事实)

    def match(self, rule):
        """匹配：找出使规则全部前提在综合数据库中成立的所有变元绑定。"""
        bindings = [{}]
        for cond in rule.conditions:
            bindings = [nb for b in bindings for f in self.wm.candidates(cond, b)
                        if (nb := unify(cond, f, b)) is not None]
            if not bindings:
                return []
        return [b for b in bindings if rule.test is None or rule.test(b)]

    def run(self, max_cycles=10000):
        cycle = 0
        while cycle < max_cycles:
            # 1. 匹配：构造冲突集（只保留能产生新事实的规则实例）
            conflict = []
            for rule in self.rules:
                for b in self.match(rule):
                    new = [substitute(c, b) for c in rule.conclusions]
                    if any(f not in self.wm for f in new):
                        conflict.append((rule, b, new))
            # 4. 终止条件：没有可产生新结论的规则
            if not conflict:
                break
            # 2. 冲突消解：选优先级最高的规则（同级按规则库顺序），执行它的全部实例
            chosen = max(conflict, key=lambda x: x[0].priority)[0]
            cycle += 1
            # 3. 执行：把结论加入综合数据库，并记录推理依据
            for rule, b, new in conflict:
                if rule is not chosen:
                    continue
                premises = [substitute(c, b) for c in rule.conditions]
                for f in new:
                    if self.wm.add(f):
                        self.why[f] = (rule.name, premises)
                        self.trace.append((cycle, rule.name, f))
        return cycle

    def query(self, *pattern):
        return sorted((f for f in self.wm.by_pred.get(pattern[0], ())
                       if unify(pattern, f, {}) is not None), key=str)

    def explain(self, fact, fmt=str, indent=0, rule_desc=None):
        """解释推理链（为什么得出这个结论），返回多行文本。"""
        pad = "    " * indent
        if fact not in self.why:
            return [f"{pad}• {fmt(fact)}  [已知事实]"]
        rule_name, premises = self.why[fact]
        desc = f" {rule_desc[rule_name]}" if rule_desc else ""
        lines = [f"{pad}• {fmt(fact)}  ⇐ {rule_name}{desc}"]
        for p in premises:
            lines += self.explain(p, fmt, indent + 1, rule_desc)
        return lines
