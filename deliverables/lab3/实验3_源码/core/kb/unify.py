"""合一、归结与包摄：确定性推理的语义核心。

本模块实现三层递进的能力：

    ① 合一 unify        —— 求两个项/原子的最一般合一者 MGU（带出现检查的递归合一）
    ② 归结 resolve      —— 子句对的归结式（含因子 factoring），是消解原理的直接实现
    ③ 包摄 subsumes     —— 子句间的包含关系判定，用于知识库精简（去掉更弱的子句）

与第一次汇报的「元组模式匹配」相比，这里的差别是实质性的：
第一次的匹配只能把变元绑到常量上（一对一），无法处理函数项嵌套、
无法求出最一般合一者、也无法做因子化与包摄。归结反演强依赖于后者。

约定
----
合一者用 {变量名: 项} 字典表示，允许链式绑定；使用 walk/resolve_term 完全解引用。
"""

from __future__ import annotations

import itertools
from typing import Iterable, Iterator, Optional

from .fol import (
    Atom,
    Clause,
    Const,
    Func,
    Literal,
    Num,
    Subst,
    Term,
    Var,
    literal_key,
    symbol_key,
    term_key,
)

# ===========================================================================
# 一、替换的解析与施加
# ===========================================================================


def walk(t: Term, s: Subst) -> Term:
    """沿绑定链把变量解引到底：{x→y, y→a} 上的 x 得到 a。"""
    seen = set()
    while isinstance(t, Var) and t.name in s:
        if t.name in seen:  # 循环绑定，理论上是异常的，防御性处理
            break
        seen.add(t.name)
        t = s[t.name]
    return t


def resolve_term(t: Term, s: Subst, depth: int = 0) -> Term:
    """施加替换并递归重建，得到一个完全解引用的项。"""
    t = walk(t, s)
    if isinstance(t, Func):
        return Func(t.name, tuple(resolve_term(a, s, depth + 1) for a in t.args), skolem_id=t.skolem_id)
    return t


def resolve_atom(a: Atom, s: Subst) -> Atom:
    """施加替换到一个原子公式。"""
    if not a.args:
        return a
    return Atom(a.pred, tuple(resolve_term(t, s) for t in a.args))


def resolve_literal(l: Literal, s: Subst) -> Literal:
    return Literal(resolve_atom(l.atom, s), l.negated)


def resolve_clause(c: Clause, s: Subst) -> Clause:
    """施加替换到一个子句，并重新规范化。"""
    return Clause.make([resolve_literal(l, s) for l in c.literals], c.origin, c.cid)


# ===========================================================================
# 二、合一
# ===========================================================================


class UnificationError(Exception):
    """两个项无法合一。"""


def occurs(v: Var, t: Term, s: Subst, seen: Optional[set] = None) -> bool:
    """出现检查：变量 v 是否出现在项 t 中（解引用后）。

    缺少这一检查时，合一算法会用 x ↦ f(x) 这样的无限项"解"出方程而发散。
    在一阶逻辑中没有无限项，因此这种绑定必须被拒绝。
    """
    t = walk(t, s)
    if isinstance(t, Var):
        return t.name == v.name
    if isinstance(t, Func):
        return any(occurs(v, a, s) for a in t.args)
    return False


def _bind(v: Var, t: Term, s: Subst) -> Subst:
    """扩展合一者，保证解引用不变量。"""
    t = walk(t, s)
    if isinstance(t, Var) and t.name == v.name:
        return s
    out = dict(s)
    out[v.name] = t
    return out


def unify_terms(a: Term, b: Term, s: Optional[Subst] = None) -> Optional[Subst]:
    """求两个项的最一般合一者；不可合一时返回 None。"""
    s = {} if s is None else s
    a = walk(a, s)
    b = walk(b, s)

    if isinstance(a, Var) or isinstance(b, Var):
        if isinstance(a, Var) and isinstance(b, Var) and a.name == b.name:
            return s
        if isinstance(a, Var):
            if occurs(a, b, s):
                return None
            return _bind(a, b, s)
        if occurs(b, a, s):
            return None
        return _bind(b, a, s)

    if isinstance(a, Const) and isinstance(b, Const):
        return s if symbol_key(a) == symbol_key(b) else None

    if isinstance(a, Num) and isinstance(b, Num):
        return s if a.value == b.value else None

    if isinstance(a, Func) and isinstance(b, Func):
        if symbol_key(a) != symbol_key(b) or len(a.args) != len(b.args):
            return None
        for x, y in zip(a.args, b.args):
            s = unify_terms(x, y, s)
            if s is None:
                return None
        return s

    # 类型不同（如 Const 与 Func）：不可合一
    return None


def unify_atoms(a: Atom, b: Atom, s: Optional[Subst] = None) -> Optional[Subst]:
    """求两个原子公式的最一般合一者。谓词名与元数必须相同。"""
    if a.pred != b.pred or len(a.args) != len(b.args):
        return None
    s = {} if s is None else s
    for x, y in zip(a.args, b.args):
        s = unify_terms(x, y, s)
        if s is None:
            return None
    return s


def unify_literals(a: Literal, b: Literal, s: Optional[Subst] = None) -> Optional[Subst]:
    """求两个文字的最一般合一者：极性不同则按取反后处理，极性相同要求原子可合一。"""
    if a.negated != b.negated:
        # 一个正一个负，属于"互补文字"的对立面，不能算合一成功，
        # 调用方应使用 complementary() 判断是否构成归结机会。
        return None
    return unify_atoms(a.atom, b.atom, s)


def complementary(a: Literal, b: Literal) -> bool:
    """两个文字是否互补（同一原子、极性相反），即是否构成归结的前提。"""
    return a.negated != b.negated and a.atom.pred == b.atom.pred and len(a.atom.args) == len(b.atom.args)


# ===========================================================================
# 三、变量重命名（refresh）
# ===========================================================================


_refresh_counter = itertools.count(1)


def refresh_clause(c: Clause) -> Clause:
    """子句变量重命名。

    每条子句的变量都是隐式全称量化的，参与归结前必须换名，
    否则不同子句的同名变量会被误当作同一个变量而被错误绑定。
    """
    if not c.variables:
        return c
    tag = next(_refresh_counter)
    s: Subst = {v.name: Var(f"{v.name}#{tag}") for v in sorted(c.variables, key=lambda v: v.name)}
    return resolve_clause(c, s)


# ===========================================================================
# 四、归结
# ===========================================================================


def _resolve_on(a: Literal, b: Literal) -> Optional[Subst]:
    """求使两个文字互补的合一者：把 a 的原子与 b 的原子合一（忽略极性）。"""
    if not complementary(a, b):
        return None
    return unify_atoms(a.atom, b.atom, {})


def factors(c: Clause) -> list[Clause]:
    """所有一步因子。因子是额外的逻辑后果，不可用首个因子替换原子句。"""
    out = []
    for i, j in itertools.combinations(range(len(c.literals)), 2):
        a, b = c.literals[i], c.literals[j]
        if a.negated != b.negated:
            continue
        s = unify_atoms(a.atom, b.atom, {})
        if s is not None:
            merged = resolve_clause(c, s)
            if not merged.is_tautology and len(merged) < len(c) and merged not in out:
                out.append(merged)
    return out


def factor(c: Clause) -> Optional[Clause]:
    """兼容入口：取首个一步因子；搜索器必须同时保留原子句和其他因子。"""
    items = factors(c)
    return items[0] if items else None


def factor_closure(c: Clause) -> list[Clause]:
    out, seen = [c], {c}
    for item in out:
        for child in factors(item):
            if child not in seen:
                seen.add(child)
                out.append(child)
    return out


def resolve(c1: Clause, c2: Clause, *, do_factor: bool = True) -> list[Clause]:
    """所有二元归结式；启用因子化时保留原子句及其全部因子。"""
    a, b = refresh_clause(c1), refresh_clause(c2)
    out, seen = [], set()
    left = factor_closure(a) if do_factor else [a]
    right = factor_closure(b) if do_factor else [b]
    for ca in left:
        for cb in right:
            for i, la in enumerate(ca.literals):
                for j, lb in enumerate(cb.literals):
                    s = _resolve_on(la, lb)
                    if s is None:
                        continue
                    rest = [l for k, l in enumerate(ca.literals) if k != i]
                    rest += [l for k, l in enumerate(cb.literals) if k != j]
                    new = Clause.make([resolve_literal(l, s) for l in rest],
                                      origin=f"归结({c1.cid},{c2.cid})")
                    if not new.is_tautology and new not in seen:
                        seen.add(new)
                        out.append(new)
    return out


# ===========================================================================
# 五、单向匹配与包摄（用于知识库精简与搜索剪枝）
# ===========================================================================


def match_atom(pattern: Atom, target: Atom, s: Optional[Subst] = None) -> Optional[Subst]:
    """单向匹配；目标里的变量是刚性的，绝不能继续被绑定。"""
    if pattern.pred != target.pred or len(pattern.args) != len(target.args):
        return None
    out = dict(s or {})
    def match(p, t):
        if isinstance(p, Var):
            if p.name in out:
                return out[p.name] == t
            out[p.name] = t
            return True
        if isinstance(p, Func):
            return (isinstance(t, Func) and symbol_key(p) == symbol_key(t) and len(p.args) == len(t.args)
                    and all(match(a, b) for a, b in zip(p.args, t.args)))
        return p == t
    return out if all(match(p, t) for p, t in zip(pattern.args, target.args)) else None


def subsumes(general: Clause, specific: Clause) -> bool:
    """θ-包摄：存在只作用于 general 的替换，使 generalθ ⊆ specific。

    不能在每个文字重新清空绑定，也不能按原文字数量直接拒绝（实例化可合并文字）。
    空子句包摄所有子句；变量重命名后仍只将 general 侧变量作为可绑定对象。
    """
    if not general.literals:
        return True
    if not specific.literals:
        return False
    if not general.variables:
        return set(general.literals) <= set(specific.literals)
    g, target = refresh_clause(general), refresh_clause(specific)
    def backtrack(i: int, subst: Subst) -> bool:
        if i == len(g.literals):
            return True
        literal = g.literals[i]
        for candidate in target.literals:
            if literal.negated != candidate.negated:
                continue
            extension = match_atom(literal.atom, candidate.atom, subst)
            if extension is not None and backtrack(i + 1, extension):
                return True
        return False
    return backtrack(0, {})


def eliminate_subsumed(clauses: Iterable[Clause]) -> tuple[list[Clause], int]:
    """删除被子句集中的其他子句所包摄的子句。

    返回 (精简后的子句集, 删除数量)。
    复杂度 O(n²)（对每次比较做一次合一回溯），本项目知识库规模下完全可接受；
    更大规模时应使用谓词签名索引；本实现用于教学规模与回归检查。
    """
    items = list(clauses)
    keep = []
    removed = 0
    for i, c in enumerate(items):
        redundant = False
        for j, d in enumerate(items):
            if i == j:
                continue
            if subsumes(d, c):
                # 双向包摄时保留编号较小的那个，保证结果稳定
                if subsumes(c, d) and j > i:
                    continue
                redundant = True
                break
        if redundant:
            removed += 1
        else:
            keep.append(c)
    return keep, removed


def clause_signature(c: Clause) -> tuple:
    """子句的谓词签名（谓词名 + 极性 + 元数），用于快速粗筛。"""
    return tuple(sorted((l.negated, l.atom.pred, l.atom.arity) for l in c.literals))


def clause_key(c: Clause) -> tuple:
    """子句的稳定排序键。"""
    return tuple(literal_key(l) for l in c.literals)
