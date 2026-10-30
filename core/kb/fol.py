"""一阶谓词逻辑：语法对象与规范化（Skolem 化、CNF 转换）。

领域：企业股权风险推理（第 6 组第二次汇报）。

本模块只负责"公式长什么样"，以及把任意公式整理成子句集的经典流程：

    ① 消除等值 ↔
    ② 消除蕴含 →
    ③ 否定内推（De Morgan 律与量词对偶），得到否定范式 NNF
    ④ 变量标准化（standardize apart），避免不同量词复用同名变量
    ⑤ Skolem 化：把存在量词换成 Skolem 常量或 Skolem 函数
    ⑥ 丢弃全称量词（其变量在归结时以重命名方式隐式处理，见 kb/unify.py）
    ⑦ 用分配律展开为合取范式 CNF
    ⑧ 抽取子句集

语义部分（合一、归结）见 kb/unify.py。

所有对象都是 frozen dataclass，可直接哈希，便于用 set / dict 去重与索引；
项与文字的排序统一走 term_key / literal_key，不依赖 Python 字符串哈希的随机化，
固定输入和调用顺序时可复现；全局新名编号取决于本进程此前的公式转换次数。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Iterator, Sequence, Union

# ===========================================================================
# 一、项（Term）——个体层面的语言对象
# ===========================================================================


@dataclass(frozen=True)
class Var:
    """逻辑变量 ?x。归结过程中唯一可以被绑定（binding）的对象。"""

    name: str

    def __str__(self) -> str:
        return f"?{self.name}"

    def __repr__(self) -> str:
        return f"Var({self.name})"


@dataclass(frozen=True)
class Const:
    """个体常量，如「恒通科技」「王建国」。领域知识里的具体对象。"""

    name: str
    skolem_id: int | None = field(default=None, kw_only=True)

    def __str__(self) -> str:
        return symbol_name(self)

    def __repr__(self) -> str:
        suffix = f",skolem_id={self.skolem_id}" if self.skolem_id is not None else ""
        return f"Const({self.name}{suffix})"


@dataclass(frozen=True)
class Num:
    """数值常量，如持股比例 0.65、担保金额 3000。

    单独设一个项类型，是为了让「持股 65%」这类知识能直接写进谓词，
    从而用归结本身处理比较条件外的结构化知识，而把数值比较交给规则的
    附加测试（见 kb/rules_dsl.py），避免在归结层引入算术推理。
    """

    value: int | float

    def __post_init__(self):
        if (isinstance(self.value, bool) or not isinstance(self.value, (int, float))
                or isinstance(self.value, float) and not math.isfinite(self.value)):
            raise ValueError("逻辑数值项必须为有限数")

    def __str__(self) -> str:
        v = self.value
        if isinstance(v, int):
            return str(v)
        return str(int(v)) if v.is_integer() else str(v)

    def __repr__(self) -> str:
        return f"Num({self.value})"


@dataclass(frozen=True)
class Func:
    """函数项，如 Skolem 化产生的 sk1(?x)。

    领域知识不直接使用函数项，但 Skolem 化一定会产生，
    因此语法层必须支持，否则含存在量词的公式无法规范化。
    """

    name: str
    args: tuple["Term", ...]
    skolem_id: int | None = field(default=None, kw_only=True)

    def __str__(self) -> str:
        return f"{symbol_name(self)}({','.join(str(a) for a in self.args)})"

    def __repr__(self) -> str:
        suffix = f",skolem_id={self.skolem_id}" if self.skolem_id is not None else ""
        return f"Func({self.name},{self.args!r}{suffix})"


Term = Union[Var, Const, Num, Func]
Subst = dict  # 类型别名：{变量名: 项}


def symbol_key(t: Const | Func) -> tuple:
    """用户名字与内部 Skolem 标识分开；后来输入同名常量也不能碰撞。"""
    return (t.name, t.skolem_id is not None, t.skolem_id if t.skolem_id is not None else 0)


def symbol_name(t: Const | Func) -> str:
    """证明文本明确显示内部符号；语义身份仍由 symbol_key 决定。"""
    return t.name if t.skolem_id is None else f"⟦{t.name}:{t.skolem_id}⟧"


def term_key(t: Term) -> tuple:
    """为项生成稳定且跨类型可比的排序键。

    直接比较 Var / Const 会因为类型不同抛 TypeError，故统一降级为元组比较：
    先按类型序号，再按名字（或数值、参数）。
    """
    if isinstance(t, Var):
        return (0, t.name)
    if isinstance(t, Const):
        return (1, *symbol_key(t))
    if isinstance(t, Num):
        return (2, t.value)
    return (3, *symbol_key(t), tuple(term_key(a) for a in t.args))


def substitute_term(t: Term, s: Subst) -> Term:
    """对单个项递归应用替换 s（只做一层，链式绑定由 unify.walk 负责解除）。"""
    if isinstance(t, Var):
        return s.get(t.name, t)
    if isinstance(t, Func):
        return Func(t.name, tuple(substitute_term(a, s) for a in t.args), skolem_id=t.skolem_id)
    return t


def term_vars(t: Term) -> set[Var]:
    """收集项中出现的全部变量。"""
    if isinstance(t, Var):
        return {t}
    if isinstance(t, Func):
        out: set[Var] = set()
        for a in t.args:
            out |= term_vars(a)
        return out
    return set()


def is_ground(t: Term) -> bool:
    """项是否不含变量（ground）。"""
    return not term_vars(t)


# ===========================================================================
# 二、原子公式、文字与子句
# ===========================================================================


@dataclass(frozen=True)
class Atom:
    """原子公式 Pred(t1,…,tn)。零元谓词写作 Pred()。"""

    pred: str
    args: tuple[Term, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.pred
        return f"{self.pred}({','.join(str(a) for a in self.args)})"

    def apply(self, s: Subst) -> "Atom":
        """对原子公式应用替换。"""
        if not self.args:
            return self
        return Atom(self.pred, tuple(substitute_term(a, s) for a in self.args))

    @property
    def arity(self) -> int:
        return len(self.args)


@dataclass(frozen=True)
class Literal:
    """文字 = 原子公式 + 是否取反。子句是文字的析取。"""

    atom: Atom
    negated: bool = False

    def __invert__(self) -> "Literal":
        return Literal(self.atom, not self.negated)

    def apply(self, s: Subst) -> "Literal":
        return Literal(self.atom.apply(s), self.negated)

    def __str__(self) -> str:
        return f"¬{self.atom}" if self.negated else str(self.atom)


def literal_key(l: Literal) -> tuple:
    """文字的稳定排序键（正文字在前，再按谓词名与参数）。"""
    return (l.negated, l.atom.pred, tuple(term_key(a) for a in l.atom.args))


def literal_vars(l: Literal) -> set[Var]:
    out: set[Var] = set()
    for a in l.atom.args:
        out |= term_vars(a)
    return out


@dataclass(frozen=True)
class Clause:
    """子句：文字的有限析取。

    literals 保持规范化（去重 + 稳定排序），因此内容相同的子句一定相等；
    origin / cid 只用于打印证明过程，不参与相等性与哈希（compare=False）。
    """

    literals: tuple[Literal, ...] = ()
    origin: str = field(default="", compare=False)
    cid: int = field(default=0, compare=False)

    @staticmethod
    def make(literals, origin: str = "", cid: int = 0) -> "Clause":
        """构造规范化子句：去重并按 literal_key 排序。"""
        uniq = sorted(set(literals), key=literal_key)
        return Clause(tuple(uniq), origin, cid)

    @property
    def is_empty(self) -> bool:
        """空子句，即推出矛盾（归结证明成功）。"""
        return not self.literals

    @property
    def is_tautology(self) -> bool:
        """永真子句：同时含 P 与 ¬P，对归结无贡献，可安全删除。"""
        pos = {l.atom for l in self.literals if not l.negated}
        neg = {l.atom for l in self.literals if l.negated}
        return bool(pos & neg)

    @property
    def is_unit(self) -> bool:
        """单文字子句，归结中的强约束（单元归结优先策略的核心）。"""
        return len(self.literals) == 1

    @property
    def variables(self) -> set[Var]:
        out: set[Var] = set()
        for l in self.literals:
            out |= literal_vars(l)
        return out

    @property
    def max_term_depth(self) -> int:
        """子句中项的最大嵌套深度，用于把搜索导向更"具体"的子句。"""

        def depth(t: Term) -> int:
            if isinstance(t, Func):
                return 1 + max((depth(a) for a in t.args), default=0)
            return 0

        return max((depth(a) for l in self.literals for a in l.atom.args), default=0)

    def __len__(self) -> int:
        return len(self.literals)

    def __iter__(self) -> Iterator[Literal]:
        return iter(self.literals)

    def __str__(self) -> str:
        if self.is_empty:
            return "□"
        return " ∨ ".join(str(l) for l in self.literals)

    def pretty(self) -> str:
        """带编号与来源的展示形式，用于打印归结证明过程。"""
        tag = f"[{self.cid}]" if self.cid else "[]"
        src = f"  ← {self.origin}" if self.origin else ""
        return f"{tag} {self}{src}"


# ===========================================================================
# 三、复合公式
# ===========================================================================


class Formula:
    """公式基类，仅用于类型标注。"""

    __slots__ = ()


@dataclass(frozen=True)
class Not(Formula):
    sub: Formula

    def __str__(self) -> str:
        return f"¬{self.sub}"


@dataclass(frozen=True)
class And(Formula):
    subs: tuple[Formula, ...]

    def __str__(self) -> str:
        return "(" + " ∧ ".join(str(s) for s in self.subs) + ")"


@dataclass(frozen=True)
class Or(Formula):
    subs: tuple[Formula, ...]

    def __str__(self) -> str:
        return "(" + " ∨ ".join(str(s) for s in self.subs) + ")"


@dataclass(frozen=True)
class Implies(Formula):
    lhs: Formula
    rhs: Formula

    def __str__(self) -> str:
        return f"({self.lhs} → {self.rhs})"


@dataclass(frozen=True)
class Iff(Formula):
    lhs: Formula
    rhs: Formula

    def __str__(self) -> str:
        return f"({self.lhs} ↔ {self.rhs})"


@dataclass(frozen=True)
class Forall(Formula):
    vars: tuple[Var, ...]
    sub: Formula

    def __str__(self) -> str:
        vs = ",".join(str(v) for v in self.vars)
        return f"∀{vs}({self.sub})"


@dataclass(frozen=True)
class Exists(Formula):
    vars: tuple[Var, ...]
    sub: Formula

    def __str__(self) -> str:
        vs = ",".join(str(v) for v in self.vars)
        return f"∃{vs}({self.sub})"


def substitute(f: Formula, s: Subst) -> Formula:
    """捕获规避的同时替换；绑定变量屏蔽同名替换。"""
    if isinstance(f, (Atom, Literal)):
        return f.apply(s)
    if isinstance(f, Not):
        return Not(substitute(f.sub, s))
    if isinstance(f, (And, Or)):
        return type(f)(tuple(substitute(g, s) for g in f.subs))
    if isinstance(f, (Implies, Iff)):
        return type(f)(substitute(f.lhs, s), substitute(f.rhs, s))
    if isinstance(f, (Forall, Exists)):
        hidden = {v.name for v in f.vars}
        inner = {k: v for k, v in s.items() if k not in hidden}
        replacement_vars = {v.name for t in inner.values() for v in term_vars(t)}
        used = {v.name for v in formula_vars(f)} | replacement_vars | set(s)
        renaming = {}
        for v in f.vars:
            if v.name in replacement_vars:
                name = f"{v.name}__{next(_rename_counter)}"
                while name in used:
                    name = f"{v.name}__{next(_rename_counter)}"
                used.add(name)
                renaming[v.name] = Var(name)
        variables = tuple(renaming.get(v.name, v) for v in f.vars)
        body = substitute(f.sub, renaming) if renaming else f.sub
        return type(f)(variables, substitute(body, inner))
    return f


def formula_vars(f: Formula) -> set[Var]:
    """收集公式中出现的全部变量（含量化变量）。"""
    if isinstance(f, Literal):
        return atom_vars(f.atom)
    if isinstance(f, Atom):
        out: set[Var] = set()
        for a in f.args:
            out |= term_vars(a)
        return out
    if isinstance(f, Not):
        return formula_vars(f.sub)
    if isinstance(f, (And, Or)):
        out = set()
        for g in f.subs:
            out |= formula_vars(g)
        return out
    if isinstance(f, (Implies, Iff)):
        return formula_vars(f.lhs) | formula_vars(f.rhs)
    if isinstance(f, (Forall, Exists)):
        return formula_vars(f.sub) | set(f.vars)
    return set()


def atom_vars(a: Atom) -> set[Var]:
    """原子公式中出现的变量。"""
    out: set[Var] = set()
    for t in a.args:
        out |= term_vars(t)
    return out


def all_atoms(f: Formula) -> list[Atom]:
    """收集公式中出现的全部原子公式（用于统计谓词覆盖面）。"""
    if isinstance(f, Literal):
        return [f.atom]
    if isinstance(f, Atom):
        return [f]
    if isinstance(f, Not):
        return all_atoms(f.sub)
    if isinstance(f, (And, Or)):
        out: list[Atom] = []
        for g in f.subs:
            out.extend(all_atoms(g))
        return out
    if isinstance(f, (Implies, Iff)):
        return all_atoms(f.lhs) + all_atoms(f.rhs)
    if isinstance(f, (Forall, Exists)):
        return all_atoms(f.sub)
    return []


# ===========================================================================
# 四、规范化：任意公式 → 子句集
# ===========================================================================


def eliminate_iff(f: Formula) -> Formula:
    """① 消除等值：A ↔ B 等价于 (A→B) ∧ (B→A)。"""
    if isinstance(f, Iff):
        return And(
            (
                eliminate_iff(Implies(f.lhs, f.rhs)),
                eliminate_iff(Implies(f.rhs, f.lhs)),
            )
        )
    if isinstance(f, Not):
        return Not(eliminate_iff(f.sub))
    if isinstance(f, (And, Or)):
        return type(f)(tuple(eliminate_iff(g) for g in f.subs))
    if isinstance(f, Implies):
        return Implies(eliminate_iff(f.lhs), eliminate_iff(f.rhs))
    if isinstance(f, (Forall, Exists)):
        return type(f)(f.vars, eliminate_iff(f.sub))
    return f


def eliminate_implies(f: Formula) -> Formula:
    """② 消除蕴含：A → B 等价于 ¬A ∨ B。"""
    if isinstance(f, Implies):
        return Or((Not(eliminate_implies(f.lhs)), eliminate_implies(f.rhs)))
    if isinstance(f, Not):
        return Not(eliminate_implies(f.sub))
    if isinstance(f, (And, Or)):
        return type(f)(tuple(eliminate_implies(g) for g in f.subs))
    if isinstance(f, (Forall, Exists)):
        return type(f)(f.vars, eliminate_implies(f.sub))
    return f


def nnf(f: Formula, negated: bool = False) -> Formula:
    """③ 否定内推：把 ¬ 一路推到原子公式上，得到否定范式 NNF。

    negated 参数携带"当前处于多少次否定之下"的奇偶性，
    一次自顶向下遍历即可完成，不必先构造 Not 再内外翻转。
    """
    if isinstance(f, Literal):
        return ~f if negated else f
    if isinstance(f, Not):
        return nnf(f.sub, not negated)
    if isinstance(f, And):
        parts = tuple(nnf(g, negated) for g in f.subs)
        return Or(parts) if negated else And(parts)
    if isinstance(f, Or):
        parts = tuple(nnf(g, negated) for g in f.subs)
        return And(parts) if negated else Or(parts)
    if isinstance(f, Forall):
        inner = nnf(f.sub, negated)
        return Exists(f.vars, inner) if negated else Forall(f.vars, inner)
    if isinstance(f, Exists):
        inner = nnf(f.sub, negated)
        return Forall(f.vars, inner) if negated else Exists(f.vars, inner)
    return Not(f) if negated else f  # Atom：内推完成


_rename_counter = itertools.count(1)


def standardize_apart(f: Formula) -> Formula:
    """每个量词引入新作用域；离开内层作用域后恢复外层同名变量。"""
    used = {v.name for v in formula_vars(f)}
    def rec(g, mapping):
        if isinstance(g, (Forall, Exists)):
            inner = dict(mapping)
            fresh = []
            for v in g.vars:
                name = f"{v.name}__{next(_rename_counter)}"
                while name in used:
                    name = f"{v.name}__{next(_rename_counter)}"
                used.add(name)
                new = Var(name)
                inner[v.name] = new
                fresh.append(new)
            return type(g)(tuple(fresh), rec(g.sub, inner))
        if isinstance(g, Not):
            return Not(rec(g.sub, mapping))
        if isinstance(g, (And, Or)):
            return type(g)(tuple(rec(h, mapping) for h in g.subs))
        if isinstance(g, (Implies, Iff)):
            return type(g)(rec(g.lhs, mapping), rec(g.rhs, mapping))
        return substitute(g, mapping)
    return rec(f, {})


_skolem_counter = itertools.count(1)
_skolem_identity_counter = itertools.count(1)


def skolemize(f: Formula) -> Formula:
    """⑤ Skolem 化：消去存在量词。

    存在量词之外没有全称量词时，替换为 Skolem 常量 sk_i；
    处于 ∀x∀y 作用域内时，替换为 Skolem 函数 sk_i(x,y)，
    以保留"该个体随外层变量而变"的语义。
    内部标识跨调用唯一，且与用户常量/函数隔离；显示编号不承担身份语义。
    """

    used = set()
    def symbols(t):
        if isinstance(t, (Const, Func)):
            used.add(t.name)
        if isinstance(t, Func):
            for a in t.args:
                symbols(a)
    for a in all_atoms(f):
        for t in a.args:
            symbols(t)
    def mk_skolem(universals: tuple[Var, ...]) -> Term:
        name = f"sk{next(_skolem_counter)}"
        while name in used:
            name = f"sk{next(_skolem_counter)}"
        used.add(name)
        identity = next(_skolem_identity_counter)
        return (Func(name, universals, skolem_id=identity) if universals
                else Const(name, skolem_id=identity))

    def rec(g: Formula, universals: tuple[Var, ...]) -> Formula:
        if isinstance(g, Not):
            return Not(rec(g.sub, universals))
        if isinstance(g, (And, Or)):
            return type(g)(tuple(rec(h, universals) for h in g.subs))
        if isinstance(g, Forall):
            return Forall(g.vars, rec(g.sub, universals + g.vars))
        if isinstance(g, Exists):
            m: Subst = {v.name: mk_skolem(universals) for v in g.vars}
            return rec(substitute(g.sub, m), universals)
        return g

    return rec(f, ())


def drop_universals(f: Formula) -> Formula:
    """⑥ 丢弃全称量词：剩下公式中的自由变量默认全称量化。

    归结时通过变量重命名（refresh）保证各子句变量互不冲突，
    等价于每一步重新实例化全称量词。
    """
    if isinstance(f, Exists):
        raise ValueError("必须先消去存在量词")
    if isinstance(f, Forall):
        return drop_universals(f.sub)
    if isinstance(f, Not):
        return Not(drop_universals(f.sub))
    if isinstance(f, (And, Or)):
        return type(f)(tuple(drop_universals(g) for g in f.subs))
    return f


def to_clause_lists(f: Formula) -> list[list[Literal]]:
    """把公式转成子句集的列表表示：list[list[Literal]]。

    这是标准的递归分配算法，按公式结构自上而下处理：
        Atom P     → [[P]]
        ¬P         → [[¬P]]
        A ∧ B      → 两侧子句集直接合并
        A ∨ B      → 两侧子句集做笛卡尔积合并（分配律）
        残留量词   → 报错（必须先完成量词规范化）

    注意不同于「先转 CNF 树再抽取」的写法：这里让 Or 的两个子式各自先规范化，
    因此 ¬A ∨ ¬B 这类"两支都是子句集"的情况能正常处理，
    不必假设析取支是单个文字。分配到最后一层一定是文字，
    所以对 Atom / Not(Atom) 之外的输入直接报错，可尽早暴露规范化流程的缺陷。
    """
    if isinstance(f, (Forall, Exists)):
        raise ValueError("抽取子句前必须完成量词规范化")
    if isinstance(f, And):
        out: list[list[Literal]] = []
        for g in f.subs:
            out.extend(to_clause_lists(g))
        return out
    if isinstance(f, Or):
        parts = [to_clause_lists(g) for g in f.subs]
        acc = [[]]
        for p in parts:
            acc = [x + y for x in acc for y in p]
        return acc
    if isinstance(f, Literal):
        # 否定内推的产物：¬原子 已经被直接表示为带 negated 标记的文字
        return [[f]]
    if isinstance(f, Not):
        if not isinstance(f.sub, Atom):
            raise ValueError(f"内部错误：子句中出现非文字的否定 {f.sub}")
        return [[Literal(f.sub, True)]]
    if isinstance(f, Atom):
        return [[Literal(f)]]
    raise ValueError(f"内部错误：子句中出现非法子公式 {f}")


def to_cnf(f: Formula) -> Formula:
    """⑦ 分配律展开：把 NNF 转为合取范式（显式的 ∧ 树）。

    语义等价于 to_clause_lists，这里额外把结果重建为公式树，
    便于需要肉眼检查规范化中间结果的场景（scripts/dump_cnf.py）。
    """
    clauses = to_clause_lists(f)
    return And(tuple(Or(tuple(lits)) for lits in clauses))


def dedupe_clauses(clauses: Sequence[Clause]) -> list[Clause]:
    """按内容去重，并保持首次出现的顺序（保证输出稳定）。"""
    seen: set[Clause] = set()
    out: list[Clause] = []
    for c in clauses:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def extract_clauses(cnf: Formula, origin: str = "") -> list[Clause]:
    """⑧ 从 CNF 抽取子句集：规范化、去重、删除永真子句。

    永真子句（含 P 与 ¬P）恒为真，归结时对任何归结式都无贡献，删除可显著压缩搜索空间。
    """
    out = []
    for lits in to_clause_lists(cnf):
        c = Clause.make(lits, origin=origin)
        if not c.is_tautology:
            out.append(c)
    return dedupe_clauses(out)


def free_vars(f: Formula, bound: frozenset[str] = frozenset()) -> set[Var]:
    """自由变量默认全称闭包，保证 Skolem 函数依赖这些变量。"""
    if isinstance(f, Literal):
        return {v for v in literal_vars(f) if v.name not in bound}
    if isinstance(f, Atom):
        return {v for v in atom_vars(f) if v.name not in bound}
    if isinstance(f, (Forall, Exists)):
        return free_vars(f.sub, bound | {v.name for v in f.vars})
    if isinstance(f, Not):
        return free_vars(f.sub, bound)
    if isinstance(f, (And, Or)):
        return set().union(*(free_vars(g, bound) for g in f.subs))
    if isinstance(f, (Implies, Iff)):
        return free_vars(f.lhs, bound) | free_vars(f.rhs, bound)
    return set()


def to_clauses(f: Formula, origin: str = "") -> list[Clause]:
    """把任意一阶公式转换成子句集（八步流程总入口）。"""
    variables = tuple(sorted(free_vars(f), key=lambda v: v.name))
    g = eliminate_iff(Forall(variables, f) if variables else f)
    g = eliminate_implies(g)
    g = nnf(g)
    g = standardize_apart(g)
    g = skolemize(g)
    g = drop_universals(g)
    g = to_cnf(g)
    return extract_clauses(g, origin=origin)


# ===========================================================================
# 五、便捷构造器
# ===========================================================================


def is_var_name(x) -> bool:
    """约定：以 ? 开头的标识符视为变量名。"""
    return isinstance(x, str) and x.startswith("?")


def to_term(x) -> Term:
    """字符串 / 数值 → 项。

    '?x' → Var，数值 0.65 → Num，字符串 '0.65' 仍为 Const。
    已经是项对象时原样返回，便于 atom() 接受混合参数。
    """
    if isinstance(x, (Var, Const, Num, Func)):
        return x
    if isinstance(x, bool):  # bool 是 int 的子类，需先拦下
        raise TypeError("布尔值不是合法的逻辑项")
    if isinstance(x, (int, float)):
        return Num(x)
    if not isinstance(x, str):
        raise TypeError("逻辑项必须为项对象、数值或字符串")
    if is_var_name(x):
        return Var(x[1:])
    return Const(x)


def atom(pred: str, *args) -> Atom:
    """构造原子公式：atom('Holds', '?x', '?y', '?r')。"""
    return Atom(pred, tuple(to_term(a) for a in args))


def rule_formula(antecedents: Sequence[Atom], consequent: Atom) -> Formula:
    """由若干前件与一个后件构造全称量化的蕴含式 ∀x…(A1 ∧ … → C)。"""
    body: Formula
    if len(antecedents) > 1:
        body = Implies(And(tuple(antecedents)), consequent)
    elif len(antecedents) == 1:
        body = Implies(antecedents[0], consequent)
    else:
        body = consequent
    variables = sorted(
        {v for a in list(antecedents) + [consequent] for v in atom_vars(a)},
        key=lambda v: v.name,
    )
    return Forall(tuple(variables), body) if variables else body


def facts_to_clauses(facts: Sequence[Atom], origin: str = "知识库") -> list[Clause]:
    """一组基原子事实 → 单元子句集（f = 1 个正文字的子句）。"""
    return [Clause.make([Literal(a)], origin=origin) for a in facts]
