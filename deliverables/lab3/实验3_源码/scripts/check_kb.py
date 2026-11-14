"""kb 层自检：不改动全局状态，直接跑一遍关键路径。

运行：python scripts/check_kb.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from core.kb.fol import (  # noqa: E402
    And,
    Clause,
    Const,
    Exists,
    Func,
    Iff,
    Implies,
    Literal,
    Not,
    Num,
    Or,
    Var,
    atom,
    rule_formula,
    to_clauses,
)
from core.kb.unify import (  # noqa: E402
    complementary,
    eliminate_subsumed,
    factor,
    refresh_clause,
    resolve,
    subsumes,
    unify_atoms,
    unify_terms,
)

ok = 0
fail = 0


def check(label, got, want):
    global ok, fail
    if got == want:
        ok += 1
        print(f"  [通过] {label}")
    else:
        fail += 1
        print(f"  [失败] {label}\n         期望 {want}\n         实际 {got}")


def show(label, value):
    print(f"  {label}: {value}")


def main() -> int:
    print("=" * 72)
    print("① 项与合一")
    print("=" * 72)
    check(
        "常量合一",
        unify_terms(Const("恒通科技"), Const("恒通科技"), {}),
        {},
    )
    check("常量与不同常量不可合一", unify_terms(Const("恒通科技"), Const("华鑫控股"), {}), None)
    show("变量绑定", unify_terms(Var("x"), Const("华鑫控股"), {}))
    show("结构化合一", unify_terms(Func("f", (Var("x"), Num(1))), Func("f", (Const("a"), Var("y"))), {}))
    check("出现检查拒绝 x↦f(x)", unify_terms(Var("x"), Func("f", (Var("x"),)), {}), None)
    check("元数不同不可合一", unify_atoms(atom("Holds", "?x", "?y", "?r"), atom("Holds", "?x", "?y"), {}), None)
    show("谓词不同返回 None", unify_atoms(atom("Holds", "?x"), atom("Controls", "?x"), {}))

    print()
    print("=" * 72)
    print("② 规范化：任意公式 → 子句集")
    print("=" * 72)
    f = rule_formula(
        [atom("Controls", "?x", "?y"), atom("Controls", "?y", "?z")],
        atom("Controls", "?x", "?z"),
    )
    cl = to_clauses(f)
    show("控制传递律", f)
    for c in cl:
        show("  子句", c)
    check("传递律得到 1 个子句", len(cl), 1)

    eq = to_clauses(Iff(atom("P", "?x"), atom("Q", "?x")))
    show("等值式 A↔B", eq)
    check("等值式拆成 2 个子句", len(eq), 2)

    ex = to_clauses(Exists((Var("y"),), atom("Controls", "王建国", "?y")))
    show("存在量词 Skolem 化", ex)
    check("Skolem 化产生常量", isinstance(ex[0].literals[0].atom.args[0], Const), True)

    uni = to_clauses(Implies(atom("P", "?x"), atom("P", "?x")))
    show("永真子句被删除", uni)
    check("P→P 规范化后无子句", len(uni), 0)

    dup = to_clauses(And((atom("Person", "?x"), atom("Person", "?x"))))
    check("重复子句去重", len(dup), 1)

    demorgan = to_clauses(Not(Or((atom("P", "?x"), atom("Q", "?x")))))
    show("¬(P∨Q) 否定内推", demorgan)
    check("De Morgan 得到 2 个子句", len(demorgan), 2)

    print()
    print("=" * 72)
    print("③ 归结")
    print("=" * 72)
    c1 = Clause.make([Literal(atom("Holds", "?x", "?y", "?r")), Literal(atom("Controls", "?x", "?y"))])
    c2 = Clause.make([Literal(atom("Holds", "王建国", "C1", Num(0.8)), True)])
    rs = resolve(c1, c2)
    show("二元归结式", rs)
    check("归结得到 Controls 单元子句", [str(r) for r in rs], ["Controls(王建国,C1)"])

    c3 = Clause.make([Literal(atom("Alert", "?c", "?t"))])
    c4 = Clause.make([Literal(atom("Alert", "?c", "?t"), True)])
    check("互补单元子句归结出空子句", [r.is_empty for r in resolve(c3, c4)], [True])

    c5 = Clause.make([Literal(atom("P", "?x")), Literal(atom("P", "a"))])
    show("因子化 P(x)∨P(a)", factor(c5))
    check("因子化得到更短子句", len(factor(c5)), 1)

    check("同名变量自动换名", refresh_clause(c3) == c3, False)

    print()
    print("=" * 72)
    print("④ 包摄与知识库精简")
    print("=" * 72)
    gen = Clause.make([Literal(atom("Holds", "?x", "?y", "?r"))])
    spc = Clause.make([Literal(atom("Holds", "王建国", "C1", Num(0.8)))])
    check("单文字子句包摄其实例", subsumes(gen, spc), True)
    check("实例不包摄一般式", subsumes(spc, gen), False)
    kept, removed = eliminate_subsumed([gen, spc])
    show(f"精简 {len([gen, spc])} → {len(kept)}，删除 {removed}", [str(c) for c in kept])
    check("冗余子句被删除", removed, 1)

    print()
    print("=" * 72)
    print(f"结果：{ok} 项通过，{fail} 项失败")
    print("=" * 72)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
