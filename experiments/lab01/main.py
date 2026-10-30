"""课堂 Demo 主程序：知识表示 → 产生式推理 → 图推理 → 知识增强 → 可视化。

运行：python main.py           一次性输出全部步骤
      python main.py --step    每一步按回车继续（录制演示视频时控制节奏）
"""
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import unicodedata
from collections import defaultdict

from experiments.lab01.kr.frames import build_company_frames
from experiments.lab01.kr.knowledge_graph import INFERRED, SCHEMA, EquityKG
from experiments.lab01.kr.production import InferenceEngine
from experiments.lab01.reasoning.graph_reasoning import find_cycles, penetrate, relation_path
from experiments.lab01.reasoning.rules import PRED_CN, RULE_DESC, RULES
from experiments.lab01.viz import draw_kg, draw_kg_html, layout

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = str(PROJECT_ROOT / "outputs" / "lab01")
STEP = "--step" in sys.argv


def section(title):
    if STEP:
        input("\n按回车继续 ▶ ")
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def pad(s, width):
    """按终端显示宽度左对齐（中文占两列，str.ljust 按字符数会对不齐）。"""
    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)
    return s + " " * max(width - w, 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    from experiments.lab01.service import read_dataset
    data = read_dataset()

    # ------------------------------------------------------------------
    section("① 知识表示：知识图谱（模式层 + 数据层三元组）")
    kg = EquityKG(data)
    name = kg.name
    types = SCHEMA["entity_types"]
    print(f"模式层：实体类型 {'、'.join(types.values())}；关系类型：")
    for rel, (heads, tails, cn, attr) in SCHEMA["relations"].items():
        domain = f"{'|'.join(types[t] for t in heads)} → {'|'.join(types[t] for t in tails)}"
        print(f"    {pad(cn, 10)}{pad(domain, 22)}属性 {pad(attr, 10)}{'（推理得出，回写图谱）' if rel in INFERRED else ''}")
    print(f"数据层：实体 {kg.G.number_of_nodes()} 个，关系 {kg.G.number_of_edges()} 条。部分三元组：")
    for t in kg.triples()[:6]:
        print("   ", t)
    pos = layout(kg)
    draw_kg(kg, os.path.join(OUT, "kg_raw.png"), pos, "企业股权知识图谱（推理前）")
    print("图谱已保存：outputs/lab01/kg_raw.png")

    section("② 知识表示：一阶谓词（由图谱自动转换，作为产生式系统的初始事实）")
    facts = kg.to_predicates()

    def fmt(f):
        args = []
        for index, value in enumerate(f[1:]):
            if isinstance(value, str) and value in kg.G:
                args.append(name(value))
            elif isinstance(value, (int, float)) and f[0] == "Holds" and index == 2:
                args.append(f"{value:.0%}")
            elif isinstance(value, (int, float)) and f[0] in ("Trade", "Guarantee") and index == 2:
                args.append(f"{value:g}万元")
            else:
                args.append(str(value))
        return f"{PRED_CN.get(f[0], f[0])}({', '.join(args)})"

    for pred in ("Person", "IsA", "Holds", "Kin", "Guarantee", "Trade"):
        f = min((f for f in facts if f[0] == pred), key=str)
        print(pad(f"    {f[0]}({', '.join(map(str, f[1:]))})", 32) + f"即 {fmt(f)}")
    print(f"    …… 共 {len(facts)} 条事实")

    section("③ 知识表示：框架（企业档案，含继承与缺省值）")
    classes, frames = build_company_frames(data)
    print(frames["C3"].show())

    # ------------------------------------------------------------------
    section("④ 产生式推理：正向链（匹配 → 冲突消解 → 执行 → 终止判断）")
    engine = InferenceEngine(RULES, facts)
    cycles = engine.run()
    print(f"规则库 {len(RULES)} 条，推理 {cycles} 个周期，综合数据库由 {len(facts)} 条扩充到 {len(engine.wm)} 条")
    print("\n实际控制人：")
    controlled = defaultdict(list)
    for _, c, p in engine.query("ActualController", "?c", "?p"):
        controlled[p].append(name(c))
    for p, cs in sorted(controlled.items()):
        print(f"    {name(p)} → {'、'.join(cs)}")
    print("\n关联交易：")
    for _, a, b, p in engine.query("RelatedTrade", "?a", "?b", "?p"):
        print(f"    {name(a)} → {name(b)}  （共同实控人：{name(p)}）")
    print("\n风险预警：")
    alerts = engine.query("Alert", "?c", "?t")
    by_type, risks = defaultdict(list), defaultdict(list)
    for _, c, t in alerts:
        by_type[t].append(name(c))
        risks[c].append(t)
    for t, cs in by_type.items():
        print(f"    [{t}] {'、'.join(cs)}")

    section("⑤ 推理解释：为什么判定『恒通科技存在关联交易待核查』？")
    print("\n".join(engine.explain(("Alert", "C3", "关联交易待核查"), fmt, rule_desc=RULE_DESC)))

    # ------------------------------------------------------------------
    section("⑥ 图推理：股权穿透（教学筛查阈值 ≥ 25%，非法律认定）")
    for target in ("C3", "C12"):
        print(f"\n【{name(target)}】")
        for o in penetrate(kg, target):
            flag = "★达到教学筛查阈值" if o["beneficial"] else ""
            print(f"    {pad(o['name'], 14)}穿透持股 {o['ratio']:6.2%} {flag}")
            for path, r in o["paths"]:
                print(f"        路径: {' → '.join(name(n) for n in path)}  ({r:.2%})")

    section("⑦ 图推理：环路检测 与 关联路径查询")
    for c in find_cycles(kg, "HOLDS"):
        print("    循环持股：", " → ".join(name(n) for n in c + c[:1]))
    for c in find_cycles(kg, "GUARANTEES"):
        print("    担保圈：  ", " → ".join(name(n) for n in c + c[:1]))
    print(f"\n    {name('C3')} 与 {name('C9')} 的关联路径：")
    for u, rel, v, d in relation_path(kg, "C3", "C9"):
        print(f"        {name(u)} {d}[{SCHEMA['relations'][rel][2]}]{d} {name(v)}")

    # ------------------------------------------------------------------
    section("⑧ 知识增强：推理结论回写框架与知识图谱")
    for _, c, p in engine.query("ActualController", "?c", "?p"):
        frames[c].set("实际控制人", name(p))
        kg.add_relation(p, "CONTROLS", c, engine.why[("Controls", p, c)][0])   # 边上记录推出控制关系的规则
    for _, c, level in engine.query("RiskLevel", "?c", "?l"):
        frames[c].set("风险等级", level)
    for c, ts in risks.items():
        frames[c].set("风险提示", "、".join(ts))
        kg.G.nodes[c]["risks"] = ts
    print(frames["C3"].show())
    added = sum(1 for *_, d in kg.G.edges(data=True) if d["rel"] in INFERRED)
    print(f"\n知识图谱新增『实际控制』关系 {added} 条、风险标签 {len(risks)} 个")
    draw_kg(kg, os.path.join(OUT, "kg_reasoned.png"), pos, "推理增强后的知识图谱（回写实际控制关系与风险标签）",
            enriched=True)
    print("图谱已保存：outputs/lab01/kg_reasoned.png")
    if draw_kg_html(kg, os.path.join(OUT, "kg_reasoned.html")):
        print("交互式图谱已保存：outputs/lab01/kg_reasoned.html")


if __name__ == "__main__":
    main()
