"""结构推理的公平计时：同事实、同规则、同闭包；目标切片单独报告。

时间来自 perf_counter；候选检查次数、规则实例数、保留子句数是真实计数。
不使用 sleep 或模拟耗时。图构造/预处理不包含在推理内核时间中。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from core.kb.domain import load_from_equity_kg, scaled_case
from core.kb.facts import build_facts
from core.kb.rules import build_rules
from core.kb.fol import Clause, Literal, atom
from experiments.lab02.deterministic.forward import ForwardEngine
from experiments.lab02.deterministic.backward import BackwardEngine
from experiments.lab02.deterministic.resolution import ResolutionEngine, equity_resolution_example
from experiments.lab02.optimize.search import relevant_rules
from .artifacts import aggregate, execution_manifest, write_csv, write_json, seal_manifest

LAB_ROOT = Path(__file__).resolve().parents[3]


def _digest(facts):
    return hashlib.sha256(json.dumps(sorted(facts, key=str), ensure_ascii=False).encode("utf-8")).hexdigest()


def run_benchmarks(output: Path, *, groups=(1, 4, 8, 16), repeats=3, seed=42, progress=print) -> dict:
    if repeats < 1 or not groups or any(n < 1 for n in groups):
        raise ValueError("组数和重复次数须为正数")
    output.mkdir(parents=True, exist_ok=True)
    rules = build_rules()
    variants = {"naive_scan": (False, False), "indexed_naive": (True, False), "semi_naive": (True, True)}
    warm = build_facts(scaled_case(1, seed)).to_tuples()
    for indexed, semi in variants.values():
        ForwardEngine(rules, warm, use_index=indexed).run(semi_naive=semi)
    rows, slice_rows, queries = [], [], []
    for n in groups:
        progress(f"[基准] {n} 组股权结构，{repeats} 次重复")
        case = scaled_case(n, seed)
        facts = build_facts(case).to_tuples()
        reference = ForwardEngine(rules, facts)
        reference.run()
        closure = reference.wm.facts
        expected_controls = set(reference.query("Controls", "?x", "?y"))
        for repetition in range(repeats):
            # 交替顺序，降低固定执行顺序带来的系统负载/热身偏差。
            names = list(variants) if repetition % 2 == 0 else list(reversed(variants))
            for name in names:
                indexed, semi = variants[name]
                engine = ForwardEngine(rules, facts, use_index=indexed)
                engine.run(semi_naive=semi)
                if engine.wm.facts != closure:
                    raise AssertionError(f"{name} 与参考闭包不一致，不得报告加速")
                rows.append({"groups": n, "repetition": repetition, "model": name,
                             "entities": len(case.entities), "input_facts": len(facts),
                             "rules": len(rules), "closure_equal": True,
                             "closure_sha256": _digest(closure), **engine.stats()})
            pruned = ForwardEngine(relevant_rules(rules, {"Controls"}), facts)
            pruned.run()
            if set(pruned.query("Controls", "?x", "?y")) != expected_controls:
                raise AssertionError("目标依赖切片改变了 Controls 结论")
            slice_rows.append({"groups": n, "repetition": repetition, "target": "Controls",
                               "input_rules": len(rules), "retained_rules": len(pruned.rules),
                               "target_answers": len(expected_controls), "target_equal": True,
                               **pruned.stats()})
        goal = sorted(expected_controls, key=str)[-1] if expected_controls else ("Controls", "P1", "C1_0")
        backward = BackwardEngine(rules, facts)
        proof = backward.solve(goal)
        expected = goal in closure
        if bool(proof) != expected:
            raise AssertionError("反向单点查询与完整闭包不一致")
        queries.append({"groups": n, "query": str(goal), "expected": expected,
                        "proved": proof is not None, "proof_nodes": proof.size() if proof else 0,
                        **backward.stats()})
    resolution_rows = []
    clauses, goal = equity_resolution_example(load_from_equity_kg())
    # 明确添加无关事实，验证谓词索引/选择策略，不暗改需要证明的结论。
    clauses += [Clause.make([Literal(atom("Unrelated", f"noise{i}"))], origin="无关事实对照") for i in range(20)]
    for repetition in range(repeats):
        for optimized in (False, True):
            result = ResolutionEngine(optimized=optimized, max_pairs=250000, max_seconds=30).prove(clauses, goal)
            resolution_rows.append({"repetition": repetition, "model": "optimized" if optimized else "fifo_baseline",
                                    "input_clauses": len(clauses), "status": result.status,
                                    "proof_steps": len(result.proof_steps()), **result.stats})
            if not result.proved:
                write_csv(output / "resolution.csv", resolution_rows)
                raise AssertionError(f"归结基准未完成证明: {result.status} {result.reason}")
    summary = aggregate(rows, ("groups", "model"), ("elapsed_ms", "candidate_checks", "rule_instances", "cycles", "facts"))
    write_csv(output / "forward.csv", rows)
    write_csv(output / "forward_summary.csv", summary)
    write_csv(output / "goal_slicing.csv", slice_rows)
    write_csv(output / "backward_queries.csv", queries)
    write_csv(output / "resolution.csv", resolution_rows)
    manifest = execution_manifest(LAB_ROOT)
    manifest.update({"groups": groups, "repeats": repeats, "seed": seed,
                     "variants": {n: {"indexed": v[0], "semi_naive": v[1]} for n, v in variants.items()},
                     "scope": "forward.csv 各方法必须有完全相同闭包；goal_slicing.csv 只要求 Controls 答案相同",
                     "timing_scope": "推理内核计时，不含图构造、结构预处理、文件I/O；所有方法保留推理轨迹",
                     "search_space_note": "candidate_checks为实际候选检查次数，不是RAM字节数；facts为保留事实数",
                     "resolution_budget": {"max_pairs": 250000, "max_clauses": 3000, "max_seconds": 30},
                     "query_note": "反向表为单点目标查询，不能直接与全库物化的时间视为同等工作量"})
    write_json(output / "manifest.json", manifest)
    lines = ["# 确定性推理性能实测", "", "同一输入、27条规则、相同闭包；计时不含数据构建和文件I/O。", "",
             "| 组数 | 方法 | 平均毫秒±SD | 候选检查次数 | 事实数 |", "|---:|---|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['groups']} | {r['model']} | {r['elapsed_ms_mean']:.3f}±{r['elapsed_ms_sd']:.3f} | "
                     f"{r['candidate_checks_mean']:.0f} | {r['facts_mean']:.0f} |")
    lines += ["", "目标切片只要求保留 Controls 的答案，不声称计算了其他风险结论。",
              "backward_queries.csv 是单点查询，不与全库物化作不公平的直接倍数比较。",
              "resolution.csv 包含同一子案例、20条无关事实、FIFO基线与优化搜索的真实归结对数和计时。",
              "小数据上半朴素连接的分支/去重开销可能超过其收益；不预设任何规模下必然加速。", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    seal_manifest(output, manifest)
    progress(f"[基准] 完成，闭包一致性检查全部通过，结果位于 {output}")
    return {"forward": rows, "summary": summary, "resolution": resolution_rows, "queries": queries}
