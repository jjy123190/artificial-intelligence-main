"""补齐第一次改进项的实测：交叉持股权益扩容、时态撤回、凭证控制冲突。

用线性方程求解与独立不动点迭代核对，时间来自真实perf_counter。
保存封闭100%环的无解状态，不将循环检测、经济权益和法律控制权混为一谈。
"""
from __future__ import annotations
import time
from pathlib import Path
from core.kb.domain import EquityCase, Entity, Holding, load_from_equity_kg
from core.kb.graph_io import case_to_dict
from core.kb.equity import economic_equity
from core.kb.governance import governance_snapshot, claim_changes
from .artifacts import write_json, write_csv, execution_manifest, seal_manifest, aggregate

ROOT = Path(__file__).resolve().parents[3]


def ring_case(n):
    case = EquityCase(entities=[Entity("P", "教学自然人", "Person")]
        + [Entity(f"C{i}", f"教学企业{i}", "Company", capital=1000) for i in range(n)],
        holdings=[Holding("P", "C0", .5)] + [Holding(f"C{i}", f"C{(i+1)%n}", .2) for i in range(n)])
    return case.validate()


def iterate(case):
    ids = [e.eid for e in case.companies]
    direct = {c: sum(h.ratio for h in case.holdings if h.target == c and h.holder == "P") for c in ids}
    values = {c: 0.0 for c in ids}
    for steps in range(1, 10001):
        updated = {c: direct[c] + sum(h.ratio * values[h.holder] for h in case.holdings
                   if h.target == c and h.holder in values) for c in ids}
        difference = max(abs(updated[c] - values[c]) for c in ids)
        values = updated
        if difference < 1e-13:
            return values, steps
    raise ValueError("独立迭代未收敛，不能报告通过")


def base_payload():
    return {"case": case_to_dict(load_from_equity_kg()), "as_of": "2026-10-22", "target": "C3",
            "strict_evidence": False, "strict_identity": False, "documents": [],
            "identities": [], "relations": [], "rights": []}


def run_governance_experiments(output: Path, *, sizes=(8, 32, 64, 128), repeats=3, progress=print):
    if not sizes or any(type(n) is not int or n < 2 for n in sizes) or type(repeats) is not int or repeats < 1:
        raise ValueError("权益规模至少2、重复次数正整数")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in sizes:
        case = ring_case(size)
        reference, steps = iterate(case)
        for repetition in range(repeats):
            result = economic_equity(case, "C0")
            if result["status"] != "solved":
                raise ValueError("非封闭教学环应可求解")
            share = next(o["share"] for o in result["owners"] if o["entity_id"] == "P")
            error = abs(share - reference["C0"])
            if error > 1e-10 or result["residual"] > 1e-10:
                raise ValueError("权益结果与独立迭代不一致")
            rows.append({"companies": size, "repetition": repetition, "share": share,
                "unknown_share": result["unknown_share"], "elapsed_ms": result["elapsed_ms"],
                "residual": result["residual"], "reference_error": error, "iteration_steps": steps,
                "correct": True})
        progress(f"[证据/权益] {size}家循环企业：与独立迭代一致")
    payload = base_payload()
    payload["relations"] = [{"key": "holdings:C2:C3", "evidence_id": "", "valid_from": "2026-10-01", "valid_to": "2026-10-23"}]
    before = governance_snapshot(payload)
    after = governance_snapshot({**payload, "as_of": "2026-10-23"})
    changes = claim_changes(before, after)
    if ("Controls", "P1", "C3") not in changes["withdrawn"]:
        raise ValueError("有效期失效后控制结论未撤回")
    evidence = {"id": "demo-doc", "title": "教学表决权凭证", "content": "虚构的人工核查文本，仅教学", "sha256": "",
                "status": "verified", "reviewer": "教学核验人", "valid_from": "", "valid_to": ""}
    payload2 = base_payload(); payload2["documents"] = [evidence]
    payload2["rights"] = [{"id": "vote1", "party": "P3", "target": "C3", "kind": "votes", "share": .6,
                          "evidence_id": "demo-doc", "valid_from": "", "valid_to": ""}]
    votes = governance_snapshot(payload2)
    if ("Controls", "P3", "C3") not in votes["claims"] or ("Controls", "P1", "C3") in votes["claims"]:
        raise ValueError("明确表决权未正确替代股权默认规则")
    payload3 = {**payload2, "rights": [{**payload2["rights"][0], "kind": "agreement"},
                                        {**payload2["rights"][0], "id": "agreement2", "kind": "agreement", "party": "P2"}]}
    conflict = governance_snapshot(payload3)
    if not conflict["conflicts"] or conflict["accepted_rights"]:
        raise ValueError("互斥控制协议未阻断")
    closed = EquityCase(entities=[Entity("A", "A", "Company", capital=1000), Entity("B", "B", "Company", capital=1000)],
                        holdings=[Holding("A", "B", 1), Holding("B", "A", 1)])
    indeterminate = economic_equity(closed, "A")
    if indeterminate["status"] != "indeterminate":
        raise ValueError("100%封闭循环不应强行给出唯一经济权益")
    summary = aggregate(rows, ("companies",), ("elapsed_ms", "reference_error", "residual"))
    write_csv(output / "equity_scaling.csv", rows)
    write_csv(output / "equity_summary.csv", summary)
    write_json(output / "temporal_case.json", {"input": payload, "after_input": {**payload, "as_of": "2026-10-23"}, "before": before, "after": after, "changes": changes})
    write_json(output / "rights_cases.json", {"votes_input": payload2, "conflict_input": payload3, "votes": votes, "conflict": conflict})
    write_json(output / "closed_cycle.json", indeterminate)
    lines = ["# 第一次改进项的工程验证", "", "全部为教学数据，不是身份、协议或企业风险的外部认证。", "",
             "| 循环企业数 | 求解平均ms | 与独立迭代最大误差 | 方程残差 |", "|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['companies']} | {row['elapsed_ms_mean']:.4f} | {row['reference_error_mean']:.2e} | {row['residual_mean']:.2e} |")
    lines += ["", "关系在2026-10-23失效后，P1控制C3从新快照撤回；旧快照保留。", "明确60%表决权可支持P3控制C3，经济权益不因此改写。",
              "互斥协议被阻断；100%封闭持股环返回不可唯一求解。", "求解范围为目标上游子图；不支持任意法律协议解释，也不保证现实经济模型有效。", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = execution_manifest(ROOT)
    manifest.update({"sizes": sizes, "repeats": repeats, "reference": "independent_fixed_point_iteration",
                     "temporal_semantics": "start inclusive, end exclusive; rebuild and report delta"})
    seal_manifest(output, manifest)
    return {"rows": rows, "summary": summary, "temporal_withdrawal_checked": True,
            "verified_votes_checked": True, "conflicts_blocked": True, "singular_rejected": True}


def make_governance_figure(source: Path, output: Path):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (source / "equity_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar([int(r["companies"]) for r in rows], [float(r["elapsed_ms_mean"]) for r in rows],
                yerr=[float(r["elapsed_ms_sd"]) for r in rows], marker="o", capsize=3, color="#355C7D")
    ax.set(xlabel="Companies in upstream ownership cycle", ylabel="Linear solve time (ms)",
           title="Cyclic economic ownership: equal-result scaling")
    ax.grid(alpha=.2)
    fig.text(.02, .015, "Known synthetic ring: compare with independent convergent iteration. Singular closed cycles reported separately.", fontsize=8)
    fig.tight_layout(rect=(0,.07,1,1))
    output.mkdir(parents=True, exist_ok=True)
    target = output / "09_cyclic_equity_scaling.png"
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return target
