"""正式结果验收：指纹、划分、指标、汇总、锁定对照与模型回放，标准库实现。

默认每个实验条件回放首条样本；--full-replay回放全部模型预测。不是重新计时。
缺文件、过期代码、被篡改的数据、重复样本、指标错配均失败，不静默跳过。
"""
from __future__ import annotations
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from .artifacts import execution_manifest, fingerprint
from .metrics import evaluate
from .end_to_end import decision_metrics, _method, E2E_MODELS
from .experiments import (MODEL_NAMES, DS_METHODS, SCENARIOS, ABLATION_SPECS,
                          build_ablation_models, ablation_records, ablation_summary_fields)
from experiments.lab02.uncertain.dempster_shafer import Mass, fuse
from core.kb.features import RiskObservation, FEATURES
from core.kb.screening import ScreeningRules
from core.kb.graph_io import case_from_dict
from experiments.lab02.hybrid.model import HybridModel
from experiments.lab02.hybrid.pipeline import prepare_case, assess_company
from experiments.lab02.uncertain.bayes import RiskBayes
from experiments.lab02.uncertain.fuzzy import MamdaniFuzzy
from experiments.lab02.hybrid.calibration import ReviewPolicy


class VerificationError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise VerificationError(message)


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(bool(rows), f"结果表为空: {path}")
    return rows


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _groups(rows, keys):
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row)
    return out


def _compare(actual, expected, label):
    count = 0
    for key, value in actual.items():
        _require(key in expected, f"{label}: 缺少结果字段 {key}")
        raw = expected[key]
        if value is None:
            _require(raw in (None, "", "None"), f"{label}/{key}: 缺失值不一致")
        else:
            _require(math.isclose(float(value), float(raw), rel_tol=1e-9, abs_tol=1e-9),
                     f"{label}/{key}: 重算={value}, 保存={raw}")
        count += 1
    return count


def _unique(rows, keys, label):
    signatures = [tuple(r[k] for k in keys) for r in rows]
    _require(len(signatures) == len(set(signatures)), f"重复记录: {label}")


def _verify_governance(output):
    from core.kb.equity import economic_equity
    from core.kb.governance import governance_snapshot, claim_changes
    from .governance_experiments import ring_case, iterate
    manifest = _json(output / "governance/manifest.json")
    rows = _csv(output / "governance/equity_scaling.csv")
    _unique(rows, ("companies", "repetition"), "循环权益重复实验")
    expected = {(str(n), str(r)) for n in manifest["sizes"] for r in range(manifest["repeats"])}
    _require({(r["companies"], r["repetition"]) for r in rows} == expected, "循环权益规模或重复次数不完整")
    summaries = _csv(output / "governance/equity_summary.csv")
    _unique(summaries, ("companies",), "循环权益汇总")
    _require({r["companies"] for r in summaries} == {str(n) for n in manifest["sizes"]}, "循环权益汇总规模不完整")
    checks, replayed = 0, 0
    for size in manifest["sizes"]:
        case = ring_case(size)
        reference, steps = iterate(case)
        actual = economic_equity(case, "C0")
        _require(actual["status"] == "solved", "循环权益重放未求解")
        owner_share = next(o["share"] for o in actual["owners"] if o["entity_id"] == "P")
        group = [r for r in rows if r["companies"] == str(size)]
        for row in group:
            _require(row["correct"] == "True" and float(row["elapsed_ms"]) >= 0
                     and float(row["residual"]) < 1e-10, "循环权益数值核对失败")
            checks += _compare({"share": owner_share, "unknown_share": actual["unknown_share"],
                "reference_error": abs(owner_share - reference["C0"]), "iteration_steps": steps}, row, "循环权益重放")
        summary = next(r for r in summaries if r["companies"] == str(size))
        checks += _compare({"repetitions": len(group)}, summary, "循环权益汇总")
        for metric in ("elapsed_ms", "reference_error", "residual"):
            values = [float(r[metric]) for r in group]
            checks += _compare({metric + "_mean": statistics.mean(values),
                metric + "_sd": statistics.stdev(values) if len(values) > 1 else 0.0}, summary, "循环权益汇总")
        replayed += 1
    temporal = _json(output / "governance/temporal_case.json")
    before, after = (governance_snapshot(temporal[k]) for k in ("input", "after_input"))
    actual_delta = claim_changes(before, after)
    for key, value in (("before", before), ("after", after)):
        _require(value["claims"] == [tuple(f) for f in temporal[key]["claims"]], "日期快照结论重放不一致")
        _require(value["excluded_relations"] == temporal[key]["excluded_relations"], "日期快照失效关系重放不一致")
    _require(actual_delta["withdrawn"] == [tuple(f) for f in temporal["changes"]["withdrawn"]]
             and ("Controls", "P1", "C3") in actual_delta["withdrawn"], "失效关系撤回重放不一致")
    rights = _json(output / "governance/rights_cases.json")
    for scenario in ("votes", "conflict"):
        current = governance_snapshot(rights[scenario + "_input"])
        _require(current["claims"] == [tuple(f) for f in rights[scenario]["claims"]], "权利结论重放不一致")
        _require(current["conflicts"] == rights[scenario]["conflicts"]
                 and current["accepted_rights"] == rights[scenario]["accepted_rights"], "权利冲突重放不一致")
    _require(["Controls", "P3", "C3"] in rights["votes"]["claims"] and rights["conflict"]["conflicts"]
             and not rights["conflict"]["accepted_rights"], "有效权利与冲突核对失败")
    _require(_json(output / "governance/closed_cycle.json")["status"] == "indeterminate", "封闭环无解处理失败")
    return checks + 7, replayed + 4


def verify_results(output: Path, root: Path, *, full_replay=False) -> dict:
    output, root = output.resolve(), root.resolve()
    current = execution_manifest(root)
    checks = 0
    for folder in ("demo", "experiments", "end_to_end", "benchmarks", "governance"):
        base = output / folder
        manifest = _json(base / "manifest.json")
        _require(manifest["source_sha256"] == current["source_sha256"], f"{folder}: 实验源码版本已过期，需重跑")
        _require(manifest["snapshot_sha256"] == current["snapshot_sha256"], f"{folder}: 原始数据已变化")
        artifacts = manifest.get("artifact_sha256")
        _require(bool(artifacts), f"{folder}: 缺少结果封存指纹，需重跑")
        for relative, digest in artifacts.items():
            p = (base / relative).resolve()
            _require(p.is_relative_to(base.resolve()) and p.is_file(), f"非法或缺失结果路径: {relative}")
            _require(fingerprint(p) == digest, f"结果文件摘要不一致: {folder}/{relative}")
            checks += 1
    indicator_manifest = _json(output / "experiments/manifest.json")
    graph_settings = _json(output / "end_to_end/config.json")
    indicator_config = indicator_manifest["config"]
    graph_config = graph_settings["config"]
    governance_checks, governance_replayed = _verify_governance(output)
    checks += governance_checks
    predictions = _csv(output / "experiments/predictions.csv")
    _unique(predictions, ("seed", "scenario", "sample_id"), "指标预测")
    groups = _groups(predictions, ("seed", "scenario"))
    _require(set(groups) == {(str(s), condition[0]) for s in indicator_config["seeds"] for condition in SCENARIOS},
             "指标实验种子或条件不完整")
    _require(all(len(g) == indicator_config["test_size"] for g in groups.values()), "指标实验样本数不符")
    indicator_metrics = _csv(output / "experiments/metrics.csv")
    _unique(indicator_metrics, ("seed", "scenario", "model"), "指标结果")
    _require(len(indicator_metrics) == len(groups) * len(MODEL_NAMES), "指标方法或结果行缺失")
    for row in indicator_metrics:
        group = groups[(row["seed"], row["scenario"])]
        checks += _compare(evaluate([int(r["label"]) for r in group],
            [float(r[row["model"]]) for r in group]), row, "指标/" + row["model"])
    conflict = _csv(output / "experiments/conflict_predictions.csv")
    _unique(conflict, ("seed", "conflict_rate", "sample_id"), "D-S预测")
    cg = _groups(conflict, ("seed", "conflict_rate"))
    _require(set(cg) == {(str(s), str(rate)) for s in indicator_config["seeds"]
             for rate in (0.0, .2, .4, .6, .8, 1.0)}, "D-S条件不完整")
    _require(all(len(g) == indicator_config["test_size"] for g in cg.values()), "D-S样本数不符")
    conflict_replayed = 0
    seen_conflict = set()
    for row in conflict:
        key = (row["seed"], row["conflict_rate"])
        if not full_replay and key in seen_conflict:
            continue
        seen_conflict.add(key)
        sources = [Mass.from_score(float(row["source_a"])), Mass.from_score(float(row["source_b"]))]
        for name, kwargs in DS_METHODS.items():
            checks += _compare({name: fuse(sources, **kwargs).mass.pignistic}, row, "D-S来源回放")
            conflict_replayed += 1
    for row in _csv(output / "experiments/conflict_metrics.csv"):
        g = cg[(row["seed"], row["conflict_rate"])]
        checks += _compare(evaluate([int(r["label"]) for r in g], [float(r[row["model"]]) for r in g]), row, "D-S")
    graph = _csv(output / "end_to_end/predictions.csv")
    _unique(graph, ("seed", "scenario", "sample_id", "model"), "图级预测")
    gg = _groups(graph, ("seed", "scenario", "model"))
    _require(set(gg) == {(str(s), c["name"], m) for s in graph_config["seeds"]
             for c in graph_settings["conditions"] for m in E2E_MODELS}, "图级种子/条件/方法不完整")
    _require(all(len(g) == graph_config["test_size"] for g in gg.values()), "图级样本数不符")
    for row in _csv(output / "end_to_end/metrics.csv"):
        g = gg[tuple(row[k] for k in ("seed", "scenario", "model"))]
        checks += _compare(decision_metrics(g), row, "图级/" + row["model"])
    paired_count = 0
    for key, group in _groups(graph, ("seed", "scenario", "sample_id")).items():
        by = {r["model"]: r for r in group}
        _require(set(by) == set(E2E_MODELS), f"缺少图级方法: {key}")
        for family in ("fixed", "tuned", "adaptive"):
            a, b = by[f"hybrid_{family}_unlocked"], by[f"hybrid_{family}_locked"]
            _require(a["score"] == b["score"], "硬锁定改写评分")
            _require(int(b["review_required"]) >= int(a["review_required"]), "硬锁定降低行动")
            _require(int(b["constraint_violation"]) == 0, "硬锁定结构违反")
            paired_count += 1
        _require(by["hybrid_adaptive_calibrated"]["score"] == by["hybrid_adaptive_unlocked"]["score"], "门控改写评分")
    for raw_path, summary_path, keys in (
        ("experiments/metrics.csv", "experiments/summary.csv", ("scenario", "model")),
        ("experiments/conflict_metrics.csv", "experiments/conflict_summary.csv", ("conflict_rate", "model")),
        ("end_to_end/metrics.csv", "end_to_end/summary.csv", ("scenario", "model")),
        ("end_to_end/paired_effects.csv", "end_to_end/paired_summary.csv", ("scenario", "family")),
        ("benchmarks/forward.csv", "benchmarks/forward_summary.csv", ("groups", "model"))):
        raw = _groups(_csv(output / raw_path), keys)
        for row in _csv(output / summary_path):
            g = raw[tuple(row[k] for k in keys)]
            actual = {"repetitions": len(g)}
            for key in row:
                if key.endswith("_mean") or key.endswith("_sd"):
                    stem = key[:-5] if key.endswith("_mean") else key[:-3]
                    values = [float(r[stem]) for r in g if r[stem] not in ("", "None")]
                    actual[key] = (statistics.mean(values) if key.endswith("_mean") else
                        statistics.stdev(values) if len(values) > 1 else 0.0) if values else None
            checks += _compare(actual, row, summary_path)
    forward = _csv(output / "benchmarks/forward.csv")
    for key, g in _groups(forward, ("groups", "repetition")).items():
        _require(len({r["closure_sha256"] for r in g}) == 1
                 and all(r["closure_equal"] == "True" and r["converged"] == "True" for r in g), f"闭包不一致: {key}")
        _require({r["model"] for r in g} == {"naive_scan", "indexed_naive", "semi_naive"}, "缺少优化基线")
    _require(all(r["target_equal"] == "True" for r in _csv(output / "benchmarks/goal_slicing.csv")), "目标切片改变答案")
    _require(all(r["status"] == "proved" for r in _csv(output / "benchmarks/resolution.csv")), "归结基准未完成证明")
    samples = _csv(output / "experiments/samples.csv")
    for seed, group in _groups(samples, ("seed",)).items():
        sets = {s: {r["sample_id"] for r in group if r["split"] == s} for s in ("train", "validation", "test")}
        _require(all(sets.values()), f"指标缺少划分: {seed}")
        _require(not (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
                      or sets["validation"] & sets["test"]), f"指标划分泄漏: {seed}")
    _unique(samples, ("seed", "split", "scenario", "sample_id"), "指标样本")
    refs = [json.loads(line) for line in (output / "end_to_end/reference_graphs.jsonl").read_text(encoding="utf-8").splitlines()]
    _unique(refs, ("seed", "split", "sample_id"), "图级样本")
    graph_validation_ids = {}
    for seed, group in _groups(refs, ("seed",)).items():
        sets = {s: {r["sample_id"] for r in group if r["split"] == s} for s in ("train", "validation", "test")}
        _require(all(sets.values()) and not (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
                      or sets["validation"] & sets["test"]), f"图级划分不完整或泄漏: {seed}")
        graph_validation_ids[str(seed[0])] = sets["validation"]
        entity_sets = {s: {e["eid"] for r in group if r["split"] == s for e in r["case"]["entities"]}
                       for s in ("train", "validation", "test")}
        _require(not (entity_sets["train"] & entity_sets["validation"] or entity_sets["train"] & entity_sets["test"]
                      or entity_sets["validation"] & entity_sets["test"]), "图级实体ID划分泄漏")
    lookup = {(r["seed"], r["scenario"], r["sample_id"]): r for r in samples if r["split"] == "test"}
    bundles = _json(output / "experiments/models.json")["models_by_seed"]
    models = {}
    recorded_ablations = _csv(output / "experiments/ablations.csv")
    _unique(recorded_ablations, ("seed", "model"), "消融记录")
    _require({(r["seed"], r["model"]) for r in recorded_ablations}
             == {(seed, name) for seed in bundles for name in ABLATION_SPECS}, "消融记录不完整")
    expected_ablations = []
    for seed, bundle in bundles.items():
        tuned = HybridModel.from_dict(bundle["model"])
        adaptive = HybridModel.from_dict(bundle["adaptive_model"])
        ablated = build_ablation_models(tuned, adaptive)
        _require(bundle.get("ablations") == {name: model.ablation for name, model in ablated.items()},
                 "消融操作说明与模型权重不一致")
        expected_ablations.extend(ablation_records(seed, ablated))
        models[seed] = {"rules": ScreeningRules(), "bayes_manual": RiskBayes(), "bayes_learned": tuned.bayes,
            "fuzzy": MamdaniFuzzy(), "hybrid_fixed": HybridModel.from_dict(bundle["fixed_model"]),
            "hybrid_tuned": tuned,
            "fuzzy_learned": adaptive.fuzzy, "hybrid_adaptive": adaptive,
            **ablated}
        _require(set(models[seed]) == set(MODEL_NAMES), "验收模型集合与实验不匹配")
    ablation_lookup = {(r["seed"], r["model"]): r for r in recorded_ablations}
    for expected in expected_ablations:
        actual = ablation_lookup[(str(expected["seed"]), expected["model"])]
        _require(all(actual.get(k) == str(v) for k, v in expected.items()), "消融明细与实际操作不一致")
        checks += len(expected)
    for row in _csv(output / "experiments/summary.csv"):
        fields = ablation_summary_fields(row["model"], expected_ablations)
        _require(all(row.get(k) == str(v) for k, v in fields.items()), "消融汇总回退标记不一致")
        checks += len(fields)
    replayed = 0
    seen = set()
    for row in predictions:
        key = (row["seed"], row["scenario"])
        sample = lookup[(row["seed"], row["scenario"], row["sample_id"])]
        _require(sample["label"] == row["label"], "保存标签不一致")
        if not full_replay and key in seen:
            continue
        seen.add(key)
        obs = RiskObservation(row["sample_id"], {f: float(sample[f]) if sample[f] else None for f in FEATURES})
        for name, model in models[row["seed"]].items():
            checks += _compare({name: model.predict(obs)}, row, "模型回放")
            replayed += 1
    gb = _json(output / "end_to_end/models.json")["models_by_seed"]
    pv = _groups(_csv(output / "end_to_end/policy_validation.csv"), ("seed",))
    for seed, bundle in gb.items():
        policy = ReviewPolicy.from_dict(bundle["adaptive_model"]["review_policy"])
        _require(all(r["split"] == "validation" and r["sample_id"] in graph_validation_ids[seed]
                 for r in pv[(seed,)]), "门控训练读取非验证集样本")
        _require(len(pv[(seed,)]) == graph_config["validation_size"] * len(bundle["policy_fit_conditions"]),
                 "门控验证样本数不符")
        rows = [{"score": float(r["score"]), "observed_required": bool(int(r["observed_required"])),
                 "label": int(r["label"])} for r in pv[(seed,)]]
        refit = ReviewPolicy(false_negative_cost=policy.false_negative_cost).fit(rows,
            threshold=_json(output / "end_to_end/config.json")["config"]["threshold"])
        _require(policy.as_dict() == refit.as_dict(), "保存的门控与验证选择不一致")
        checks += len(rows)
    by_sample = _groups(graph, ("seed", "scenario", "sample_id"))
    seen = set()
    graph_replayed = 0
    for line in (output / "end_to_end/observed_graphs.jsonl").read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        _require("label" not in raw and "oracle_alerts" not in raw, "推理输入含真值标签")
        key = (str(raw["seed"]), raw["scenario"])
        if not full_replay and key in seen:
            continue
        seen.add(key)
        context = prepare_case(case_from_dict(raw["case"]), completeness=raw["completeness"])
        for row in by_sample[(str(raw["seed"]), raw["scenario"], raw["sample_id"])]:
            name = row["model"]
            bundle_key = "adaptive_model" if name.startswith("hybrid_adaptive") or name == "fuzzy_learned" else (
                "model" if name.startswith("hybrid_tuned") else "fixed_model")
            model = HybridModel.from_dict(gb[str(raw["seed"])][bundle_key])
            threshold = _json(output / "end_to_end/config.json")["config"]["threshold"]
            result = assess_company(context, raw["target_id"], model, method=_method(name), threshold=threshold)
            checks += _compare({k: result[k] for k in ("score", "risk_class", "review_required",
                "lock_changed_action", "constraint_violation", "priority_floor_violation")}, row, "图级回放")
            graph_replayed += 1
    return {"successful": True, "checked_values": checks, "indicator_predictions": len(predictions),
            "graph_predictions": len(graph), "hard_lock_pairs": paired_count,
            "indicator_scores_replayed": replayed, "DS_scores_replayed": conflict_replayed, "graph_decisions_replayed": graph_replayed,
            "governance_cases_replayed": governance_replayed,
            "full_replay": full_replay, "source_and_artifact_hashes_match": True,
            "note": "合成实验验收，不是现实风险能力认证；耗时未重计，正式报告/PPT未生成"}
