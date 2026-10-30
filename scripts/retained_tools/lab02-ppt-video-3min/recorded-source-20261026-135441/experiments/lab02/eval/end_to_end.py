"""完整股权图链路对照：数据源扰动、规则推理、评分、锁定/不锁定行动。

风险标签与结构真值分开；模型只接收观测图。最终分类指标使用 review_required，
评分损失仍使用未经锁定篡改的 score。公共图推理耗时和决策耗时分列。
"""
from __future__ import annotations
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from core.kb.graph_io import case_to_dict
from experiments.lab02.uncertain.bayes import RiskBayes
from experiments.lab02.uncertain.fuzzy import MamdaniFuzzy
from experiments.lab02.hybrid.calibration import ReviewPolicy
from experiments.lab02.hybrid.model import HybridModel
from experiments.lab02.hybrid.pipeline import prepare_case, assess_company
from .graph_data import (GRAPH_GENERATOR, GRAPH_CONDITIONS, generate_graphs,
                         assert_graph_splits_disjoint, perturb_graph)
from .metrics import evaluate, roc_auc
from .artifacts import aggregate, execution_manifest, write_csv, write_json, seal_manifest

ROOT = Path(__file__).resolve().parents[3]
E2E_MODELS = ("rules_only", "bayes_only", "fuzzy_only", "hybrid_fixed_unlocked",
              "hybrid_fixed_locked", "hybrid_tuned_unlocked", "hybrid_tuned_locked",
              "fuzzy_learned", "hybrid_adaptive_unlocked", "hybrid_adaptive_locked",
              "hybrid_adaptive_calibrated")
AGG_METRICS = ("accuracy", "precision", "recall", "f1", "macro_f1", "false_positive_rate",
               "score_brier", "constraint_violation_rate", "observed_alert_retention",
               "true_alert_retention", "unsupported_forced_rate", "priority_floor_violation_rate", "decision_ms", "end_to_end_ms", "decision_cost")


@dataclass(frozen=True)
class EndToEndConfig:
    seeds: tuple[int, ...] = (7, 19, 43)
    train_size: int = 200
    validation_size: int = 80
    test_size: int = 120
    threshold: float = .5

    @classmethod
    def smoke(cls):
        return cls((7,), 20, 8, 10)

    def validate(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("端到端种子不能为空或重复")
        if min(self.train_size, self.validation_size, self.test_size) < 1 or not 0 < self.threshold < 1:
            raise ValueError("样本数须为正数，决策阈值须在(0,1)")


def _ratio(a, b):
    return a / b if b else None


def decision_metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("不能对空预测表计算指标")
    y = [int(r["label"]) for r in rows]
    actions = [int(r["review_required"]) for r in rows]
    scores = [float(r["score"]) for r in rows]
    base = evaluate(y, [float(a) for a in actions])
    # 不把强制行动1/0当成校准概率，从分类结果中剔除其Brier/log-loss/AUC。
    result = {k: v for k, v in base.items() if k not in ("brier", "log_loss", "auc")}
    observed = sum(int(r["observed_required"]) for r in rows)
    truth = sum(int(r["oracle_required"]) for r in rows)
    violations = sum(int(r["observed_required"]) and not int(r["review_required"]) for r in rows)
    unsupported = sum(int(r["lock_changed_action"]) and not int(r["oracle_required"]) for r in rows)
    result.update({
        "false_positive_rate": _ratio(base["fp"], base["fp"] + base["tn"]),
        "decision_cost": (base["fp"] + 2 * base["fn"]) / len(rows),
        "score_brier": sum((s - label) ** 2 for s, label in zip(scores, y)) / len(y),
        "score_auc": roc_auc(y, scores), "observed_alert_cases": observed,
        "oracle_alert_cases": truth, "constraint_violations": violations,
        "constraint_violation_rate": _ratio(violations, observed),
        "observed_alert_retention": _ratio(sum(int(r["observed_required"]) and int(r["review_required"]) for r in rows), observed),
        "true_alert_retention": _ratio(sum(int(r["oracle_required"]) and int(r["review_required"]) for r in rows), truth),
        "unsupported_forced_actions": unsupported, "unsupported_forced_rate": unsupported / len(rows),
        "priority_floor_violation_rate": _ratio(sum(int(r["priority_floor_violation"]) for r in rows),
                                                sum(int(r["has_structural_floor"]) for r in rows)),
        "lock_changed_actions": sum(int(r["lock_changed_action"]) for r in rows),
        "graph_preparation_ms": sum(float(r["graph_preparation_ms"]) for r in rows) / len(rows),
        "decision_ms": sum(float(r["decision_ms"]) for r in rows) / len(rows),
        "end_to_end_ms": sum(float(r["end_to_end_ms"]) for r in rows) / len(rows),
    })
    return result


def _emit(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def _reference(sample, seed, split):
    return {"sample_id": sample.sample_id, "seed": seed, "split": split, "target_id": sample.target_id,
            "label": sample.label, "oracle_alerts": sample.oracle_alerts, "case": case_to_dict(sample.case)}


def _method(model_name):
    if model_name == "fuzzy_learned":
        return "fuzzy_only"
    if model_name.endswith("_calibrated"):
        return "hybrid_calibrated"
    if model_name.startswith("hybrid_"):
        return "hybrid_locked" if model_name.endswith("_locked") else "hybrid_unlocked"
    return model_name


def run_end_to_end(output: Path, config: EndToEndConfig | None = None, progress=print) -> dict:
    config = EndToEndConfig() if config is None else config
    config.validate()
    output.mkdir(parents=True, exist_ok=True)
    # 首轮正式计算前固定并落盘配置，后续不根据测试集表现改参数。
    settings = {"config": asdict(config), "generator": GRAPH_GENERATOR,
                "conditions": [asdict(c) for c in GRAPH_CONDITIONS], "methods": E2E_MODELS,
                "label_note": "外部模拟标签先于业务图生成；风险标签不等于是否存在结构预警",
                "decision_note": "分类指标基于最终review_required；Brier仅评价原始score，锁定不修改score"}
    write_json(output / "config.json", settings)
    metrics, predictions, paired, graph_audit, split_audit = [], [], [], [], []
    policy_validation = []
    bundles, examples = {}, {}
    started = time.perf_counter()
    with (output / "reference_graphs.jsonl").open("w", encoding="utf-8", newline="\n") as reference_file, \
         (output / "observed_graphs.jsonl").open("w", encoding="utf-8", newline="\n") as observed_file:
        for seed in config.seeds:
            progress(f"[端到端] seed={seed}；图级训练/验证/测试={config.train_size}/{config.validation_size}/{config.test_size}")
            train, validation, test = [generate_graphs(n, seed, split) for n, split in [
                (config.train_size, "train"), (config.validation_size, "validation"), (config.test_size, "test")]]
            assert_graph_splits_disjoint(train, validation, test)
            split_audit.append({"seed": seed, "train_n": len(train), "validation_n": len(validation),
                                "unique_test_n": len(test), "sample_id_overlap": 0, "entity_id_overlap": 0,
                                "train_positive": sum(s.label for s in train),
                                "validation_positive": sum(s.label for s in validation),
                                "test_positive": sum(s.label for s in test)})
            for split, samples in [("train", train), ("validation", validation), ("test", test)]:
                for sample in samples:
                    _emit(reference_file, _reference(sample, seed, split))
            train_contexts = [prepare_case(s.case) for s in train]
            val_contexts = [prepare_case(s.case) for s in validation]
            train_obs = [c.observations[s.target_id] for c, s in zip(train_contexts, train)]
            val_obs = [c.observations[s.target_id] for c, s in zip(val_contexts, validation)]
            train_labels, val_labels = [s.label for s in train], [s.label for s in validation]
            learned = RiskBayes.fit(train_obs, train_labels)
            fixed = HybridModel(bayes=learned)
            tuned = HybridModel(bayes=learned).fit_weights(val_obs, val_labels)
            fitted_fuzzy = MamdaniFuzzy.fit(train_obs, train_labels)
            adaptive = HybridModel(bayes=learned, fuzzy=fitted_fuzzy).fit_weights(val_obs, val_labels)
            # 门控只看预先固定的验证扰动；不读取测试图、oracle或扰动审计标志。
            calibration_rows = []
            calibration_audit = []
            for condition in GRAPH_CONDITIONS[:4]:
                for index, sample in enumerate(validation):
                    observed = perturb_graph(sample.case, sample.target_id, condition,
                                             seed * 100003 + index * 59 + 811)
                    context = prepare_case(observed.case, completeness=observed.completeness)
                    d = assess_company(context, sample.target_id, adaptive, method="hybrid_unlocked",
                                       threshold=config.threshold)
                    calibration_audit.append({"seed": seed, "split": "validation", "scenario": condition.name,
                        "sample_id": sample.sample_id, "score": d["score"],
                        "observed_required": int(d["structural_alert_requires_review"]), "label": sample.label})
                    calibration_rows.append({"score": d["score"],
                        "observed_required": d["structural_alert_requires_review"], "label": sample.label})
            adaptive.review_policy = ReviewPolicy().fit(calibration_rows, threshold=config.threshold)
            policy_validation.extend(calibration_audit)
            frozen = tuned.as_dict()
            bundles[str(seed)] = {"model": frozen, "fixed_model": fixed.as_dict(),
                                  "adaptive_model": adaptive.as_dict(), "fuzzy_model": fitted_fuzzy.as_dict(),
                                  "policy_fit_conditions": [c.name for c in GRAPH_CONDITIONS[:4]],
                                  "policy_fit_scope": "验证图固定扰动，FP+2FN；不读取测试标签或oracle",
                                  "fit_scope": "完整训练图派生的指标；权重仅用验证图，不查看压力测试标签"}
            del train_contexts, val_contexts
            for condition in GRAPH_CONDITIONS:
                grouped = {m: [] for m in E2E_MODELS}
                for index, sample in enumerate(test):
                    observed = perturb_graph(sample.case, sample.target_id, condition, seed * 100003 + index * 59 + 17)
                    context = prepare_case(observed.case, completeness=observed.completeness)
                    # 本调用仅含观测图和完整性标志；干净图/标签只在下面的评估代码中使用。
                    observed_alerts = sorted(f[2] for f in context.established_for(sample.target_id) if f[0] == "Alert")
                    if condition.name == "clean" and observed_alerts != list(sample.oracle_alerts):
                        raise AssertionError("干净图的规则预警与生成结构真值不一致")
                    _emit(observed_file, {"seed": seed, "scenario": condition.name, "sample_id": sample.sample_id,
                        "target_id": sample.target_id, "case": case_to_dict(observed.case),
                        "completeness": observed.completeness, "corruption_stats": observed.corruption_stats})
                    graph_audit.append({"seed": seed, "scenario": condition.name, "sample_id": sample.sample_id,
                        "label": sample.label, "oracle_required": int(bool(sample.oracle_alerts)),
                        "observed_required": int(bool(observed_alerts)),
                        "oracle_alerts": ";".join(sample.oracle_alerts), "observed_alerts": ";".join(observed_alerts),
                        "false_structural_alert": int(bool(observed_alerts) and not sample.oracle_alerts),
                        "lost_structural_alert": int(bool(sample.oracle_alerts) and not observed_alerts),
                        "removed_edges": sum(observed.corruption_stats["removed"].values()),
                        "injected_returns": observed.corruption_stats["injected_guarantee_returns"],
                        "feature_coverage": context.observations[sample.target_id].coverage,
                        "preparation_ms": context.preparation_ms, "closure_facts": len(context.engine.wm.facts)})
                    decisions = {}
                    for name in E2E_MODELS:
                        model = (adaptive if name.startswith("hybrid_adaptive") or name == "fuzzy_learned"
                                 else tuned if name.startswith("hybrid_tuned") else fixed)
                        d = assess_company(context, sample.target_id, model, method=_method(name), threshold=config.threshold)
                        decisions[name] = d
                        row = {"seed": seed, "scenario": condition.name, "sample_id": sample.sample_id,
                               "target_id": sample.target_id, "model": name, "label": sample.label,
                               "oracle_required": int(bool(sample.oracle_alerts)),
                               "observed_required": int(d["structural_alert_requires_review"]),
                               "score": d["score"], "risk_class": d["risk_class"],
                               "review_required": int(d["review_required"]), "review_priority": d["review_priority"],
                               "lock_changed_action": int(d["lock_changed_action"]),
                               "constraint_violation": int(d["constraint_violation"]),
                               "priority_floor_violation": int(d["priority_floor_violation"]),
                               "has_structural_floor": int(d["rule_priority_floor"] != "低"),
                               "feature_coverage": d["coverage"], "graph_preparation_ms": context.preparation_ms,
                               "decision_ms": d["decision_ms"], "end_to_end_ms": d["end_to_end_ms"]}
                        grouped[name].append(row)
                        predictions.append(row)
                    for family in ("fixed", "tuned", "adaptive"):
                        a, b = decisions[f"hybrid_{family}_unlocked"], decisions[f"hybrid_{family}_locked"]
                        if a["score"] != b["score"] or b["review_required"] < a["review_required"]:
                            raise AssertionError("锁定只能增加必要行动，不能篡改原始评分或降低行动")
                        if b["constraint_violation"]:
                            raise AssertionError("锁定模式违反了已知结构约束")
                    if decisions["hybrid_adaptive_calibrated"]["score"] != decisions["hybrid_adaptive_unlocked"]["score"]:
                        raise AssertionError("门控不能篡改原始评分")
                    lock = decisions["hybrid_fixed_locked"]
                    category = None
                    if lock["lock_changed_action"] and sample.label and sample.oracle_alerts:
                        category = "retained_true_positive"
                    elif lock["lock_changed_action"] and not sample.label:
                        category = "extra_false_positive"
                    elif not lock["review_required"] and sample.label:
                        category = "missed_positive"
                    if category and category not in examples:
                        examples[category] = {"seed": seed, "scenario": condition.name, "sample_id": sample.sample_id,
                            "label": sample.label, "oracle_alerts": sample.oracle_alerts,
                            "request": {"case": case_to_dict(observed.case), "target_id": sample.target_id,
                                        "completeness": observed.completeness},
                            "unlocked": decisions["hybrid_fixed_unlocked"],
                            "locked": assess_company(context, sample.target_id, fixed, method="hybrid_locked",
                                                      threshold=config.threshold, include_explanation=True)}
                condition_metrics = {}
                for name in E2E_MODELS:
                    value = decision_metrics(grouped[name])
                    condition_metrics[name] = value
                    metrics.append({"seed": seed, "scenario": condition.name, "model": name, **value})
                for family in ("fixed", "tuned", "adaptive"):
                    a, b = condition_metrics[f"hybrid_{family}_unlocked"], condition_metrics[f"hybrid_{family}_locked"]
                    paired.append({"seed": seed, "scenario": condition.name, "family": family,
                        "delta_tp": b["tp"] - a["tp"], "delta_fp": b["fp"] - a["fp"],
                        "delta_fn": b["fn"] - a["fn"], "delta_accuracy": b["accuracy"] - a["accuracy"],
                        "delta_f1": b["f1"] - a["f1"], "delta_recall": b["recall"] - a["recall"],
                        "delta_score_brier": b["score_brier"] - a["score_brier"],
                        "removed_constraint_violations": a["constraint_violations"] - b["constraint_violations"],
                        "action_changes": b["lock_changed_actions"],
                        "unsupported_forced_actions": b["unsupported_forced_actions"]})
            if (tuned.as_dict() != frozen
                    or adaptive.as_dict() != bundles[str(seed)]["adaptive_model"]
                    or fitted_fuzzy.as_dict() != bundles[str(seed)]["fuzzy_model"]):
                raise AssertionError("测试期间模型参数被更改")
    summary = aggregate(metrics, ("scenario", "model"), AGG_METRICS)
    paired_summary = aggregate(paired, ("scenario", "family"),
        ("delta_tp", "delta_fp", "delta_fn", "delta_accuracy", "delta_f1", "delta_recall",
         "delta_score_brier", "removed_constraint_violations", "action_changes", "unsupported_forced_actions"))
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "paired_effects.csv", paired)
    write_csv(output / "paired_summary.csv", paired_summary)
    write_csv(output / "graph_audit.csv", graph_audit)
    write_csv(output / "policy_validation.csv", policy_validation)
    write_json(output / "split_audit.json", split_audit)
    write_json(output / "models.json", {"version": 1, "models_by_seed": bundles})
    write_json(output / "examples.json", examples)
    for name, example in examples.items():
        write_json(output / "examples" / (name + "_input.json"), example["request"])
    manifest = execution_manifest(ROOT)
    manifest.update(settings)
    manifest.update({"elapsed_seconds": time.perf_counter() - started,
        "scoring_contract": "使用完整图链路的review_required评价行动，不再只用predict的数值评分代替复核结论",
        "timing_contract": "同图公共预处理只执行一次，preparation_ms对各方法相同；end_to_end_ms是公共阶段与各自决策阶段之和，不含文件I/O/诊断解释",
        "zero_violation_note": "已知事实约束违反率为0是锁定策略的程序不变量，不是风险识别正确率为100%",
        "adverse_results": "必须同时保留额外误报、源错误放大、缺边漏报及固定/验证权重的无收益情形",
        "example_selection": "各类首次出现的真实测试案例，仅用于解释，不用于选择参数或替代总体指标",
        "uncertainty_scope": "规则事实可信度取决于源数据；错误闭环注入时保留已知预警可能增加误报"})
    write_json(output / "manifest.json", manifest)
    write_results(output, summary, paired_summary, bundles)
    seal_manifest(output, manifest)
    progress(f"[端到端] 完成；{len(predictions)} 条方法预测；保存至 {output}")
    return {"metrics": metrics, "summary": summary, "paired": paired,
            "paired_summary": paired_summary, "models": bundles, "examples": examples}


def write_results(output, summary, paired_summary, bundles):
    def number(x):
        return "N/A" if x is None else f"{x:.4f}"
    lines = ["# 完整股权链路实验结果", "", "本文件由真实运行结果生成，不是课程报告。全部标签与业务图均为合成数据。",
        "分类指标使用最终复核行动；评分损失不把被锁定的行动1/0冒充概率。", "",
        "## 清洁观测图", "", "| 方法 | Accuracy | F1 | Recall | FPR | 已知约束违反率 | 原始评分Brier↓ |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        if r["scenario"] == "clean":
            lines.append(f"| {r['model']} | {number(r['accuracy_mean'])} | {number(r['f1_mean'])} | "
                f"{number(r['recall_mean'])} | {number(r['false_positive_rate_mean'])} | "
                f"{number(r['constraint_violation_rate_mean'])} | {number(r['score_brier_mean'])} |")
    lines += ["", "## 锁定相对不锁定：同图配对的固定权重对照", "",
        "| 条件 | ΔTP | ΔFP | ΔRecall | ΔF1 | 去除的约束违反数 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in paired_summary:
        if r["family"] == "fixed":
            lines.append(f"| {r['scenario']} | {r['delta_tp_mean']:.2f} | {r['delta_fp_mean']:.2f} | "
                         f"{r['delta_recall_mean']:.4f} | {r['delta_f1_mean']:.4f} | {r['removed_constraint_violations_mean']:.2f} |")
    lines += ["", "## 自动拟合与校准行动对照", "",
        "模糊参数只读取训练图；融合权重只读取清洁验证图。行动门控仅读取预先固定的四种验证扰动。",
        "代价为(FP+2FN)/N；硬锁定、不锁定、门控共用评分，门控不保证零结构违反或所有指标提升。", "",
        "| 条件 | 方法 | F1 | 误报率 | 代价 | 已知约束违反率 |", "|---|---|---:|---:|---:|---:|"]
    for r in summary:
        if r["model"].startswith("hybrid_adaptive") or r["model"] == "fuzzy_learned":
            lines.append(f"| {r['scenario']} | {r['model']} | {number(r['f1_mean'])} | "
                f"{number(r['false_positive_rate_mean'])} | {number(r['decision_cost_mean'])} | "
                f"{number(r['constraint_violation_rate_mean'])} |")
    lines += ["", "## 验证集权重", ""]
    for seed, bundle in bundles.items():
        weights = bundle["model"]["weights"]
        lines.append(f"- seed {seed}: 原模型{weights}；自动拟合{bundle['adaptive_model']['weights']}；"
                     f"门控最低支持评分={bundle['adaptive_model']['review_policy']['minimum_support']}。")
        if sum(w > 0 for w in weights.values()) == 1:
            lines.append("  评分器退化为单组件；是否仍有结构锁定作用，应看最终行动而非只看score。")
    lines += ["", "## 解读和复现边界", "",
        "- 锁定与不锁定共享同一观测图、同一特征、同一模型参数和同一阈值，原始评分逐样本完全相同。",
        "- 已知约束违反率分母是观测图推出高预警的样本数；为0是策略保证，不是对真实风险的完美预测。",
        "- true_alert_retention 以干净图独立结构真值为分母；缺边造成未证实预警时，锁定也不保证保留它。",
        "- 独立的外部模拟风险标签用于Precision/Recall/F1；存在结构线索的正常样本仍会计入误报，不能删除。",
        "- reference_graphs.jsonl 保存标签和干净图；observed_graphs.jsonl 只保存观测输入及扰动审计信息，模型不读取标签或干净图。",
        "- 源记录不完整时，未能确定的交易/担保比例和未证循环使用None，而不是假装为0。",
        "- 错误闭环注入可以使硬规则保留错误线索。graph_audit.csv 与 unsupported_forced_actions 明确记录这类代价。",
        "- 配对变化按三个种子汇总；没有作统计显著性宣称，也不保证融合在任何分布下优于单模型。",
        "- examples 中展示的是首次出现的解释案例，全部总体预测均保留在 predictions.csv。", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
