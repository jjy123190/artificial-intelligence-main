"""风险推理对照、噪声/缺测压力、参数敏感性、消融及 D-S 冲突实验。"""
from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from core.kb.features import FEATURES
from core.kb.screening import ScreeningRules
from experiments.lab02.uncertain.bayes import RiskBayes
from experiments.lab02.uncertain.fuzzy import MamdaniFuzzy
from experiments.lab02.uncertain.dempster_shafer import Mass, TotalConflict, dempster, fuse
from experiments.lab02.hybrid.model import HybridModel
from .synthetic import GENERATOR, assert_disjoint, generate, perturb
from .metrics import evaluate, reliability_bins
from .artifacts import aggregate, execution_manifest, write_csv, write_json, seal_manifest

LAB_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (7, 19, 43)
    train_size: int = 1000
    validation_size: int = 300
    test_size: int = 600

    @classmethod
    def smoke(cls):
        return cls((7,), 120, 40, 60)


SCENARIOS = (
    ("clean", 0.0, 0.0),
    ("noise_010", .10, 0.0), ("noise_020", .20, 0.0), ("noise_030", .30, 0.0),
    ("missing_010", 0.0, .10), ("missing_030", 0.0, .30),
    ("mixed", .20, .20), ("severe", .35, .40),
)
MODEL_NAMES = ("rules", "bayes_manual", "bayes_learned", "fuzzy", "hybrid_fixed",
               "hybrid_tuned", "hybrid_no_rule", "hybrid_no_fuzzy", "hybrid_no_bayes",
               "fuzzy_learned", "hybrid_adaptive", "adaptive_no_rule", "adaptive_no_fuzzy", "adaptive_no_bayes")
ABLATION_SPECS = {
    "hybrid_no_rule": ("hybrid_tuned", "rule", "bayes"),
    "hybrid_no_fuzzy": ("hybrid_tuned", "fuzzy", "bayes"),
    "hybrid_no_bayes": ("hybrid_tuned", "bayes", "fuzzy"),
    "adaptive_no_rule": ("hybrid_adaptive", "rule", "bayes"),
    "adaptive_no_fuzzy": ("hybrid_adaptive", "fuzzy", "bayes"),
    "adaptive_no_bayes": ("hybrid_adaptive", "bayes", "fuzzy"),
}
DS_METHODS = {
    "ds_classic": {"robust": False},
    "ds_yager": {"robust": True, "conflict_threshold": .6},
    "ds_discount": {"robust": False, "reliabilities": [.95, .55]},
    "ds_robust": {"robust": True, "reliabilities": [.95, .55], "conflict_threshold": .45},
}
SCORE_METRICS = ("accuracy", "precision", "recall", "f1", "macro_f1", "brier", "auc", "inference_us")


def _timed_scores(model, observations):
    start = time.perf_counter_ns()
    scores = [model.predict(o) for o in observations]
    elapsed = (time.perf_counter_ns() - start) / 1000 / len(observations)
    return scores, elapsed


def build_ablation_models(tuned: HybridModel, adaptive: HybridModel) -> dict:
    """在看测试集之前规定回退组件，并保存实际操作；不静默冒充严格消融。"""
    parents = {"hybrid_tuned": tuned, "hybrid_adaptive": adaptive}
    return {name: parents[parent].without(component, fallback=fallback)
            for name, (parent, component, fallback) in ABLATION_SPECS.items()}


def ablation_records(seed, models: dict) -> list[dict]:
    rows = []
    for name, model in models.items():
        info = model.ablation
        rows.append({"seed": seed, "model": name, "base_model": ABLATION_SPECS[name][0],
                     "removed_component": info["removed_component"], "kind": info["kind"],
                     "fallback_if_empty": info["fallback_if_empty"],
                     "fallback_component": info["fallback_component"] or "",
                     **{"before_" + key: value for key, value in info["base_weights"].items()},
                     **{"after_" + key: value for key, value in info["result_weights"].items()}})
    return rows


def ablation_summary_fields(model_name: str, records: list[dict]) -> dict:
    rows = [r for r in records if r["model"] == model_name]
    if not rows:
        return {}
    return {"ablation_base_model": rows[0]["base_model"],
            "ablation_runs": len(rows),
            "ablation_fallback_runs": sum(r["kind"] == "baseline_fallback" for r in rows),
            "ablation_fallback_component": ";".join(sorted({r["fallback_component"] for r in rows if r["fallback_component"]}))}


def run_experiments(output: Path, config: ExperimentConfig | None = None, progress=print) -> dict:
    config = ExperimentConfig() if config is None else config
    if not config.seeds or len(set(config.seeds)) != len(config.seeds) or min(config.train_size, config.validation_size, config.test_size) < 1:
        raise ValueError("种子需非空且不重复，划分样本数须为正数")
    output.mkdir(parents=True, exist_ok=True)
    raw, predictions, sample_rows, sensitivity, calibration = [], [], [], [], []
    bundles, split_audit, ablation_rows = {}, [], []
    for seed in config.seeds:
        progress(f"[实验] seed={seed}: 训练 {config.train_size} / 验证 {config.validation_size} / 测试 {config.test_size}")
        train = generate(config.train_size, seed, "train")
        validation = generate(config.validation_size, seed, "validation")
        test = generate(config.test_size, seed, "test")
        assert_disjoint(train, validation, test)
        train_o, train_y = [s.observation for s in train], [s.label for s in train]
        val_o, val_y = [s.observation for s in validation], [s.label for s in validation]
        started = time.perf_counter()
        learned = RiskBayes.fit(train_o, train_y)
        training_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        fuzzy_learned = MamdaniFuzzy.fit(train_o, train_y)
        fuzzy_training_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        # 参数学习用训练集，组合权重用验证集；测试集只用于冻结后的最终比较。
        adaptive = HybridModel(bayes=learned, fuzzy=fuzzy_learned).fit_weights(val_o, val_y)
        adaptive_validation_ms = (time.perf_counter() - started) * 1000
        fixed = HybridModel(bayes=learned)
        started = time.perf_counter()
        tuned = HybridModel(bayes=learned).fit_weights(val_o, val_y)
        validation_ms = (time.perf_counter() - started) * 1000
        ablated = build_ablation_models(tuned, adaptive)
        ablation_rows.extend(ablation_records(seed, ablated))
        models = {"rules": ScreeningRules(), "bayes_manual": RiskBayes(), "bayes_learned": learned,
                  "fuzzy": MamdaniFuzzy(), "hybrid_fixed": fixed, "hybrid_tuned": tuned,
                  "fuzzy_learned": fuzzy_learned, "hybrid_adaptive": adaptive,
                  **ablated}
        models = {name: models[name] for name in MODEL_NAMES}
        frozen_bundle = tuned.as_dict()
        bundles[str(seed)] = {"model": frozen_bundle, "fixed_model": fixed.as_dict(),
                              "adaptive_model": adaptive.as_dict(), "fuzzy_model": fuzzy_learned.as_dict(),
                               "ablations": {name: model.ablation for name, model in ablated.items()},
                              "training_ms": training_ms, "fuzzy_training_ms": fuzzy_training_ms,
                              "adaptive_validation_ms": adaptive_validation_ms,
                              "validation_ms": validation_ms, "train_seed": seed,
                              "selection_rule": "模糊15组参数仅用训练集；融合15组权重仅用验证集；不按测试结果选择"}
        split_audit.append({"seed": seed, "train_n": len(train), "validation_n": len(validation),
                            "unique_test_n": len(test), "id_intersections": 0,
                            "train_positive": sum(train_y), "validation_positive": sum(val_y),
                            "test_positive": sum(s.label for s in test)})
        for split, samples in [("train", train), ("validation", validation)]:
            sample_rows.extend(_sample_rows(seed, split, "clean", samples))
        for scenario, sigma, missing in SCENARIOS:
            # 同一扰动样本同时交给全部模型，避免不同抽样造成不公平的性能比较。
            samples = perturb(test, sigma=sigma, missing=missing, seed=seed + 9001)
            observations, labels = [s.observation for s in samples], [s.label for s in samples]
            sample_rows.extend(_sample_rows(seed, "test", scenario, samples))
            score_table = {}
            for name in MODEL_NAMES:
                scores, inference_us = _timed_scores(models[name], observations)
                score_table[name] = scores
                row = {"seed": seed, "scenario": scenario, "sigma": sigma, "missing": missing,
                       "model": name, "inference_us": inference_us,
                       "mean_feature_coverage": sum(o.coverage for o in observations) / len(observations),
                       "all_missing_n": sum(o.coverage == 0 for o in observations)}
                row.update(evaluate(labels, scores))
                raw.append(row)
                if scenario == "clean":
                    calibration.extend({"seed": seed, "model": name, **b}
                                       for b in reliability_bins(labels, scores))
            for i, sample in enumerate(samples):
                predictions.append({"seed": seed, "scenario": scenario,
                                    "sample_id": sample.observation.company_id, "label": sample.label,
                                    **{name: score_table[name][i] for name in MODEL_NAMES}})
        # 压力实验不得在线重训或通过测试标签改权重。
        if (tuned.as_dict() != frozen_bundle
                or adaptive.as_dict() != bundles[str(seed)]["adaptive_model"]
                or fuzzy_learned.as_dict() != bundles[str(seed)]["fuzzy_model"]):
            raise AssertionError("测试期间模型参数发生改变")
        clean_o, clean_y = [s.observation for s in test], [s.label for s in test]
        for prior in (.10, .25, .40, .55, .70):
            model = RiskBayes(prior, learned.cpt)
            scores, micros = _timed_scores(model, clean_o)
            sensitivity.append({"seed": seed, "model": "bayes", "parameter": "prior", "value": prior,
                                "inference_us": micros, **evaluate(clean_y, scores)})
        for shift in (-.15, -.075, 0.0, .075, .15):
            cpt = {f: (v[0], min(1 - 1e-6, max(1e-6, v[1] + shift))) for f, v in learned.cpt.items()}
            scores, micros = _timed_scores(RiskBayes(learned.prior, cpt), clean_o)
            sensitivity.append({"seed": seed, "model": "bayes", "parameter": "cpt_positive_shift", "value": shift,
                                "inference_us": micros, **evaluate(clean_y, scores)})
        for start in (.30, .40, .45, .50, .60):
            scores, micros = _timed_scores(MamdaniFuzzy(high_start=start), clean_o)
            sensitivity.append({"seed": seed, "model": "fuzzy", "parameter": "high_start", "value": start,
                                "inference_us": micros, **evaluate(clean_y, scores)})
    summary = aggregate(raw, ("scenario", "model"), SCORE_METRICS)
    for row in summary:
        row.update(ablation_summary_fields(row["model"], ablation_rows))
    write_csv(output / "metrics.csv", raw)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "samples.csv", sample_rows)
    write_csv(output / "sensitivity.csv", sensitivity)
    write_csv(output / "calibration.csv", calibration)
    write_csv(output / "ablations.csv", ablation_rows)
    write_json(output / "models.json", {"version": 2, "models_by_seed": bundles})
    write_json(output / "split_audit.json", split_audit)
    conflict = run_conflict_experiment(output, config, progress)
    manifest = execution_manifest(LAB_ROOT)
    manifest.update({"config": asdict(config), "generator": GENERATOR, "scenarios": SCENARIOS,
                     "models": MODEL_NAMES, "classification_threshold": .5,
                     "mean_sd_unit": "独立随机种子间的均值/样本标准差；各压力场景使用同一批配对测试样本",
                     "timing_scope": "各方法逐样本 predict 全过程；不含训练/验证调参及文件I/O，无人为延时",
                     "scoring_note": "Brier 在此也作为归一化风险评分的平方损失；模糊/规则/融合评分未经概率校准",
                      "ablation_policy": {"strict_default": True, "records": "ablations.csv",
                          "fallback_if_empty": {name: fallback for name, (_, _, fallback) in ABLATION_SPECS.items()},
                          "note": "剩余权重非零时只删除并归一化；全零时显式回退为预先指定的单模型基线，非组件边际贡献"},
                     "limitations": ["全部实验样本与标签均合成，不是来自29实体教学案例的真实标签",
                                     "生成分布接近朴素贝叶斯假设，可能有利于贝叶斯模型",
                                     "指标压力实验不等同于对全部一阶逻辑事实随机加噪",
                                      "移除后剩余权重非零才归一化；全零时的显式回退单列标注，不当作严格消融",
                                     "零权重组件被移除时结果相同是有效实验结果"]})
    write_json(output / "manifest.json", manifest)
    _write_results_note(output, summary, conflict, bundles, config)
    seal_manifest(output, manifest)
    progress(f"[实验] 完成，原始预测与配置已写入 {output}")
    return {"metrics": raw, "summary": summary, "conflict_summary": conflict, "models": bundles}


def _sample_rows(seed, split, scenario, samples):
    return [{"seed": seed, "split": split, "scenario": scenario, "sample_id": s.observation.company_id,
             "label": s.label, **{f: s.observation.values.get(f) for f in FEATURES}} for s in samples]


def run_conflict_experiment(output, config, progress=print):
    raw, predictions = [], []
    methods = DS_METHODS
    for seed in config.seeds:
        samples = generate(config.test_size, seed, "test")
        labels = [s.label for s in samples]
        for rate in (0.0, .2, .4, .6, .8, 1.0):
            rng = random.Random(seed + 424242)
            sources, audit = [], []
            for sample in samples:
                center = .9 if sample.label else .1
                a = min(.98, max(.02, rng.gauss(center, .13)))
                b = min(.98, max(.02, rng.gauss(center, .25)))
                flipped = rng.random() < rate
                b = 1 - b if flipped else b
                sources.append([Mass.from_score(a), Mass.from_score(b)])
                audit.append((a, b, flipped))
            scores = {}
            for name, kwargs in methods.items():
                start = time.perf_counter_ns()
                results = [fuse(source, **kwargs) for source in sources]
                micros = (time.perf_counter_ns() - start) / 1000 / len(results)
                scores[name] = [r.mass.pignistic for r in results]
                raw.append({"seed": seed, "conflict_rate": rate, "model": name,
                            "inference_us": micros, "mean_conflict": sum(r.conflict for r in results) / len(results),
                            "mean_ignorance": sum(r.mass.ignorance for r in results) / len(results),
                            "yager_fraction": sum("yager" in r.method for r in results) / len(results),
                            **evaluate(labels, scores[name])})
            for i, sample in enumerate(samples):
                a, b, flip = audit[i]
                predictions.append({"seed": seed, "conflict_rate": rate,
                    "sample_id": sample.observation.company_id, "label": sample.label,
                    "source_a": a, "source_b": b, "b_injected_flip": int(flip),
                    **{name: scores[name][i] for name in methods}})
    summary = aggregate(raw, ("conflict_rate", "model"),
                        ("accuracy", "f1", "brier", "mean_conflict", "mean_ignorance", "yager_fraction", "inference_us"))
    write_csv(output / "conflict_metrics.csv", raw)
    write_csv(output / "conflict_summary.csv", summary)
    write_csv(output / "conflict_predictions.csv", predictions)
    try:
        dempster(Mass(1, 0, 0), Mass(0, 1, 0))
        raise AssertionError("完全冲突应该抛错")
    except TotalConflict as exc:
        total_conflict = str(exc)
    write_json(output / "conflict_examples.json", {
        "classic_total_conflict": total_conflict,
        "robust_total_conflict": fuse([Mass(1, 0, 0), Mass(0, 1, 0)]).mass.as_dict(),
        "source_order": ["A", "B"], "source_quality_discount": [.95, .55],
        "source_note": "A/B是独立生成的两个模拟测度来源，不是同一指标算出的Bayes/Fuzzy输出",
        "reliability_note": "来源质量系数是预设元数据，所有冲突率相同，不读取每条样本的label或flip标志",
        "flip_note": "flip仅保存用于审计，不作为融合模型输入；压力越大来源B越不可信",
        "order_note": "Yager逐对组合不保证结合律，本实验固定来源顺序"})
    progress("[实验] D-S 来源冲突/可靠性折扣对照已完成")
    return summary


def _write_results_note(output, summary, conflict, bundles, config):
    def method_label(row):
        count = row.get("ablation_fallback_runs", 0)
        return (f"{row['model']}（{count}/{row['ablation_runs']}种子回退为{row['ablation_fallback_component']}基线）"
                if count else row["model"])
    lines = ["# 第二次汇报：本机实测结果", "", "> 全部分类标签均为合成标签，不能解释为真实企业风险准确率。",
             "", f"配置：随机种子 {list(config.seeds)}；每个种子训练 {config.train_size}、验证 {config.validation_size}、测试 {config.test_size}。",
             "各压力场景使用相同测试企业，标签不变；模型在测试前冻结。均值±标准差按种子计算。", "",
             "## 无额外噪声测试集", "", "| 方法 | Accuracy | F1 | Brier↓ | 单样本微秒 |",
             "|---|---:|---:|---:|---:|"]
    for r in summary:
        if r["scenario"] == "clean":
            lines.append(f"| {method_label(r)} | {r['accuracy_mean']:.4f}±{r['accuracy_sd']:.4f} | "
                         f"{r['f1_mean']:.4f}±{r['f1_sd']:.4f} | {r['brier_mean']:.4f} | {r['inference_us_mean']:.1f} |")
    lines += ["", "## 严重噪声与缺测（σ=0.35，缺测率40%）", "", "| 方法 | Accuracy | F1 | Brier↓ |", "|---|---:|---:|---:|"]
    for r in summary:
        if r["scenario"] == "severe":
            lines.append(f"| {method_label(r)} | {r['accuracy_mean']:.4f} | {r['f1_mean']:.4f} | {r['brier_mean']:.4f} |")
    lines += ["", "## 验证集选择的权重（Bayes / Fuzzy / Rule）", ""]
    for seed, data in bundles.items():
        m = data["model"]
        lines.append(f"- seed {seed}: {m['weights']}；验证集 Brier={m['validation_brier']:.6f}。")
        if sum(w > 0 for w in m["weights"].values()) == 1:
            lines.append("  此种子退化为单组件模型；不能声称该次实验验证了融合增益。")
    lines += ["", "## 组件移除与显式回退基线", "",
              "下表逐种子说明实际操作。回退项是替代基线对照，不是纯删除后的组件边际贡献；模型参数不按测试结果重选。", "",
              "| 种子 | 方法 | 对照基准 | 实际操作 | 权重（Bayes/Fuzzy/Rule） |", "|---|---|---|---|---|"]
    for seed, data in bundles.items():
        for name, info in data["ablations"].items():
            operation = (f"回退为 {info['fallback_component']} 基线" if info["kind"] == "baseline_fallback"
                         else "零权重组件，结果不变" if info["kind"] == "unchanged_zero_weight" else "删除并归一化")
            weights = "/".join(f"{info['result_weights'][component]:.4g}" for component in HybridModel.NAMES)
            lines.append(f"| {seed} | {name} | {ABLATION_SPECS[name][0]} | {operation} | {weights} |")
    lines += ["", "## 模糊隶属度自动拟合", ""]
    for seed, bundle in bundles.items():
        fuzzy = bundle["fuzzy_model"]
        lines.append(f"- seed {seed}: high_start={fuzzy['high_start']}，medium_weight={fuzzy['medium_weight']}；"
                     f"训练样本={fuzzy['training_count']}，训练Brier={fuzzy['training_brier']:.6f}；"
                     f"自动融合权重={bundle['adaptive_model']['weights']}。")
    lines.append("仅用训练集选择15组预设参数；权重只用验证集。训练损失改善不保证测试改善，所有压力条件和组件消融均保留。")
    lines += ["", "## D-S：来源B全部被反转的压力测试", "", "| 方法 | Accuracy | F1 | Brier↓ | 平均无知质量 |",
              "|---|---:|---:|---:|---:|"]
    for r in conflict:
        if r["conflict_rate"] == 1:
            lines.append(f"| {r['model']} | {r['accuracy_mean']:.4f} | {r['f1_mean']:.4f} | "
                         f"{r['brier_mean']:.4f} | {r['mean_ignorance_mean']:.4f} |")
    lines += ["", "## 如何审计与解读", "",
              "- samples.csv 保存实际输入和合成标签；predictions.csv 保存每条样本每个方法的评分，可独立重算指标。",
              "- metrics.csv 为逐种子实测值，summary.csv 为跨种子均值/标准差；不使用人为延时伪造加速。",
              "- models.json 保存仅由训练/验证集确定的参数；split_audit.json 检查样本ID不交叉。",
              "- sensitivity.csv 是固定其余参数、单独扰动先验/CPT/隶属度后的测试结果，不用于择优报告。",
              "- 严格消融只删除并归一化非零剩余权重；删除零权重组件保持结果不变。全零时预先指定的回退基线详见 ablations.csv，不据此归因组件边际贡献。",
              "- Brier 对非概率评分是平方误差诊断，不能据此把模糊隶属度称为校准概率。",
              "- 风险指标实验与结构推理效率实验不同：前者允许错误/缺测输入，后者验证同一知识库的精确闭包。",
              "- 来源可靠性正确时折扣可能改善冲突融合；若可靠性标注本身错误，效果不受保证。",
              "- 生成分布接近朴素贝叶斯假设；没有真实标签/跨分布验证，不能宣称可用于实际监管或优于所有方法。", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
