"""只从保存的 CSV 绘图，不内置实验结果。matplotlib 是可选依赖。"""
from __future__ import annotations
import csv
import textwrap
from pathlib import Path
from .artifacts import aggregate
from .metrics import reliability_bins

LABELS = {"rules": "Deterministic rules", "bayes_manual": "Bayes (manual CPT)",
          "bayes_learned": "Bayes (learned CPT)", "fuzzy": "Mamdani fuzzy",
          "hybrid_fixed": "Hybrid (fixed)", "hybrid_tuned": "Hybrid (validation)",
          "hybrid_no_rule": "Hybrid w/o rules", "hybrid_no_fuzzy": "Hybrid w/o fuzzy",
          "hybrid_no_bayes": "Hybrid w/o Bayes",
          "fuzzy_learned": "Mamdani (fitted)", "hybrid_adaptive": "Adaptive hybrid",
          "adaptive_no_rule": "Adaptive w/o rules", "adaptive_no_fuzzy": "Adaptive w/o fuzzy",
          "adaptive_no_bayes": "Adaptive w/o Bayes",
          "ds_classic": "Dempster", "ds_yager": "Conflict to ignorance",
          "ds_discount": "Reliability discount", "ds_robust": "Discount + conflict handling"}
COLORS = ["#355C7D", "#C66B3D", "#4F8061", "#8C6BA1", "#D0A23A", "#A85459", "#527F8C", "#726A5A", "#484848"]


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _model_label(row):
    label = LABELS[row["model"]]
    fallbacks = int(row.get("ablation_fallback_runs") or 0)
    if fallbacks:
        component = row["ablation_fallback_component"]
        label += f"\n{component.title()} baseline fallback ({fallbacks}/{row['ablation_runs']} seeds)"
    return label


def make_figures(experiments: Path, benchmarks: Path, output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True, "grid.alpha": .18, "figure.facecolor": "white"})
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    def save(fig, name, caption):
        fig.text(.01, .012, textwrap.fill(caption, width=int(fig.get_size_inches()[0] * 14)),
                 fontsize=8, color="#555555", va="bottom")
        fig.tight_layout(rect=(0, .065, 1, 1))
        target = output / name
        fig.savefig(target, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(target)
    if (experiments / "summary.csv").exists():
        summary = _csv(experiments / "summary.csv")
        lookup = {(r["scenario"], r["model"]): r for r in summary}
        clean = [r for r in summary if r["scenario"] == "clean"]
        fig, axes = plt.subplots(1, 2, figsize=(13.8, max(5.5, len(clean) * .48 + 1.8)))
        for ax, metric, title in [(axes[0], "accuracy", "Clean synthetic test: accuracy"),
                                   (axes[1], "brier", "Clean synthetic test: squared score loss")]:
            vals = [float(r[metric + "_mean"]) for r in clean]
            err = [float(r[metric + "_sd"]) for r in clean]
            ax.barh(range(len(clean)), vals, xerr=err, color=COLORS[:len(clean)], capsize=3)
            ax.set_yticks(range(len(clean)), [_model_label(r) for r in clean])
            ax.invert_yaxis()
            ax.set_title(title)
            ax.grid(axis="x")
        axes[0].set_xlim(0, 1.03)
        save(fig, "01_method_comparison.png", "Synthetic labels only. Bars: mean across seeds; error bars: sample SD. Fuzzy/rule scores are not calibrated probabilities.")
        main = ["rules", "bayes_learned", "fuzzy", "hybrid_fixed", "hybrid_tuned"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
        for ax, scenarios, x, title, xlabel in [
            (axes[0], ["clean", "noise_010", "noise_020", "noise_030"], [0, .1, .2, .3], "Additive Gaussian noise", "Noise standard deviation"),
            (axes[1], ["clean", "missing_010", "missing_030"], [0, .1, .3], "Missing indicators", "Missing probability")]:
            for i, model in enumerate(main):
                y = [float(lookup[s, model]["accuracy_mean"]) for s in scenarios]
                sd = [float(lookup[s, model]["accuracy_sd"]) for s in scenarios]
                ax.plot(x, y, "o-", color=COLORS[i], label=LABELS[model], markersize=4)
                ax.fill_between(x, [max(0, a - b) for a, b in zip(y, sd)], [min(1, a + b) for a, b in zip(y, sd)], color=COLORS[i], alpha=.10)
            ax.set(title=title, xlabel=xlabel, ylabel="Accuracy", ylim=(0, 1.03))
            ax.grid()
        axes[1].legend(fontsize=8, loc="lower left")
        save(fig, "02_noise_missing.png", "Paired synthetic test samples and labels across conditions. Models are frozen before stress testing; bands are between-seed SD.")
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        ablated = ["hybrid_fixed", "hybrid_tuned", "hybrid_no_rule", "hybrid_no_fuzzy", "hybrid_no_bayes"]
        vals = [float(lookup["mixed", m]["brier_mean"]) for m in ablated]
        err = [float(lookup["mixed", m]["brier_sd"]) for m in ablated]
        ax.barh(range(len(ablated)), vals, xerr=err, color=COLORS[:5], capsize=3)
        ax.set_yticks(range(len(ablated)), [_model_label(lookup["mixed", m]) for m in ablated])
        ax.invert_yaxis()
        ax.set(title="Component removal and explicit fallback baselines", xlabel="Squared score loss (lower is better)")
        ax.grid(axis="x")
        save(fig, "03_hybrid_ablation.png", "Noise SD=0.20; missing=20%. No retuning. Labels mark baseline fallback when removal leaves zero weight; these rows are not strict ablation. Details: ablations.csv.")
    if (experiments / "predictions.csv").exists():
        rows = [r for r in _csv(experiments / "predictions.csv") if r["scenario"] == "clean"]
        labels = [int(r["label"]) for r in rows]
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        ax.plot([0, 1], [0, 1], "--", color="#999999", label="Perfect reliability")
        for i, model in enumerate(["bayes_manual", "bayes_learned", "fuzzy", "hybrid_tuned"]):
            bins = reliability_bins(labels, [float(r[model]) for r in rows])
            ax.plot([b["mean_score"] for b in bins], [b["positive_fraction"] for b in bins], "o-", color=COLORS[i], label=LABELS[model])
        ax.set(xlabel="Mean forecast / normalized risk score", ylabel="Observed synthetic positive fraction",
               title="Reliability diagnostic (clean synthetic test)", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
        ax.grid()
        save(fig, "04_reliability.png", "Empty bins omitted. This diagnostic does not turn fuzzy memberships into probabilities or establish real-world calibration.")
    if (experiments / "conflict_summary.csv").exists():
        rows = _csv(experiments / "conflict_summary.csv")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
        for i, model in enumerate(["ds_classic", "ds_yager", "ds_discount", "ds_robust"]):
            items = sorted([r for r in rows if r["model"] == model], key=lambda r: float(r["conflict_rate"]))
            x = [float(r["conflict_rate"]) for r in items]
            for ax, metric in zip(axes, ["accuracy_mean", "mean_ignorance_mean"]):
                ax.plot(x, [float(r[metric]) for r in items], "o-", color=COLORS[i], label=LABELS[model], markersize=4)
                ax.set(xlabel="Injected source-B flip probability", ylim=(-.02, 1.03))
                ax.grid()
        axes[0].set(ylabel="Accuracy", title="Source conflict: classification")
        axes[1].set(ylabel="Mass assigned to ignorance", title="Source conflict: uncertainty preserved")
        axes[1].legend(fontsize=8, loc="upper left")
        save(fig, "05_evidence_conflict.png", "Two simulated sources, fixed A/B order. Source-quality discounts are preset metadata, not inferred from test labels or flip flags.")
    if (experiments / "sensitivity.csv").exists():
        rows = _csv(experiments / "sensitivity.csv")
        numeric = [{**r, "value": float(r["value"]), "brier": float(r["brier"])} for r in rows]
        stats = aggregate(numeric, ("parameter", "value"), ("brier",))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, (ax, param, title) in enumerate(zip(axes, ["prior", "cpt_positive_shift", "high_start"],
                                                 ["Bayes prior sensitivity", "Bayes CPT sensitivity", "Fuzzy membership sensitivity"])):
            items = sorted([r for r in stats if r["parameter"] == param], key=lambda r: r["value"])
            ax.errorbar([r["value"] for r in items], [r["brier_mean"] for r in items],
                        yerr=[r["brier_sd"] for r in items], color=COLORS[i], marker="o", capsize=3)
            ax.set(title=title, xlabel=param, ylabel="Squared score loss")
            ax.grid()
        save(fig, "06_parameter_sensitivity.png", "One parameter changed at a time on the clean synthetic test set. These sweeps are diagnostic and are not used to select test-optimal parameters.")
    if (benchmarks / "forward_summary.csv").exists():
        rows = _csv(benchmarks / "forward_summary.csv")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
        for i, model in enumerate(["naive_scan", "indexed_naive", "semi_naive"]):
            items = sorted([r for r in rows if r["model"] == model], key=lambda r: int(r["groups"]))
            x = [int(r["groups"]) for r in items]
            axes[0].errorbar(x, [float(r["elapsed_ms_mean"]) for r in items],
                            yerr=[float(r["elapsed_ms_sd"]) for r in items], marker="o", capsize=3,
                            color=COLORS[i], label=model)
            axes[1].plot(x, [float(r["candidate_checks_mean"]) for r in items], "o-", color=COLORS[i], label=model)
        axes[0].set(ylabel="Inference time (ms)", title="Same closure: wall-clock time", yscale="log")
        axes[1].set(ylabel="Actual candidate checks", title="Same closure: search work", yscale="log")
        for ax in axes:
            ax.set_xlabel("Synthetic equity groups")
            ax.grid()
            ax.legend(fontsize=9)
        save(fig, "07_deterministic_scaling.png", "Same facts, 27 rules and equal closure hashes. Construction / file I/O excluded. Query-pruned workloads are reported separately.")
    if (experiments / "summary.csv").exists():
        rows = _csv(experiments / "summary.csv")
        lookup = {(r["scenario"], r["model"]): r for r in rows}
        if ("clean", "fuzzy_learned") in lookup:
            models = ["fuzzy", "fuzzy_learned", "bayes_learned", "hybrid_adaptive"]
            scenarios = ["clean", "noise_010", "noise_020", "noise_030", "missing_010", "missing_030", "mixed", "severe"]
            fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
            for i, model in enumerate(models):
                for ax, metric in zip(axes, ["f1", "brier"]):
                    ax.errorbar(range(len(scenarios)), [float(lookup[s, model][metric + "_mean"]) for s in scenarios],
                                yerr=[float(lookup[s, model][metric + "_sd"]) for s in scenarios],
                                marker="o", capsize=2, color=COLORS[i], label=LABELS[model])
                    ax.set_xticks(range(len(scenarios)), ["Clean", "N.1", "N.2", "N.3", "M.1", "M.3", "Mixed", "Severe"], rotation=20)
                    ax.grid(axis="y")
            axes[0].set(title="Automatic fuzzy fitting: F1", ylabel="F1", ylim=(0, 1.03))
            axes[1].set(title="Automatic fuzzy fitting: score loss", ylabel="Brier / squared score error")
            axes[1].legend(fontsize=8)
            save(fig, "08_adaptive_fuzzy.png", "Mamdani parameters fitted on training data only; fusion weights use validation data. All conditions retained; test performance need not improve.")
    return saved
