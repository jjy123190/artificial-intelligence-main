"""从端到端CSV作图，不内置结果，不绘制课程PPT。"""
import csv
import textwrap
from pathlib import Path

LABELS = {"rules_only": "Structure rules", "bayes_only": "Bayes",
          "fuzzy_only": "Mamdani", "hybrid_fixed_unlocked": "Fixed hybrid / no lock",
          "hybrid_fixed_locked": "Fixed hybrid / locked", "hybrid_tuned_locked": "Tuned hybrid / locked"}
PALETTE = {"rules_only": "#6D7787", "bayes_only": "#355C7D", "fuzzy_only": "#8B6EA1",
           "hybrid_fixed_unlocked": "#C77A49", "hybrid_fixed_locked": "#4B8066",
           "hybrid_tuned_locked": "#A15155"}
CONDITIONS = ("clean", "value_noise", "edge_missing_020", "false_loop_020", "combined", "severe")
SHORT = ("Clean", "Values", "Missing edges", "False loop", "Combined", "Severe")


def make_e2e_figures(source: Path, output: Path) -> list[Path]:
    if not (source / "summary.csv").exists():
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (source / "summary.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    lookup = {(r["scenario"], r["model"]): r for r in rows}
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "grid.alpha": .18})
    def save(fig, name, note):
        fig.text(.01, .015, textwrap.fill(note, 150), fontsize=8, color="#555555", va="bottom")
        fig.tight_layout(rect=(0, .09, 1, 1))
        target = output / name
        fig.savefig(target, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(target)
    def series(model, metric):
        return [float(lookup[c, model][metric + "_mean"]) if lookup[c, model][metric + "_mean"] else float("nan")
                for c in CONDITIONS]
    models = ["rules_only", "bayes_only", "hybrid_fixed_unlocked", "hybrid_fixed_locked", "hybrid_tuned_locked"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for model in models:
        for ax, metric in zip(axes, ("f1", "recall")):
            ax.plot(range(len(CONDITIONS)), series(model, metric), "o-", color=PALETTE[model],
                    label=LABELS[model], markersize=4)
    for ax, title in zip(axes, ("End-to-end action F1", "End-to-end action recall")):
        ax.set(title=title, ylim=(-.02, 1.03))
        ax.set_xticks(range(len(CONDITIONS)), SHORT, rotation=20, ha="right")
        ax.grid(axis="y")
    axes[1].legend(fontsize=8, loc="lower left")
    save(fig, "01_action_performance.png", "Entire observed graph is reasoned before every decision. Means across seeds; synthetic audit labels. Locking changes actions, never the underlying score.")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for model in ["rules_only", "bayes_only", "hybrid_fixed_unlocked", "hybrid_fixed_locked"]:
        for ax, metric in zip(axes, ("constraint_violation_rate", "false_positive_rate")):
            ax.plot(range(len(CONDITIONS)), series(model, metric), "o-", color=PALETTE[model], label=LABELS[model], markersize=4)
    for ax, title in zip(axes, ("Known high-alert violations", "False-positive rate vs audit labels")):
        ax.set(title=title, ylim=(-.02, 1.03))
        ax.set_xticks(range(len(CONDITIONS)), SHORT, rotation=20, ha="right")
        ax.grid(axis="y")
    axes[0].legend(fontsize=8)
    save(fig, "02_constraint_and_cost.png", "Zero known-alert violations is a policy invariant, NOT perfect risk prediction. Incorrect source facts can cause extra forced reviews; these errors remain in the metrics.")
    with (source / "paired_summary.csv").open(encoding="utf-8-sig", newline="") as fh:
        paired = {r["scenario"]: r for r in csv.DictReader(fh) if r["family"] == "fixed"}
    fig, ax = plt.subplots(figsize=(10, 4.9))
    x = list(range(len(CONDITIONS)))
    ax.bar([i - .18 for i in x], [float(paired[c]["delta_tp_mean"]) for c in CONDITIONS],
           width=.36, color="#4B8066", label="Additional true positives")
    ax.bar([i + .18 for i in x], [float(paired[c]["delta_fp_mean"]) for c in CONDITIONS],
           width=.36, color="#C77A49", label="Additional false positives")
    ax.set_xticks(x, SHORT, rotation=15, ha="right")
    ax.set(title="Paired locking benefit and cost", ylabel="Mean extra reviews per seed")
    ax.grid(axis="y")
    ax.legend()
    save(fig, "03_locking_tradeoff.png", "Same graph, same weights, same threshold. All changed actions are retained in the evaluation; no selecting only the favorable cases.")
    if ("clean", "hybrid_adaptive_calibrated") in lookup:
        fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))
        for model, label, color in [("hybrid_adaptive_unlocked", "Adaptive / unlocked", "#355C7D"),
                                   ("hybrid_adaptive_locked", "Adaptive / hard lock", "#4B8066"),
                                   ("hybrid_adaptive_calibrated", "Adaptive / calibrated", "#C77A49")]:
            for ax, metric in zip(axes, ("f1", "false_positive_rate", "decision_cost")):
                ax.plot(range(len(CONDITIONS)), series(model, metric), "o-", color=color, label=label)
                ax.set_xticks(range(len(CONDITIONS)), SHORT, rotation=25, ha="right")
                ax.grid(axis="y")
        axes[0].set(title="Adaptive action F1", ylim=(-.02, 1.03))
        axes[1].set(title="False-positive rate", ylim=(-.02, 1.03))
        axes[2].set(title="Decision cost (FP + 2FN) / N")
        axes[2].legend(fontsize=8)
        save(fig, "04_calibrated_policy.png", "Training-only fuzzy fitting; validation-only support gating. Structure proofs are retained even when review is deferred. Calibration can trade missed alerts for fewer false positives; no universal improvement claim.")
    return saved
