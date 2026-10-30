"""Finish the resumed lab02 audit without changing production code or formal results."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.kb.features import FEATURES, RiskObservation
from core.kb.graph_io import case_from_dict
from experiments.lab02.eval.artifacts import execution_manifest, fingerprint
from experiments.lab02.eval.graph_data import GRAPH_CONDITIONS, perturb_graph
from experiments.lab02.hybrid.model import HybridModel
from experiments.lab02.hybrid.pipeline import assess_company, prepare_case


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def read_csv(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)


def verify_saved_evidence():
    current = execution_manifest(ROOT)
    manifests = {}
    for folder in ("demo", "experiments", "end_to_end", "benchmarks", "governance"):
        relative = f"outputs/lab02/{folder}/manifest.json"
        saved = read_json(relative)
        assert saved["source_sha256"] == current["source_sha256"], folder
        assert saved["snapshot_sha256"] == current["snapshot_sha256"], folder
        base = (ROOT / "outputs/lab02" / folder).resolve()
        for name, digest in saved["artifact_sha256"].items():
            path = (base / name).resolve()
            assert path.is_relative_to(base) and fingerprint(path) == digest, path
        manifests[folder] = {"manifest_sha256": fingerprint(ROOT / relative),
                             "verified_artifacts": len(saved["artifact_sha256"])}
    project_tests = read_json("outputs/qa/test-summary.json")
    replay = read_json("outputs/lab02/qa/programming_acceptance.json")
    assert project_tests["successful"] and project_tests["tests_run"] == 192
    assert replay["successful"] and replay["full_replay"]
    browser = read_json("outputs/qa/course-review/after/browser-checks.json")
    for category in ("source_sha256", "protected_files_sha256"):
        for relative, digest in browser[category].items():
            assert fingerprint(ROOT / relative) == digest, relative
    independent = read_json("outputs/qa/lab02-review/independent-review.json")
    assert independent["supported_domain_oracles_passed"]
    return {"current_source_files": len(current["source_sha256"]),
            "current_data_files": len(current["snapshot_sha256"]), "manifests": manifests,
            "reused_project_tests": project_tests,
            "reused_full_replay": replay,
            "browser_source_files_unchanged": len(browser["source_sha256"]),
            "reused_task_journey_groups": len(browser["checks"]),
            "reused_independent_check_groups": len(independent["checks"]),
            "reuse_note": "Prior session completed these checks; current code/data/artifact hashes match. Timings were not rerun."}


def validation_weight_selection():
    """Enumerate all 15 losses from saved validation inputs, without fit_weights()."""
    indicator = read_csv("outputs/lab02/experiments/samples.csv")
    graphs = [json.loads(line) for line in (ROOT / "outputs/lab02/end_to_end/reference_graphs.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    candidates = [(a / 4, b / 4, c / 4) for a in range(5) for b in range(5)
                  for c in range(5) if a + b + c == 4]
    checked = []
    for family, samples in (("experiments", indicator), ("end_to_end", graphs)):
        for seed, bundle in read_json(f"outputs/lab02/{family}/models.json")["models_by_seed"].items():
            rows = [r for r in samples if str(r["seed"]) == seed and r["split"] == "validation"]
            observations = []
            for row in rows:
                if family == "experiments":
                    observation = RiskObservation(row["sample_id"], {
                        f: float(row[f]) if row[f] else None for f in FEATURES})
                else:
                    observation = prepare_case(case_from_dict(row["case"])).observations[row["target_id"]]
                observations.append(observation)
            labels = [int(r["label"]) for r in rows]
            for variant in ("model", "adaptive_model"):
                model = HybridModel.from_dict(bundle[variant])
                scores = [model.components(o) for o in observations]
                losses = {w: math.fsum((math.fsum(w[i] * s[name]
                          for i, name in enumerate(model.NAMES)) - y) ** 2
                          for s, y in zip(scores, labels)) / len(labels) for w in candidates}
                expected = min(candidates, key=lambda w: (losses[w], -w[0], -w[1]))
                assert model.weights == expected, (family, seed, variant, model.weights, expected)
                assert model.validation_count == len(rows)
                close(model.validation_brier, losses[expected])
                checked.append({"family": family, "seed": seed, "variant": variant,
                                "validation_rows": len(rows), "candidate_count": len(candidates),
                                "selected_weights": list(expected), "validation_brier": losses[expected]})
    return {"model_selections": checked, "fit_weights_called": False, "test_rows_used": 0}


def validation_policy_replay():
    """Rebuild validation perturbations, then independently count FP/FN for six gates."""
    refs = [json.loads(line) for line in (ROOT / "outputs/lab02/end_to_end/reference_graphs.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    saved = read_csv("outputs/lab02/end_to_end/policy_validation.csv")
    lookup = {(r["seed"], r["scenario"], r["sample_id"]): r for r in saved}
    assert len(lookup) == len(saved)
    threshold = read_json("outputs/lab02/end_to_end/config.json")["config"]["threshold"]
    conditions = {c.name: c for c in GRAPH_CONDITIONS}
    results, total = [], 0
    for seed, bundle in read_json("outputs/lab02/end_to_end/models.json")["models_by_seed"].items():
        refs_for_seed = [r for r in refs if str(r["seed"]) == seed and r["split"] == "validation"]
        model = HybridModel.from_dict(bundle["adaptive_model"])
        rebuilt = []
        for condition_name in bundle["policy_fit_conditions"]:
            for index, reference in enumerate(refs_for_seed):
                changed = perturb_graph(case_from_dict(reference["case"]), reference["target_id"],
                                        conditions[condition_name], int(seed) * 100003 + index * 59 + 811)
                context = prepare_case(changed.case, completeness=changed.completeness)
                decision = assess_company(context, reference["target_id"], model,
                                          method="hybrid_unlocked", threshold=threshold)
                row = lookup[(seed, condition_name, reference["sample_id"])]
                assert row["split"] == "validation" and int(row["label"]) == reference["label"]
                close(float(row["score"]), decision["score"])
                assert int(row["observed_required"]) == int(decision["structural_alert_requires_review"])
                rebuilt.append((decision["score"], decision["structural_alert_requires_review"], reference["label"]))
        policy = bundle["adaptive_model"]["review_policy"]
        history = []
        for support in (0.0, .15, .25, .35, .45, .5):
            fp = fn = 0
            for score, observed, label in rebuilt:
                action = score >= threshold or (observed and score >= support)
                fp += int(action and label == 0)
                fn += int(not action and label == 1)
            history.append({"minimum_support": support, "fp": fp, "fn": fn,
                            "cost": (fp + policy["false_negative_cost"] * fn) / len(rebuilt)})
        expected = min(history, key=lambda r: (r["cost"], r["minimum_support"]))
        assert policy["minimum_support"] == expected["minimum_support"]
        assert policy["validation_count"] == len(rebuilt)
        assert history == policy["selection_history"]
        total += len(rebuilt)
        results.append({"seed": seed, "validation_rows_replayed": len(rebuilt),
                        "candidates": len(history), "selected": expected})
    assert total == len(saved)
    return {"policies": results, "total_validation_rows_replayed": total,
            "policy_fit_called": False, "test_rows_used": 0}


def ablation_scope():
    models = read_json("outputs/lab02/experiments/models.json")["models_by_seed"]
    switches = []
    for seed, bundle in models.items():
        for variant in ("model", "adaptive_model"):
            model = HybridModel.from_dict(bundle[variant])
            assert model.weights == (1.0, 0.0, 0.0)
            removed = model.without("bayes")
            assert removed.weights == (0.0, 1.0, 0.0)
            switches.append({"seed": seed, "variant": variant, "before": list(model.weights),
                             "after": list(removed.weights), "remaining_positive_weight_before_fallback": 0})
    rows = read_csv("outputs/lab02/experiments/predictions.csv")
    for row in rows:
        close(float(row["hybrid_no_bayes"]), float(row["fuzzy"]))
        close(float(row["adaptive_no_bayes"]), float(row["fuzzy_learned"]))
    return {"switches": switches, "fallback_score_comparisons": 2 * len(rows),
            "classification": "evaluation description and interpretation issue; fallback is partly documented",
            "precise_behavior": "Deleting the only weighted component activates a previously zero-weight fuzzy baseline.",
            "not_a_claim": "This does not show wrong Bayes/Fuzzy arithmetic or test-set parameter leakage."}


def fresh_process_logic_boundaries():
    code = '''
import json
from core.kb.fol import Exists, Not, And, Var, atom, to_clauses, Clause, Literal
from experiments.lab02.deterministic.resolution import ResolutionEngine
f = Exists((Var('x'),), atom('P', '?x'))
g = Not(atom('P', 'sk1'))
separate = to_clauses(f) + to_clauses(g)
combined = to_clauses(And((f, g)))
n = 2 ** 53 + 1
numeric = [Clause.make([Literal(atom('Amount', n))])]
print(json.dumps({
    'counter_reset_used': False,
    'skolem': {'separate_clauses': [str(c) for c in separate],
               'incorrect_unrelated_goal_proved': ResolutionEngine().prove(separate, atom('Unrelated')).proved,
               'combined_clauses': [str(c) for c in combined],
               'combined_proves_unrelated': ResolutionEngine().prove(combined, atom('Unrelated')).proved},
    'numeric': {'input_integer': n, 'converted_value': atom('Amount', n).args[0].value,
                'incorrect_neighbor_goal_proved': ResolutionEngine().prove(numeric, atom('Amount', n-1)).proved}
}))
'''
    process = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", code],
                             cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    result = json.loads(process.stdout)
    assert result["skolem"]["incorrect_unrelated_goal_proved"]
    assert not result["skolem"]["combined_proves_unrelated"]
    assert result["numeric"]["incorrect_neighbor_goal_proved"]
    result["scope"] = "Generic FOL API boundary defects; course resolution uses universal rules and small numeric values."
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "source_session_id": "01a1729a-f46c-7723-9b78-dc4528eba7f3",
              "scope": "Review lab02 implementation and report findings; formal report/PPT remain deferred.",
              "production_code_modified": False, "formal_results_modified": False, "checks": []}
    for function in (verify_saved_evidence, validation_weight_selection, validation_policy_replay,
                     ablation_scope, fresh_process_logic_boundaries):
        started = time.perf_counter()
        try:
            item = {"name": function.__name__, "completed": True, "details": function()}
        except Exception:
            item = {"name": function.__name__, "completed": False, "error": traceback.format_exc()}
        item["elapsed_seconds"] = time.perf_counter() - started
        report["checks"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    report["review_checks_completed"] = all(item["completed"] for item in report["checks"])
    report["status"] = "review_completed_with_findings" if report["review_checks_completed"] else "review_checks_failed"
    report["finding_ids"] = ["ablation_fallback_labeling", "skolem_collision_across_formulas", "fol_large_integer_precision"]
    path = Path(__file__).with_name("completion-review.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["review_checks_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
