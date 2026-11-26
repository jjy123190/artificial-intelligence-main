"""Independent verification of artifacts, per-task metrics and paired summaries."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from datetime import datetime

from .data import digest_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name):
    require(math.isfinite(float(actual)) and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10),
            f"Metric mismatch: {name}: {actual} != {expected}")


def verify_artifacts(directory):
    directory = Path(directory).resolve()
    result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    require(result["status"] == "complete", "Result is not complete")
    checked = []
    for entry in [*result["weights"].values(), *result.get("priors", {}).values(), result["demo_pool"]]:
        path = directory / entry["path"]
        require(path.resolve().is_relative_to(directory) and not path.is_symlink(), "Unsafe artifact path")
        require(digest_file(path) == entry["sha256"], f"Hash mismatch: {path.name}")
        checked.append({"file": entry["path"], "sha256": entry["sha256"]})
    episode_data = json.loads((directory / "episodes.json").read_text(encoding="utf-8"))
    protocol = result["protocol"]
    require(episode_data["seed"] == protocol["seed"] + 200, "Episode seed mismatch")
    require(protocol["ways"] == 5 and protocol["queries_per_class"] == 15, "Unexpected task protocol")
    expected = protocol["test_episodes_per_shot"]
    require(len(episode_data["episodes"]) == expected * 2, "Incomplete episode records")
    methods = {row["method"] for row in result["rows"]}
    require(len(result["rows"]) == len(methods) * 2, "Missing or duplicate summary rows")
    for shots in (1, 5):
        records = sorted((r for r in episode_data["episodes"] if r["shots"] == shots), key=lambda r:r["episode"])
        require([r["episode"] for r in records] == list(range(expected)), "Incomplete or duplicate episode IDs")
        for record in records:
            require(len(record["classes"]) == len(set(record["classes"])) == 5, "Invalid episode classes")
            require(len(record["support_ids"]) == len(set(record["support_ids"])) == 5 * shots, "Invalid supports")
            require(len(record["query_ids"]) == len(set(record["query_ids"])) == 75, "Invalid queries")
            require(not set(record["support_ids"]) & set(record["query_ids"]), "Support/query leakage")
            require(set(record["methods"]) == methods, "Unpaired method records")
            if "query_labels" in record:
                require(len(record["query_labels"]) == 75 and all(record["query_labels"].count(c) == 15 for c in range(5)), "Invalid query labels")
            for method, entry in record["methods"].items():
                require(math.isfinite(entry["seconds"]) and entry["seconds"] >= 0, "Invalid timing")
                require(math.isfinite(entry["accuracy"]) and 0 <= entry["accuracy"] <= 1, "Invalid accuracy")
                if "predictions" in entry:
                    require(len(entry["predictions"]) == 75 and all(type(p) is int and 0 <= p < 5 for p in entry["predictions"]), "Invalid predictions")
                    correct = sum(a == b for a,b in zip(entry["predictions"], record["query_labels"]))
                    require(entry["correct"] == correct, "Incorrect per-query score")
                    close(entry["accuracy"], correct / 75, method)
        baseline = [r["methods"]["baseline"]["accuracy"] for r in records]
        for method in methods:
            row = next(row for row in result["rows"] if row["method"] == method and row["shots"] == shots)
            values = [r["methods"][method]["accuracy"] for r in records]
            delta = [a-b for a,b in zip(values,baseline)]
            close(row["accuracy"], statistics.mean(values), f"{method}/{shots} accuracy")
            close(row["ci95"], 1.96 * statistics.stdev(values) / math.sqrt(expected), f"{method}/{shots} interval")
            close(row["paired_delta"], statistics.mean(delta), f"{method}/{shots} paired delta")
            close(row["paired_ci95"], 1.96 * statistics.stdev(delta) / math.sqrt(expected), f"{method}/{shots} paired interval")
            close(row["mean_episode_seconds"], statistics.mean(r["methods"][method]["seconds"] for r in records), f"{method}/{shots} time")
    if result.get("schema_version") == 2:
        require(digest_file(directory / "v2_calibration_locked.json") == result["selection_lock_sha256"], "Selection lock changed")
        require(digest_file(directory / "evaluation_episodes.jsonl") == result["episode_journal_sha256"], "Episode journal changed")
        lock = json.loads((directory / "v2_calibration_locked.json").read_text(encoding="utf-8"))
        require(datetime.fromisoformat(lock["locked_utc"]) <= datetime.fromisoformat(result["completed_utc"]), "Invalid lock chronology")
        require(lock["test_accessed"] is False and lock["status"] == "locked", "Test-before-lock violation")
        for name in ("baseline", "aligned"):
            require(lock["weights"][name] == result["weights"][name]["sha256"], "Selection weight mismatch")
        plan = json.loads((directory / "evaluation_plan.json").read_text(encoding="utf-8"))
        require(plan["lock_sha256"] == result["selection_lock_sha256"], "Evaluation plan lock mismatch")
        require(plan["maml_sha256"] == result["weights"]["maml"]["sha256"], "MAML weight selection changed")
        require(plan["adaptation"] == result["maml_adaptation"], "MAML adaptation changed after plan lock")
        for shots in ("1","5"):
            require(lock["configurations"][shots]["config"] == result["calibration"][shots], "Calibration changed after locking")
            require(lock["configurations"][shots]["space"] == result["study"]["feature_spaces"][shots], "Feature space changed after locking")
    return {"verified": True, "files": checked, "episodes_recomputed": expected * 2,
            "methods": len(methods), "summary_rows_recomputed": len(result["rows"])}
