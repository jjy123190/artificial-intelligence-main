"""Dependency-light publication audit shared by website snapshots and CLI checks.

No torch, NumPy, checkpoint deserialization, or inference is imported here.
Scores are independently recomputed; hash-only checks do not authenticate an
edited summary. Cache keys cover every consumed file and are rechecked after
an audit to reject partial publication. Missing optional models allow results
to be viewed, but cannot report inference as ready.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics
from threading import RLock

_LOCK = RLock()
_CORE = ("results.json", "episodes.json", "baseline.pt", "aligned.pt", "maml.pt",
         "baseline_prior.npz", "aligned_prior.npz", "demo_pool.npz")
_GRAPH = ("graph_results.json", "graph_transform.npz", "graph_configuration.json",
          "graph_episodes.json", "graph_lock.json")
_MULTI = ("multiview_results.json", "multiview_transform.npz",
          "multiview_configuration.json", "multiview_episodes.json")
_PLAN = ("v2_calibration_locked.json", "evaluation_plan.json", "evaluation_episodes.jsonl")
_WIDE = ("wide_results.json", "wide.pt", "wide_transform.npz", "wide_configuration.json", "wide_episodes.json",
         "wide_validation.json", "wide_training_plan.json", "wide_training.json")
_WIDEVIEW = ("wideview_results.json", "wideview_episodes.json", "wideview_configuration.json",
             "wideview_transform.npz", "wideview_selection.json", "wideview_plan.json")
_WIDEVIEW_SELECTION = ("next_study/wide_views_20261125/scores.jsonl",
                       "next_study/wide_views_20261125/selection_lock.json")
_CONTINUED = ("continued_results.json", "continued.pt", "continued_transform.npz", "continued_training.json", "continued_training_plan.json",
              "continued_initial_validation.json", "continued_validation.json", "continued_validation_plan.json", "continued_validation_tasks.jsonl",
              "continued_checkpoint_lock.json", "continued_selection_lock.json", "continued_configuration.json", "continued_episodes.json")
_NAMES = (*_CORE, *_GRAPH, *_MULTI, *_PLAN, *_WIDE, *_WIDEVIEW, *_WIDEVIEW_SELECTION, *_CONTINUED, "edge_benchmark.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), "Nonfinite/invalid " + name)
    return value


def close(actual, expected, name):
    finite(actual, name)
    require(math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10), "Measured evidence mismatch: " + name)


def safe_path(directory, name):
    require(isinstance(name, str) and name in _NAMES, "Unexpected artifact filename")
    path = directory / name
    require(not path.is_symlink() and path.resolve().is_relative_to(directory), "Unsafe artifact path")
    return path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(directory, name):
    path = safe_path(directory, name)
    # Artifacts are local educational records, not arbitrarily large payloads.
    limit = 128 * 1024 * 1024 if "episodes" in name else 8 * 1024 * 1024
    require(path.stat().st_size <= limit, "Artifact exceeds bounded size: " + name)
    def reject_constant(value):
        raise ValueError("Invalid JSON number: " + value)
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def validate_hash(directory, name, digest, *, allow_missing=False):
    require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Invalid SHA256")
    path = safe_path(directory, name)
    if allow_missing and not path.exists():
        return False
    require(path.is_file() and file_hash(path) == digest, "Artifact checksum mismatch: " + name)
    return True


def observations(directory):
    result = []
    for name in _NAMES:
        path = safe_path(directory, name)
        if not path.exists():
            result.append((name, None))
        else:
            info = path.stat()
            result.append((name, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))
    return tuple(result)


def half_width(values):
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.


def summary(values, row, count, name):
    require(type(row["episodes"]) is int and row["episodes"] == count, "Wrong episode count: " + name)
    close(row["accuracy"], statistics.mean(values), name + "/accuracy")
    close(row["ci95"], half_width(values), name + "/interval")
    require(0 <= row["accuracy"] <= 1 and row["ci95"] >= 0, "Invalid accuracy range")


def paired(values, references, row, mean_key, ci_key):
    delta = [a - b for a, b in zip(values, references)]
    close(row[mean_key], statistics.mean(delta), mean_key)
    close(row[ci_key], half_width(delta), ci_key)


def task_ids(record, shots, total):
    support, query = record["support_ids"], record["query_ids"]
    require(isinstance(support, list) and isinstance(query, list), "Invalid image ID arrays")
    require(all(isinstance(x, str) and x for x in support + query), "Invalid image ID")
    require(len(support) == len(set(support)) == 5 * shots, "Duplicated/incomplete support")
    require(len(query) == len(set(query)) == total, "Duplicated/incomplete query")
    require(set(support).isdisjoint(query), "Support/query overlap")


def predictions(entry, targets, total, key="predictions", accuracy="accuracy"):
    values = entry[key]
    require(isinstance(values, list) and len(values) == total, "Wrong prediction count")
    require(all(type(p) is int and 0 <= p < 5 for p in values), "Invalid predicted class")
    score = sum(a == b for a, b in zip(values, targets)) / total
    close(entry[accuracy], score, accuracy + "/correct count")
    if key == "predictions" and "correct" in entry:
        require(type(entry["correct"]) is int and entry["correct"] == round(score * total), "Wrong correct count")
    return values


def index_rows(rows, methods, count):
    require(isinstance(rows, list) and len(rows) == len(methods) * 2, "Missing/duplicate summary rows")
    index = {}
    for row in rows:
        require(type(row["shots"]) is int and row["shots"] in (1, 5), "Invalid shot setting")
        require(row["method"] in methods, "Unexpected method")
        key = (row["method"], row["shots"])
        require(key not in index, "Duplicate method/shot summary")
        index[key] = row
        finite(row["accuracy"], "accuracy"); finite(row["ci95"], "ci95")
        require(type(row["episodes"]) is int and row["episodes"] == count, "Incorrect summary task count")
    return index


def audit_core(directory):
    result = read_json(directory, "results.json")
    require(result.get("status") == "complete", "Results not complete")
    require(result.get("schema_version") in (1,2), "Unsupported artifact schema")
    protocol = result["protocol"]
    require(protocol["ways"] == 5 and protocol["queries_per_class"] == 15 and protocol["shots"] == [1, 5], "Unsupported protocol")
    count = protocol["test_episodes_per_shot"]
    require(type(count) is int and 2 <= count <= 10000, "Invalid task count")
    require(type(protocol["seed"]) is int and 0 <= protocol["seed"] <= 2**32-1, "Invalid study seed")
    ready = True
    require(set(result["weights"]) == {"baseline", "aligned", "maml"}, "Unexpected checkpoint manifest")
    for name, entry in result["weights"].items():
        require(entry["path"] == name + ".pt", "Unexpected weight path")
        ready = validate_hash(directory, entry["path"], entry["sha256"], allow_missing=True) and ready
    require(set(result.get("priors", {})) == {"baseline", "aligned"}, "Incomplete prior manifest")
    for name, entry in result["priors"].items():
        require(entry["path"] == name + "_prior.npz", "Unexpected prior path")
        ready = validate_hash(directory, entry["path"], entry["sha256"], allow_missing=True) and ready
    pool = result["demo_pool"]
    require(pool["path"] == "demo_pool.npz", "Unexpected demo pool path")
    ready = validate_hash(directory, "demo_pool.npz", pool["sha256"], allow_missing=True) and ready
    data = read_json(directory, "episodes.json")
    require(data["seed"] == protocol["seed"] + 200, "Episode seed mismatch")
    require(len(data["episodes"]) == count * 2, "Incomplete core episodes")
    methods = {r["method"] for r in result["rows"]}
    require({"baseline", "maml", "full"}.issubset(methods), "Missing comparison methods")
    rows = index_rows(result["rows"], methods, count)
    index = {}
    for record in data["episodes"]:
        shots, number = record["shots"], record["episode"]
        require(type(shots) is int and shots in (1,5) and type(number) is int and 0 <= number < count, "Invalid task index")
        key = (shots, number)
        require(key not in index, "Duplicate core task")
        task_ids(record, shots, 75)
        require(len(record["classes"]) == len(set(record["classes"])) == 5, "Invalid selected classes")
        targets = record.get("query_labels")
        if targets is not None:
            require(len(targets) == 75 and all(type(x) is int and 0 <= x < 5 for x in targets), "Invalid query score labels")
            require(all(targets.count(c) == 15 for c in range(5)), "Evaluation class counts disagree")
        require(set(record["methods"]) == methods, "Methods not paired")
        for method, entry in record["methods"].items():
            score = finite(entry["accuracy"], "accuracy")
            require(0 <= score <= 1, "Score out of range")
            duration = finite(entry["seconds"], "seconds")
            require(duration >= 0, "Negative duration")
            if "predictions" in entry:
                require(targets is not None, "Predictions have no scoring labels")
                predictions(entry, targets, 75)
        # Store only references needed by optional extensions, not large score arrays.
        index[key] = {"support_ids": record["support_ids"], "query_ids": record["query_ids"],
                      "query_labels": targets, "scores": {k:v["accuracy"] for k,v in record["methods"].items()},
                      "times": {k:v["seconds"] for k,v in record["methods"].items()}}
    for shots in (1,5):
        tasks = [index[(shots, i)] for i in range(count)]
        for method in methods:
            values = [r["scores"][method] for r in tasks]; row = rows[(method, shots)]
            summary(values, row, count, method)
            paired(values, [r["scores"]["baseline"] for r in tasks], row, "paired_delta", "paired_ci95")
            close(row["mean_episode_seconds"], statistics.mean(r["times"][method] for r in tasks), "core duration")
    if result.get("schema_version") == 2:
        lock = read_json(directory, "v2_calibration_locked.json")
        validate_hash(directory, "v2_calibration_locked.json", result["selection_lock_sha256"])
        validate_hash(directory, "evaluation_episodes.jsonl", result["episode_journal_sha256"])
        require(lock["status"] == "locked" and lock["test_accessed"] is False, "Missing pre-test lock")
        require(datetime.fromisoformat(lock["locked_utc"]) <= datetime.fromisoformat(result["completed_utc"]), "Wrong test chronology")
        for name in ("baseline", "aligned"):
            require(lock["weights"][name] == result["weights"][name]["sha256"], "Locked backbone changed")
        for shots in ("1", "5"):
            require(lock["configurations"][shots]["config"] == result["calibration"][shots], "Calibration changed after lock")
            require(lock["configurations"][shots]["space"] == result["study"]["feature_spaces"][shots], "Feature space changed after lock")
        plan = read_json(directory, "evaluation_plan.json")
        require(plan["lock_sha256"] == result["selection_lock_sha256"] and plan["maml_sha256"] == result["weights"]["maml"]["sha256"], "Evaluation identity mismatch")
        require(plan["adaptation"] == result["maml_adaptation"], "Adaptation changed after lock")
    return result, index, ready


def audit_extension(directory, name, core, reference, graph=None):
    file = name + "_results.json"
    if not (directory / file).exists():
        return None, {}
    result = read_json(directory, file)
    require(result.get("status") == "complete", "Incomplete extension")
    require(result["base_weights_sha256"] == core["weights"]["aligned"]["sha256"], "Extension backbone mismatch")
    names = set(_GRAPH[1:] if name == "graph" else _MULTI[1:])
    hashes = result["validated_files" if name == "graph" else "files"]
    require(set(hashes) == names, "Incomplete extension dependencies")
    for filename, entry in hashes.items():
        require(entry["path"] == filename, "Invalid extension dependency path")
        validate_hash(directory, filename, entry["sha256"])
    config = read_json(directory, name + "_configuration.json")
    if name == "graph":
        lock = read_json(directory, "graph_lock.json")
        require(lock["test_accessed"] is False and lock["weights_sha256"] == result["base_weights_sha256"], "Invalid graph selection lock")
        require(config["configurations"] == lock["selected"], "Graph inference config differs from test lock")
    else:
        lock = config
        require(lock["status"] == "locked" and lock["test_accessed"] is False, "Missing multi-view lock")
        require(lock["checkpoint_sha256"] == result["base_weights_sha256"], "Multi-view lock backbone mismatch")
        require(file_hash(directory / "multiview_configuration.json") == result["evaluation_plan"]["selection_lock_sha256"], "Multi-view lock bytes changed")
        require(datetime.fromisoformat(lock["locked_utc"]) <= datetime.fromisoformat(result["completed_utc"]), "Invalid multi-view chronology")
        require(graph is not None, "Multi-view needs paired graph reference")
    count = core["protocol"]["test_episodes_per_shot"]
    rows = index_rows(result["rows"], {name}, count)
    records = read_json(directory, name + "_episodes.json")["episodes"]
    require(len(records) == count * 2, "Incomplete extension task records")
    index = {}
    for record in records:
        shots, number = record["shots"], record["episode"]
        require(type(shots) is int and type(number) is int and (shots,number) in reference, "Invalid extension task index")
        key = (shots, number)
        require(key not in index, "Duplicate extension task")
        old = reference[key]
        require(record["support_ids"] == old["support_ids"] and record["query_ids"] == old["query_ids"], "Unpaired extension image IDs")
        require(record["query_labels"] == old["query_labels"], "Extension scoring labels changed")
        labels = predictions(record, record["query_labels"], 75)
        if name == "multiview":
            selection = lock["selected"][str(shots)]
            require(selection["assignment"] in ("soft", "balanced_capacity") and selection["views"] in ("original", "mean", "concat"), "Invalid inference rule")
            require(rows[(name,shots)]["decision_rule"] == selection["assignment"] and rows[(name,shots)]["views"] == selection["views"], "Inference/test mode mismatch")
            probabilities = record["probabilities"]
            require(len(probabilities) == 75, "Invalid probability length")
            for index_in_task, values in enumerate(probabilities):
                require(len(values) == 5 and all(type(x) in (int,float) and math.isfinite(x) and 0 <= x <= 1 for x in values), "Invalid probability values")
                close(sum(values), 1, "probability sum")
                if selection["assignment"] == "soft":
                    require(labels[index_in_task] == max(range(5), key=lambda c:values[c]), "Prediction not score argmax")
            if selection["assignment"] == "balanced_capacity":
                require(all(labels.count(c) == 15 for c in range(5)), "Capacity constraint violated")
            soft = predictions(record, record["query_labels"], 75, "soft_predictions", "soft_accuracy")
            require(soft == [max(range(5), key=lambda c:row[c]) for row in probabilities], "Soft ablation prediction differs from scores")
            close(record["baseline_accuracy"], old["scores"]["baseline"], "baseline reference")
            close(record["full_accuracy"], old["scores"]["full"], "full reference")
            close(record["graph_accuracy"], graph[key]["accuracy"], "graph reference")
            require(finite(record["seconds"], "duration") >= 0, "Negative extension duration")
        else:
            close(record["v2_accuracy"], old["scores"]["full"], "graph v2 reference")
        index[key] = record
    for shots in (1,5):
        tasks = [index[(shots,i)] for i in range(count)]; row = rows[(name,shots)]
        values = [r["accuracy"] for r in tasks]
        summary(values, row, count, name)
        paired(values, [reference[(shots,i)]["scores"]["baseline"] for i in range(count)], row, "paired_delta", "paired_ci95")
        paired(values, [reference[(shots,i)]["scores"]["full"] for i in range(count)], row, "paired_vs_full", "paired_vs_full_ci95")
        if name == "multiview":
            paired(values, [graph[(shots,i)]["accuracy"] for i in range(count)], row, "paired_vs_graph", "paired_vs_graph_ci95")
            close(row["soft_accuracy"], statistics.mean(r["soft_accuracy"] for r in tasks), "soft ablation mean")
            close(row["mean_episode_seconds"], statistics.mean(r["seconds"] for r in tasks), "multi-view duration")
    return result, index


def audit_wide(directory, core, reference, graph, multiview):
    if not (directory / "wide_results.json").is_file():
        return None
    result = read_json(directory, "wide_results.json")
    require(result.get("status") == "complete", "Wider study is incomplete")
    require(graph and multiview, "Wider study requires both frozen extension references")
    require(set(result["files"]) == set(_WIDE[1:]), "Incomplete wider publication dependencies")
    for name, entry in result["files"].items():
        require(entry["path"] == name, "Unexpected wider file path")
        validate_hash(directory, name, entry["sha256"])
    require(result["files"]["wide.pt"]["sha256"] == result["candidate_checkpoint_sha256"], "Wider weights not the tested model")
    lock = read_json(directory, "wide_configuration.json")
    require(lock["status"] == "locked" and lock["test_accessed"] is False, "Wider selection not locked")
    require(lock["checkpoint_sha256"] == result["candidate_checkpoint_sha256"], "Wider selected checkpoint mismatch")
    require(file_hash(directory / "wide_configuration.json") == result["selection_lock_sha256"], "Wider lock changed")
    require(file_hash(directory / "wide_validation.json") == lock["selection_sha256"], "Wider validation record changed")
    require(file_hash(directory / "wide_training_plan.json") == lock["training_plan_sha256"], "Wider training plan changed")
    require(datetime.fromisoformat(lock["locked_utc"]) <= datetime.fromisoformat(result["completed_utc"]), "Wider test chronology")
    selection = read_json(directory, "wide_validation.json")
    require(selection["status"] == "complete" and selection["test_accessed"] is False, "Unfinished wider selection")
    for shots in ("1", "5"):
        chosen = selection["shots"][shots]["selected"]
        require(lock["selected"][shots] == {k:chosen[k] for k in ("space", "config")}, "Wider inference/test config mismatch")
    training = read_json(directory, "wide_training.json")
    require(training["status"] == "complete" and training["test_accessed"] is False, "Wider training incomplete")
    require(result["training"]["selected_epoch"] == training["best_epoch"], "Wrong wider selected epoch")
    require(result["training"]["epochs_completed"] == len(training["rows"]), "Wrong wider training budget")
    plan = result["evaluation_plan"]
    require(plan["selection_lock_sha256"] == result["selection_lock_sha256"], "Wrong wider evaluation lock")
    require(plan["baseline_checkpoint_sha256"] == core["weights"]["baseline"]["sha256"], "Wider baseline reference changed")
    require(plan["seed"] == core["protocol"]["seed"], "Wider task seed changed")
    require(plan["validation_reference_sha256"] == file_hash(directory/"wide_validation.json"), "Wider validation reference mismatch")
    records = read_json(directory, "wide_episodes.json")["episodes"]
    count = core["protocol"]["test_episodes_per_shot"]
    require(len(records) == count * 2, "Incomplete wider task set")
    rows = index_rows(result["rows"], {"wide"}, count)
    index = {}
    for record in records:
        shots, number = record["shots"], record["episode"]
        require(type(shots) is int and type(number) is int and (shots,number) in reference, "Invalid wider task ID")
        key = (shots, number)
        require(key not in index, "Duplicate wider task")
        previous = reference[key]
        require(record["support_ids"] == previous["support_ids"] and record["query_ids"] == previous["query_ids"], "Wider samples not paired")
        require(record["query_labels"] == previous["query_labels"], "Wider labels changed")
        pred = predictions(record, record["query_labels"], 75)
        predictions(record, record["query_labels"], 75, "raw_predictions", "raw_accuracy")
        require(len(record["probabilities"]) == 75, "Invalid wider model score rows")
        for y, values in zip(pred, record["probabilities"]):
            require(len(values) == 5 and all(type(v) in (int,float) and math.isfinite(v) and 0 <= v <= 1 for v in values), "Invalid wider probabilities")
            close(sum(values), 1, "wider score sum")
            require(y == max(range(5),key=lambda c:values[c]), "Wider label differs from maximum score")
        close(record["baseline_accuracy"], previous["scores"]["baseline"], "wider baseline reference")
        close(record["full_accuracy"], previous["scores"]["full"], "wider full reference")
        close(record["graph_accuracy"], graph[key]["accuracy"], "wider graph reference")
        close(record["multiview_accuracy"], multiview[key]["accuracy"], "wider multi-view reference")
        require(finite(record["seconds"], "wider time") >= 0, "Negative wider duration")
        index[key] = record
    for shots in (1,5):
        tasks = [index[(shots,i)] for i in range(count)]
        row = rows[("wide", shots)]
        values = [r["accuracy"] for r in tasks]
        summary(values, row, count, "wide")
        for field, mean_key, ci_key in [("baseline_accuracy","paired_delta","paired_ci95"),
                ("full_accuracy","paired_vs_full","paired_vs_full_ci95"),("graph_accuracy","paired_vs_graph","paired_vs_graph_ci95"),
                ("multiview_accuracy","paired_vs_multiview","paired_vs_multiview_ci95")]:
            paired(values, [r[field] for r in tasks], row, mean_key, ci_key)
        close(row["raw_accuracy"], statistics.mean(r["raw_accuracy"] for r in tasks), "wider raw backbone score")
        close(row["mean_episode_seconds"], statistics.mean(r["seconds"] for r in tasks), "wider mean duration")
    return result


def audit_timing(directory, core, wide, revision, continued=None):
    if not (directory / "edge_benchmark.json").is_file():
        return None
    timing = read_json(directory, "edge_benchmark.json")
    methods = {"baseline":"baseline", "maml":"maml", "calibrated":"aligned",
               "graph":"aligned", "multiview":"aligned", "wide":"wide", "wideview":"wide", "continued":"continued"}
    seen = set()
    def percentile(values, q):
        ordered = sorted(values);position=(len(ordered)-1)*q
        low,high=math.floor(position),math.ceil(position)
        return ordered[low] + (ordered[high]-ordered[low])*(position-low)
    for row in timing["rows"]:
        key=(row["method"], row["shots"])
        require(key not in seen and row["method"] in methods and type(row["shots"]) is int and row["shots"] in (1,5), "Invalid timing method/shot")
        seen.add(key)
        n=row["repetitions"]
        require(type(n) is int and n>=3, "Insufficient timing repeats")
        model=methods[row["method"]]
        expected=continued["checkpoint_sha256"] if model=="continued" and continued else wide["candidate_checkpoint_sha256"] if model=="wide" and wide else core["weights"].get(model,{}).get("sha256")
        require(expected is not None and row["weights_sha256"]==expected, "Timing belongs to different weights")
        if "artifact_revision" in row:
            require(row["artifact_revision"]==revision, "Timing belongs to another publication revision")
        for values_name,median_name,p95_name in [("wall_samples_ms","warm_wall_median_ms","warm_wall_p95_ms"),
            ("compute_samples_ms","encoding_and_classification_median_ms","encoding_and_classification_p95_ms")]:
            values=row[values_name]
            require(len(values)==n and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values), "Invalid raw timing samples")
            close(row[median_name],statistics.median(values),median_name)
            close(row[p95_name],percentile(values,.95),p95_name)
        require(finite(row["cold_wall_ms"],"cold timing")>=0,"Negative cold timing")
    return timing


def prediction_diagnostics(records, graph):
    """Descriptive task/score reliability; never used for prediction or tuning."""
    rows = []
    for shots in (1,5):
        tasks = sorted((r for (s,_),r in records.items() if s == shots), key=lambda r:r["episode"])
        if not tasks:
            continue
        bins = [{"lower":i/10,"upper":(i+1)/10,"occurrences":0,"score_sum":0.,"correct":0} for i in range(10)]
        brier = nll = 0.
        occurrences = 0
        for task in tasks:
            for scores, truth, predicted in zip(task["probabilities"], task["query_labels"], task["predictions"]):
                confidence = scores[predicted]
                index = min(int(confidence * 10), 9)
                bucket = bins[index]
                bucket["occurrences"] += 1
                bucket["score_sum"] += confidence
                bucket["correct"] += int(truth == predicted)
                brier += sum((score - int(i == truth))**2 for i,score in enumerate(scores))
                nll -= math.log(max(scores[truth], 1e-15))
                occurrences += 1
        ece = 0.
        for bucket in bins:
            n = bucket["occurrences"]
            bucket["mean_score"] = bucket.pop("score_sum") / n if n else None
            bucket["accuracy"] = bucket.pop("correct") / n if n else None
            if n:
                ece += n / occurrences * abs(bucket["mean_score"] - bucket["accuracy"])
        values = sorted(task["accuracy"] for task in tasks)
        def quantile(q):
            position = (len(values)-1)*q
            low,high = math.floor(position),math.ceil(position)
            return values[low] + (values[high]-values[low])*(position-low)
        difference = [task["accuracy"] - graph[(shots,task["episode"])]["accuracy"] for task in tasks]
        rows.append({"shots":shots,"episodes":len(tasks),"query_occurrences":occurrences,
            "task_p05":quantile(.05),"task_median":quantile(.5),"task_p95":quantile(.95),
            "paired_wins":sum(x>0 for x in difference),"paired_ties":sum(x==0 for x in difference),
            "paired_losses":sum(x<0 for x in difference),"score_bins":bins,
            "ece10":ece,"brier_score":brier/occurrences,"negative_log_likelihood":nll/occurrences})
    return {"method":"multiview","reference":"graph","rows":rows,
            "scope":"固定预测后的描述性诊断，不用于选型。查询图片跨任务可重复，因此按任务出现次数加权。"}


@lru_cache(maxsize=3)
def _cached(directory_name, signature):
    directory = Path(directory_name)
    core, reference, ready = audit_core(directory)
    from .module_contrasts import summarize_contrasts
    contrasts = summarize_contrasts(reference)
    graph, graph_records = audit_extension(directory, "graph", core, reference)
    multi, multi_records = audit_extension(directory, "multiview", core, reference, graph_records if graph is not None else None)
    diagnostics = prediction_diagnostics(multi_records, graph_records) if multi else None
    wide = audit_wide(directory, core, reference, graph_records, multi_records)
    from .wideview_audit import audit_wideview
    wideview = audit_wideview(directory, core, reference, graph_records, multi_records, wide)
    from .continued_audit import audit_continued
    continued = audit_continued(directory, core, reference, wide, wideview)
    revision_bytes = (directory / "results.json").read_bytes()
    for name in ("graph_results.json", "multiview_results.json", "wide_results.json", "wideview_results.json", "continued_results.json"):
        if (directory / name).is_file():
            revision_bytes += (directory / name).read_bytes()
    revision = hashlib.sha256(revision_bytes).hexdigest()
    timing = audit_timing(directory, core, wide, revision, continued)
    require(observations(directory) == signature, "Artifacts changed during verification; retry after publication")
    return {"core":core, "graph":graph, "multiview":multi, "wide":wide, "wideview":wideview, "continued":continued, "ready":ready, "diagnostics":diagnostics,
            "artifact_revision":revision,"edge_benchmark":timing,"module_contrasts":contrasts,
            "verified":True,"episodes_recomputed":len(reference),
            "extension_episodes_recomputed":(len(reference) if graph else 0)+(len(reference) if multi else 0)+(len(reference) if wide else 0)+(len(reference) if wideview else 0)+(len(reference) if continued else 0)}


def audit_publication(directory):
    directory = Path(directory).resolve()
    with _LOCK:
        return deepcopy(_cached(str(directory), observations(directory)))


def clear_audit_cache():
    with _LOCK:
        _cached.cache_clear()
