"""Pre-registered validation study: TIM/PT-MAP on locked cached features.

No test-data path or auto-publication exists. All inputs, tasks, candidates and
selection rules are frozen before any scoring. The independent capacity probe
uses exactly paired images and identical heads; external provenance is explicit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .calibration import unit_features
from .data import digest_file, sample_episode
from .graph_calibration import GraphConfig, apply_transform, predict as graph_predict
from .literature_methods import (
    MAPConfig, TIMConfig, map_predict_batch, prepare_episode_batch, tim_predict_batch,
)
from .multiview import combine_views


SEED = 20261130
SEARCH_EPISODES = 160
CONFIRM_EPISODES = 600
DIAGNOSTIC_EPISODES = 600
CANDIDATES = (
    {"id": "map_whitened", "algorithm": "map", "space": "whitened", "task_center": False, "signed_power": None},
    {"id": "map_taskcenter_whitened", "algorithm": "map", "space": "whitened", "task_center": True, "signed_power": None},
    {"id": "ptmap_signed_taskcenter", "algorithm": "map", "space": "raw", "task_center": True, "signed_power": .5},
    {"id": "tim_whitened", "algorithm": "tim", "space": "whitened", "task_center": False, "signed_power": None},
    {"id": "tim_taskcenter_whitened", "algorithm": "tim", "space": "whitened", "task_center": True, "signed_power": None},
)
DIAGNOSTIC_IDS = (
    "continued_published", "student_centered_proto", "teacher_centered_proto",
    "student_centered_graph", "teacher_centered_graph", "student_ptmap", "teacher_ptmap",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    temporary.replace(path)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def summary(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("At least two finite episode scores required")
    mean, half = float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))
    return {"accuracy": mean, "ci95": half, "episodes": len(x),
            "ci95_low": mean - half, "ci95_high": mean + half}


def paired(values, reference):
    x, y = np.asarray(values), np.asarray(reference)
    if x.shape != y.shape:
        raise ValueError("Unpaired score vectors")
    s = summary(x - y)
    return {"mean": s["accuracy"], "ci95": s["ci95"], "low": s["ci95_low"], "high": s["ci95_high"],
            "wins": int(np.sum(x > y)), "ties": int(np.sum(x == y)), "losses": int(np.sum(x < y))}


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def validate_pool(pool, split, classes, per_class):
    ids, labels = pool["ids"], pool["labels"]
    if ids.ndim != 1 or labels.ndim != 1 or len(ids) != len(labels) or len(set(ids.tolist())) != len(ids):
        raise ValueError("Invalid or duplicate cache IDs")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Class labels must be integers")
    if not all(str(i).startswith(split + "/") for i in ids):
        raise ValueError("Cache split mismatch: train/val only")
    names = sorted({str(i).split("/")[1] for i in ids})
    if len(names) != classes or not np.array_equal(np.unique(labels), np.arange(classes)):
        raise ValueError("Class count mismatch")
    if not np.array_equal(np.bincount(labels), np.full(classes, per_class)):
        raise ValueError("Unexpected samples per class")
    if any(str(i).split("/")[1] != names[int(y)] for i, y in zip(ids, labels)):
        raise ValueError("Cache class/ID mapping mismatch")
    for key in ("features", "original", "flipped"):
        if key in pool and (pool[key].ndim != 2 or len(pool[key]) != len(ids) or not np.isfinite(pool[key]).all()):
            raise ValueError("Invalid cached features")


def aligned_indices(full, subset):
    index = {str(identity): i for i, identity in enumerate(full["ids"])}
    try:
        indices = np.array([index[str(i)] for i in subset["ids"]], dtype=int)
    except KeyError as exc:
        raise ValueError("External cache ID not in the local train/validation split") from exc
    if not np.array_equal(full["labels"][indices], subset["labels"]):
        raise ValueError("Paired cache label mappings disagree")
    return indices


def inputs(args):
    cache, artifacts, external = Path(args.cache), Path(args.artifacts), Path(args.external)
    pools = {split: load_npz(cache / f"{split}_views.npz") for split in ("train", "val")}
    ext = {split: load_npz(external / f"{split}_features.npz") for split in ("train", "val")}
    for split, count in (("train", 64), ("val", 16)):
        validate_pool(pools[split], split, count, 600)
        validate_pool(ext[split], split, count, 100 if split == "train" else 200)
        aligned_indices(pools[split], ext[split])
        stamp = read(cache / f"{split}_views.json")
        if stamp["identity"]["split"] != split or stamp["sha256"] != digest_file(cache / f"{split}_views.npz"):
            raise ValueError("Student cache stamp mismatch")
        if stamp["identity"]["checkpoint_sha256"] != digest_file(artifacts / "continued.pt"):
            raise ValueError("Student cache checkpoint mismatch")
    train_names = {str(i).split("/")[1] for i in pools["train"]["ids"]}
    val_names = {str(i).split("/")[1] for i in pools["val"]["ids"]}
    if train_names & val_names:
        raise ValueError("Train/validation class overlap")
    provenance = read(external / "provenance.json")
    if provenance["sha256"] != digest_file(external / "feat-1-shot.pth"):
        raise ValueError("External checkpoint provenance hash mismatch")
    files = [cache / f"{split}_{suffix}" for split in ("train", "val") for suffix in ("views.npz", "views.json")]
    files += [external / f"{split}_features.npz" for split in ("train", "val")]
    files += [external / "provenance.json", external / "feat-1-shot.pth"]
    files += [artifacts / f"continued{suffix}" for suffix in (".pt", "_configuration.json", "_transform.npz")]
    return pools, ext, {str(p): digest_file(p) for p in files}, provenance


def make_tasks(pools, ext):
    rng = np.random.default_rng(SEED)
    search, confirm = [], []
    labels = pools["val"]["labels"]
    for cls in range(16):
        order = rng.permutation(np.flatnonzero(labels == cls))
        search.extend(order[:200]); confirm.extend(order[200:])
    search, confirm = np.array(search), np.array(confirm)
    if set(search) & set(confirm):
        raise ValueError("Search and confirmation images overlap")
    groups = {}
    for phase, indices, count, offset in (("search", search, SEARCH_EPISODES, 100),
                                          ("confirmation", confirm, CONFIRM_EPISODES, 200)):
        for shot in (1, 5):
            generator = np.random.default_rng(SEED + offset + shot)
            tasks = []
            for _ in range(count):
                task = sample_episode(labels[indices], generator, shots=shot, queries=15)
                task["support_indices"] = indices[task["support_indices"]]
                task["query_indices"] = indices[task["query_indices"]]
                tasks.append({k: v.tolist() for k, v in task.items()})
            groups[f"{phase}/{shot}"] = tasks
    for shot in (1, 5):
        generator = np.random.default_rng(SEED + 300 + shot)
        groups[f"diagnostic/{shot}"] = [
            {k: v.tolist() for k, v in sample_episode(ext["val"]["labels"], generator, shots=shot, queries=15).items()}
            for _ in range(DIAGNOSTIC_EPISODES)]
    return {"student_val_ids": pools["val"]["ids"].tolist(), "external_val_ids": ext["val"]["ids"].tolist(),
            "search_indices": search.tolist(), "confirmation_indices": confirm.tolist(), "groups": groups}


def register(args, input_hashes, provenance, tasks):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True); (out / "scores").mkdir(exist_ok=True)
    source_names = ("study_literature.py", "literature_methods.py", "graph_calibration.py", "multiview.py", "calibration.py", "data.py")
    plan = {"schema_version": 1, "kind": "validation_only_literature_study", "seed": SEED,
            "test_accessed": False, "external_reference": provenance, "inputs": input_hashes,
            "source_hashes": {n: digest_file(Path(__file__).with_name(n)) for n in source_names},
            "tasks_content_sha256": digest_json(tasks), "candidates": list(CANDIDATES),
            "map_config": asdict(MAPConfig()), "tim_config": asdict(TIMConfig()),
            "search_episodes": SEARCH_EPISODES, "confirmation_episodes": CONFIRM_EPISODES,
            "diagnostic_episodes": DIAGNOSTIC_EPISODES, "diagnostic_methods": list(DIAGNOSTIC_IDS),
            "shortlist": 2, "minimum_gain": .01,
            "selection_rule": "Per shot: top 2 search means; select maximal confirmation paired lower 95% bound, ID as tie-break. Eligible only if both paired lower bounds >0 and both means >=1 percentage point. Never auto-test/publish.",
            "limitations": ["Validation images/classes have already been reused by earlier studies; not untouched data.",
                "Search/confirmation image pools are disjoint, but share 16 classes; confirmation participates in selection.",
                "Intervals are pointwise paired episode intervals, conditional on reused classes, not multiplicity-corrected or class-population intervals.",
                "External FEAT encoder checkpoint is author-provided, not locally trained; caches are inherited and fingerprinted here.",
                "Capacity probe jointly changes capacity, normalization and original training; it is not a capacity-only causal ablation.",
                "TIM promotes uniform marginal entropy; MAP and published graph assume balanced queries.",
                "Published continued transform uses 38400 train images; diagnostic matched-center heads use identical 6400 train IDs for both models."]}
    plan_path, task_path = out / "plan.json", out / "tasks.json"
    if plan_path.exists():
        old = read(plan_path)
        comparable = {k: v for k, v in old.items() if k != "registered_utc"}
        if comparable != plan or not task_path.exists() or read(task_path) != tasks:
            raise ValueError("Registered inputs/source/tasks changed. Preserve this study; use a new directory.")
    else:
        if any((out / "scores").iterdir()) or task_path.exists():
            raise ValueError("Unregistered partial output; inspect rather than overwrite")
        write(task_path, tasks)
        write(plan_path, {**plan, "registered_utc": datetime.now(timezone.utc).isoformat()})
    return read(plan_path)


def validate_scores(record, group, candidate_id, tasks, plan_sha):
    if record.get("plan_sha256") != plan_sha or record.get("group") != group or record.get("candidate") != candidate_id:
        raise ValueError("Score identity mismatch")
    predictions = np.asarray(record["predictions"])
    truth = np.asarray([task["query_labels"] for task in tasks])
    if predictions.shape != truth.shape or not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("Incomplete or noninteger predictions")
    if np.any(predictions < 0) or np.any(predictions >= 5):
        raise ValueError("Predictions outside task class range")
    values = (predictions == truth).mean(1)
    if not np.allclose(values, record["accuracies"], atol=1e-14, rtol=0):
        raise ValueError("Cached accuracy differs from raw predictions")
    return values.tolist()


def score(args, plan_sha, group, candidate, features, tasks, graph_config):
    out = Path(args.output)
    candidate_id = candidate["id"]
    path = out / "scores" / f"{group.replace('/', '_')}_{candidate_id}.json"
    if path.exists():
        return validate_scores(read(path), group, candidate_id, tasks, plan_sha)
    predictions, diagnostics = [], []
    started = time.perf_counter()
    for offset in range(0, len(tasks), 40):
        batch = tasks[offset:offset + 40]
        # Deliberately extract only support labels BEFORE calling predictors.
        si = np.asarray([task["support_indices"] for task in batch])
        qi = np.asarray([task["query_indices"] for task in batch])
        ys = np.asarray([task["support_labels"] for task in batch])
        support, query = features[si], features[qi]
        if candidate["algorithm"] == "graph":
            predicted = np.stack([graph_predict(s, y, q, graph_config)["predictions"]
                                  for s, y, q in zip(support, ys, query)])
        elif candidate["algorithm"] == "proto":
            centers = np.stack([np.stack([s[y == cls].mean(0) for cls in range(5)])
                                for s, y in zip(support, ys)])
            distances = ((query[:, :, None] - centers[:, None]) ** 2).sum(-1)
            predicted = distances.argmin(2)
        else:
            support, query = prepare_episode_batch(support, query, signed_power=candidate.get("signed_power"),
                                                    task_center=candidate.get("task_center", False))
            fn, cfg = (map_predict_batch, MAPConfig()) if candidate["algorithm"] == "map" else (tim_predict_batch, TIMConfig())
            result = fn(support, ys, query, cfg)
            predicted = result["predictions"]
            diagnostics.append({k: result[k] for k in ("max_transport_residual", "objective_before", "objective_after") if k in result})
        predictions.extend(predicted.tolist())
    # Query truth is used exclusively here, AFTER every prediction is produced.
    truth = np.asarray([task["query_labels"] for task in tasks])
    values = (np.asarray(predictions) == truth).mean(1).tolist()
    record = {"plan_sha256": plan_sha, "group": group, "candidate": candidate_id,
              "predictions": predictions, "accuracies": values, "test_accessed": False,
              "cached_head_seconds": time.perf_counter() - started, "diagnostics": diagnostics}
    validate_scores(record, group, candidate_id, tasks, plan_sha)
    write(path, record)
    print(json.dumps({"group": group, "candidate": candidate_id, **summary(values),
                      "cached_head_seconds": record["cached_head_seconds"]}), flush=True)
    return values


def spaces(args, pools):
    artifacts = Path(args.artifacts)
    raw = combine_views(pools["val"]["original"], pools["val"]["flipped"], "mean")
    cfg = read(artifacts / "continued_configuration.json")["selected"]
    transform_data = load_npz(artifacts / "continued_transform.npz")
    result = {}
    for shot in (1, 5):
        prefix = f"shot{shot}_"
        transform = {key[len(prefix):]: value for key, value in transform_data.items() if key.startswith(prefix)}
        result[shot] = {"raw": raw, "whitened": apply_transform(raw, transform)}
    return result, cfg


def select(args, plan, tasks, feature_spaces, configurations):
    plan_sha = digest_file(Path(args.output) / "plan.json")
    result = {"status": "complete", "test_accessed": False, "plan_sha256": plan_sha,
              "shots": {}, "limitations": plan["limitations"]}
    for shot in (1, 5):
        graph_cfg = GraphConfig(**configurations[str(shot)]["config"])
        values = {}
        references = {}
        for phase in ("search", "confirmation"):
            group = f"{phase}/{shot}"
            references[phase] = score(args, plan_sha, group, {"id": "continued_published", "algorithm": "graph"},
                                      feature_spaces[shot]["whitened"], tasks["groups"][group], graph_cfg)
        for candidate in CANDIDATES:
            values[candidate["id"]] = score(args, plan_sha, f"search/{shot}", candidate,
                feature_spaces[shot][candidate["space"]], tasks["groups"][f"search/{shot}"], graph_cfg)
        ordered = sorted(CANDIDATES, key=lambda c: (-np.mean(values[c["id"]]), c["id"]))
        rows = [{"id": c["id"], "search": summary(values[c["id"]]),
                 "search_paired": paired(values[c["id"]], references["search"])} for c in ordered]
        for row in rows[:2]:
            candidate = next(c for c in CANDIDATES if c["id"] == row["id"])
            scores = score(args, plan_sha, f"confirmation/{shot}", candidate,
                feature_spaces[shot][candidate["space"]], tasks["groups"][f"confirmation/{shot}"], graph_cfg)
            row["confirmation"] = summary(scores)
            row["confirmation_paired"] = paired(scores, references["confirmation"])
        selected = min(rows[:2], key=lambda r: (-r["confirmation_paired"]["low"], r["id"]))
        eligible = selected["confirmation_paired"]["low"] > 0 and selected["confirmation_paired"]["mean"] >= .01
        result["shots"][str(shot)] = {"candidates": rows, "selected": selected["id"], "eligible": eligible,
                                     "reference": {p: summary(v) for p, v in references.items()}}
    result["eligible"] = all(x["eligible"] for x in result["shots"].values())
    result["publication"] = "not_published; validation-only; no automatic test or release"
    write(Path(args.output) / "selection.json", result)
    return result


def diagnose(args, plan, tasks, pools, external, feature_spaces, configurations):
    plan_sha = digest_file(Path(args.output) / "plan.json")
    val_ids = aligned_indices(pools["val"], external["val"])
    train_ids = aligned_indices(pools["train"], external["train"])
    student_train = combine_views(pools["train"]["original"][train_ids], pools["train"]["flipped"][train_ids], "mean")
    student_val = feature_spaces[1]["raw"][val_ids]
    teacher_train, teacher_val = unit_features(external["train"]["features"]), unit_features(external["val"]["features"])
    vectors = {"student_raw": student_val, "teacher_raw": teacher_val,
               "student_centered": unit_features(student_val - student_train.mean(0)),
               "teacher_centered": unit_features(teacher_val - teacher_train.mean(0))}
    result = {"status": "complete", "test_accessed": False, "plan_sha256": plan_sha, "shots": {},
              "external_checkpoint": plan["external_reference"], "student_parameters": 509856,
              "external_encoder_parameters": 12424320, "paired_val_images": 3200, "paired_center_train_images": 6400,
              "scope": "validation reference diagnostic, NOT a public test result or a locally trained achievement",
              "limitations": plan["limitations"]}
    for shot in (1, 5):
        graph_cfg = GraphConfig(**configurations[str(shot)]["config"])
        values = {}
        for name in DIAGNOSTIC_IDS:
            if name == "continued_published":
                spec = {"id": name, "algorithm": "graph"}; x = feature_spaces[shot]["whitened"][val_ids]
            elif name.endswith("ptmap"):
                spec = {"id": name, "algorithm": "map", "task_center": True, "signed_power": .5}
                x = vectors[name.split("_")[0] + "_raw"]
            else:
                spec = {"id": name, "algorithm": name.split("_")[-1]}
                x = vectors[name.split("_")[0] + "_centered"]
            values[name] = score(args, plan_sha, f"diagnostic/{shot}", spec, x, tasks["groups"][f"diagnostic/{shot}"], graph_cfg)
        result["shots"][str(shot)] = {"methods": {name: summary(v) for name, v in values.items()},
            "matched_head_pairs": {head: paired(values[f"teacher_{head}"], values[f"student_{head}"])
                                    for head in ("centered_proto", "centered_graph", "ptmap")},
            "teacher_vs_published": paired(values["teacher_centered_graph"], values["continued_published"])}
    write(Path(args.output) / "diagnostic.json", result)
    return result


def audit(args, plan, tasks):
    out = Path(args.output); plan_sha = digest_file(out / "plan.json")
    results = {}
    for path in sorted((out / "scores").glob("*.json")):
        row = read(path)
        if row.get("test_accessed") is not False or row["group"] not in tasks["groups"]:
            raise ValueError("Unknown score group")
        key = (row["group"], row["candidate"])
        if key in results:
            raise ValueError("Duplicate scores")
        results[key] = validate_scores(row, *key, tasks["groups"][row["group"]], plan_sha)
    checks = {"score_files": len(results), "episode_predictions": sum(len(v) for v in results.values()),
              "test_accessed": False, "plan_sha256": plan_sha, "inputs_unchanged": True,
              "search_confirmation_disjoint": not (set(tasks["search_indices"]) & set(tasks["confirmation_indices"]))}
    selection_path = out / "selection.json"
    if selection_path.exists():
        selection = read(selection_path)
        eligible_shots = []
        for shot in (1, 5):
            ref = {phase: results[(f"{phase}/{shot}", "continued_published")] for phase in ("search", "confirmation")}
            rank = sorted([c["id"] for c in CANDIDATES], key=lambda name: (-np.mean(results[(f"search/{shot}", name)]), name))
            rows = selection["shots"][str(shot)]["candidates"]
            if [r["id"] for r in rows] != rank:
                raise ValueError("Search ranking was changed")
            for row in rows:
                scores = results[(f"search/{shot}", row["id"])]
                if row["search"] != summary(scores) or row["search_paired"] != paired(scores, ref["search"]):
                    raise ValueError("Search summary mismatch")
                if row["id"] in rank[:2]:
                    scores = results[(f"confirmation/{shot}", row["id"])]
                    if row["confirmation"] != summary(scores) or row["confirmation_paired"] != paired(scores, ref["confirmation"]):
                        raise ValueError("Confirmation summary mismatch")
            winner = min(rows[:2], key=lambda r: (-r["confirmation_paired"]["low"], r["id"]))
            eligible = winner["confirmation_paired"]["low"] > 0 and winner["confirmation_paired"]["mean"] >= .01
            recorded = selection["shots"][str(shot)]
            if recorded["selected"] != winner["id"] or recorded["eligible"] is not eligible:
                raise ValueError("Selection gate mismatch")
            if recorded["reference"] != {p: summary(v) for p, v in ref.items()}:
                raise ValueError("Reference summary mismatch")
            eligible_shots.append(eligible)
        if selection["eligible"] is not all(eligible_shots):
            raise ValueError("Overall eligibility mismatch")
        checks["selection_recomputed"] = True
    diagnostic_path = out / "diagnostic.json"
    if diagnostic_path.exists():
        diagnostic = read(diagnostic_path)
        for shot in (1, 5):
            values = {name: results[(f"diagnostic/{shot}", name)] for name in DIAGNOSTIC_IDS}
            row = diagnostic["shots"][str(shot)]
            if row["methods"] != {n: summary(v) for n, v in values.items()}:
                raise ValueError("Diagnostic summary mismatch")
            if row["matched_head_pairs"] != {h: paired(values[f"teacher_{h}"], values[f"student_{h}"])
                                               for h in ("centered_proto", "centered_graph", "ptmap")}:
                raise ValueError("Diagnostic paired comparison mismatch")
            if row["teacher_vs_published"] != paired(values["teacher_centered_graph"], values["continued_published"]):
                raise ValueError("Diagnostic reference mismatch")
        checks["diagnostic_recomputed"] = True
    checks["status"] = "passed"
    checks["files"] = {str(p.relative_to(out)): digest_file(p) for p in sorted(out.rglob("*.json")) if p.name != "audit.json"}
    write(out / "audit.json", checks)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=".local/lab04-continuation-validation-20261125")
    parser.add_argument("--external", default=".local/lab04-reference-feat")
    parser.add_argument("--artifacts", default="outputs/lab04")
    parser.add_argument("--output", default="outputs/lab04/next_study/literature_inference_20261125")
    parser.add_argument("--phase", choices=("register", "select", "diagnose", "all", "audit"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        pools, external, hashes, provenance = inputs(args)
        tasks = make_tasks(pools, external)
        plan = register(args, hashes, provenance, tasks)
        print(json.dumps({"phase": args.phase, "registered": True, "test_accessed": False}), flush=True)
        if args.phase == "register":
            return
        if args.phase != "audit":
            features, cfg = spaces(args, pools)
            if args.phase in ("select", "all"):
                select(args, plan, tasks, features, cfg)
            if args.phase in ("diagnose", "all"):
                diagnose(args, plan, tasks, pools, external, features, cfg)
        print(json.dumps(audit(args, plan, tasks), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
