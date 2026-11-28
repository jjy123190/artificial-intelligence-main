"""Independently rescore the release from ordered image IDs and predictions (standard library only)."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics as st


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(row) for row in f]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12)


def reference(artifacts, name, plan):
    val = name == "val"
    path = artifacts / ("continued_validation_tasks.jsonl" if val else "continued_episodes.json")
    check(sha(path) == plan["episode_sha256"][path.name], f"{name}: reference file changed")
    original = lines(path) if val else read(path)["episodes"]
    targets = {}
    for row in original:
        s, i = row["shots"], row["episode"]
        support, query = row["support_ids"], row["query_ids"]
        # Do not use the evaluation program's dataset indexer, task fingerprint,
        # predictions, or scoring code. Resolve episode labels from the class folders.
        ids = [k.split("/") for k in support + query]
        check(all(len(k) == 3 and k[0] == name for k in ids), "Invalid split or image ID")
        classes = list(dict.fromkeys(p[1] for p in ids[:len(support)]))
        labels = [classes.index(p[1]) for p in ids]
        y = labels[:len(support)]
        truth = labels[len(support):]
        check(len(classes) == 5 and sorted(Counter(y).values()) == [s] * 5 and
              sorted(Counter(truth).values()) == [15] * 5 and len(set(support + query)) == 5*(s+15),
              "Duplicated image or incorrect support/query counts")
        check(truth == row["query_labels"], "Original labels disagree with image IDs")
        serialized = json.dumps([s, i, support, y, query], separators=(",", ":")).encode("utf-8")
        check((s, i) not in targets, "Duplicate reference episode")
        targets[s, i] = hashlib.sha256(serialized).hexdigest(), truth
    check(set(targets) == {(s, i) for s in (1, 5) for i in range(600)}, "Incomplete reference bank")
    return targets


def verify_prediction(r, targets, s, i, adapt=False):
    h, truth = targets[s, i]
    pred = r["predictions"]
    check((r["shots"], r["episode"], r["task_sha256"]) == (s, i, h), "Wrong episode or image order")
    check(len(pred) == 75 and all(type(x) is int and 0 <= x < 5 for x in pred), "Invalid local labels")
    correct = sum(a == b for a, b in zip(pred, truth))
    check(r["correct"] == correct and close(r["accuracy"], correct/75), "Episode accuracy disagrees")
    if adapt:
        for field in ("first_conv_delta_l2", "head_delta_l2"):
            check(math.isfinite(r[field]) and r[field] > 0, "Missing per-task MAML adaptation")
    return correct


def val_scores(directory, name, candidates, budget, targets, lock):
    maml = name == "maml"
    rows = lines(directory / ("val_" + name + ".jsonl"))
    counts = {k: {"1": 0, "5": 0} for k in candidates}
    seen = set()
    for r in rows:
        k = r["epoch" if maml else "candidate"]
        s, i = r["shots"], r["episode"]
        check(k in counts and s in (1, 5) and 0 <= i < budget and (k, s, i) not in seen, "Invalid validation key")
        seen.add((k, s, i))
        counts[k][str(s)] += verify_prediction(r, targets, s, i, maml)
    check(len(seen) == len(candidates) * 2 * budget, "Insufficient validation tasks")
    table = lock["meta_validation" if maml else "calibration_validation"]
    check([r["epoch" if maml else "rule_index"] for r in table] == list(candidates), "Wrong candidates")
    for r in table:
        key = r["epoch" if maml else "rule_index"]
        check(r["correct_by_shot"] == counts[key] and r["total"] == sum(counts[key].values()), "Wrong validation total")
    best = max(candidates, key=lambda k: sum(counts[k].values()))
    check(lock["meta_epoch" if maml else "rule_index"] == best, "Wrong validation choice")
    return best


def validate_training(directory, lock):
    plan = read(directory / "training_plan.json")
    a, b = [read(directory / n) for n in ("ce_history.json", "da_history.json")]
    check(plan["channels"] == [32, 64, 128, 256] and plan["external_weights"] is False, "Wrong backbone")
    shared = plan["supervised"]["shared_epochs"]
    epochs = plan["supervised"]["epochs_total_per_arm"]
    check(len(a) == len(b) == epochs and a[:shared] == b[:shared], "Warmup or budget mismatch")
    for i, (x, y) in enumerate(zip(a, b), start=1):
        check(x["epoch"] == i and x["steps"] == 300 and
              all(x[k] == y[k] for k in ("lr", "draw_sha256", "pixels", "steps", "epoch")), "Unmatched controls")
        weight = plan["supervised"]["alignment_weights"]["da"] if i > shared else 0
        check(x["alignment_weight"] == 0 and y["alignment_weight"] == weight and
              close(x["ce"], x["loss"]) and close(y["ce"] + weight*y["alignment_stat"], y["loss"]),
              "Misreported training intervention")
    m = read(directory / "maml_history.json")
    check(m["ce_sha256"] == lock["files"]["ce.pt"] and lock["files"]["ce.pt"] != lock["files"]["da.pt"],
          "MAML initialization or training weights disagree")
    log, spec = m["epochs"], plan["maml"]
    check([x["epoch"] for x in log] == list(range(1, spec["epochs"]+1)) and
          all(x["tasks"] == spec["train_tasks_per_epoch"] and x["outer_grad_norm_preclip"] > 0 and
              x["outer_steps"]*spec["outer_tasks_per_update"] == x["tasks"] for x in log),
          "Missing meta-optimization")
    return plan


def audit(bundle, artifacts, output=None, compare=None):
    pred_dir = output or bundle
    result = read(pred_dir / "results.json")
    plan = read(bundle / "evaluation_plan.json")
    lock = read(bundle / "validation_lock.json")
    check(sha(bundle / "validation_lock.json") == result["selection_lock_sha256"], "Wrong validation lock")
    for base, records in ((bundle, lock["files"]), (pred_dir, result["files"])):
        for key, value in records.items():
            check(Path(key).name == key and sha(base / key) == value, f"Unverified artifact: {key}")
    check(sha(artifacts / "dataset_manifest.json") == plan["data_manifest_sha256"], "Wrong split manifest")
    source = Path(__file__).parent
    for f, h in plan["source_sha256"].items():
        check(sha(source / f) == h, f"Implementation changed: {f}")
    training = validate_training(bundle, lock)
    val = reference(artifacts, "val", plan)
    best_rule = val_scores(bundle, "calibration", list(range(len(plan["rules"]))),
                           plan["val_tasks_per_shot_calibration"], val, lock)
    best_meta = val_scores(bundle, "maml", training["maml"]["candidates"],
                           plan["val_tasks_per_shot_meta_checkpoint"], val, lock)
    check(plan["rules"][best_rule] == lock["rule"], "Selected rule does not match the grid")
    check(next(t["weights_sha256"] for t in lock["meta_validation"] if t["epoch"] == best_meta)
          == lock["files"]["maml.pt"], "Selected meta-checkpoint was not shipped")
    targets = reference(artifacts, "test", plan)
    names = set(plan["methods"]) | {"maml", "maml_no_adapt"}
    check(names == set(result["methods"]), "Missing method or control")
    want_files = {f"test_{name}_{s}shot.jsonl" for name in names for s in (1, 5)}
    check(set(result["files"]) == want_files, "Missing prediction file")
    counts = {}
    total = 0
    for name in sorted(names):
        for s in (1, 5):
            key = f"test_{name}_{s}shot.jsonl"
            rows = lines(pred_dir / key)
            check(len(rows) == 600, "Missing test episode")
            c = [verify_prediction(r, targets, s, i, name == "maml") for i, r in enumerate(rows)]
            if compare:
                old = lines(compare / key)
                check(len(old) == len(rows) and all(
                    x["predictions"] == y["predictions"] and x["task_sha256"] == y["task_sha256"]
                    for x, y in zip(old, rows)), "Replay predictions differ: " + key)
            counts[name, s] = c
            expected = result["methods"][name][str(s)]
            mean = sum(c) / 45000
            ci = 1.96 * st.stdev(c) / (75 * math.sqrt(len(c)))
            check(expected["episodes"] == len(c) and expected["queries"] == 45000 and expected["correct"] == sum(c)
                  and all(close(expected[k], v) for k, v in (("accuracy", mean), ("ci95", ci),
                  ("ci95_low", max(0., mean-ci)), ("ci95_high", min(1., mean+ci)))), f"Wrong summary: {key}")
            total += 75*len(rows)
    required = {"full_minus_" + k for k in ("baseline", "maml", "no_alignment", "no_mean", "no_variance",
                                            "no_update", "no_balance")}
    check(required <= set(result["paired"]), "Missing paired comparisons")
    for key, stats in result["paired"].items():
        a, b = key.split("_minus_")
        for s in (1, 5):
            d = [x-y for x, y in zip(counts[a, s], counts[b, s])]
            z = stats[str(s)]
            check(close(z["mean"], sum(d)/45000) and close(z["ci95"], 1.96*st.stdev(d)/(75*math.sqrt(600)))
                  and (z["wins"], z["ties"], z["losses"]) ==
                  (sum(v>0 for v in d), d.count(0), sum(v<0 for v in d)), f"Wrong paired result: {key}")
    return {"status": "passed", "methods": len(names), "tasks_per_shot": 600,
            "episodes_rescored": total//75, "query_predictions_rescored": total,
            "validation_rule": best_rule, "meta_epoch": best_meta, "replay_predictions_match": True if compare else None,
            "test_correct_by_method": {k: [sum(counts[k, s]) for s in (1, 5)] for k in sorted(names)}}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("outputs/lab04_unified256"))
    p.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    p.add_argument("--predictions", type=Path, help="Directory with results.json and test predictions, if separate")
    p.add_argument("--compare", type=Path, help="Additionally compare EVERY replayed class label to this directory")
    p.add_argument("--save", type=Path, help="Optional JSON audit report")
    args = p.parse_args()
    report = json.dumps(audit(args.bundle, args.artifacts, args.predictions, args.compare), indent=2)
    if args.save:
        args.save.write_text(report + "\n", encoding="utf-8")
    print(report)
