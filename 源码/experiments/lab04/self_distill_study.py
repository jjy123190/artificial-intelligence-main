"""Select a single trained ResNet12 using fixed views and classifier rules on validation."""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from .accuracy90_data import json_from, load_archive
from .data import digest_file
from .evaluate_gpu_tracks import ResNetFeatures, task_indices, verified_files
from .experiment import summarize
from .multiscale_study import representation, ViewFeatures
from .prototype_restarts import predict_split
from .rotation_study import evaluate, INFERENCE, linear_sum_assignment
from .study_cached_refinement import paired_stats
from .train_gpu_metric import reference_rows, save_json
from .train_self_distill import BRANCHES, verify_training


def correct_by_task(predictions, targets):
    pred = np.asarray(predictions)
    if pred.shape != targets.shape or pred.shape != (600, 75):
        raise ValueError("Predictions must follow the 600 released 75-query episodes")
    return (pred == targets).mean(1)


def measures(predictions, targets, baseline):
    by_task = correct_by_task(predictions, targets)
    return {**summarize(by_task), "correct": int((np.asarray(predictions) == targets).sum()),
            "queries": targets.size, "paired_vs_restarts": paired_stats(by_task, correct_by_task(baseline, targets))}


def get_tasks(args, split_name, release):
    split = load_archive(args.data_root, args.artifacts, split_name)
    rows = reference_rows(args.artifacts, split_name, release)
    return split, {str(k): task_indices(split, rows, k) for k in (1, 5)}


def predict_images(directory, split, tasks, config, *, diagnostics=False):
    extractor = ViewFeatures(directory, "cuda")
    all_features = extractor.collect(split.images, config["views"].values())
    predictions, support_only = {}, {}
    for shot, task in tasks.items():
        x = representation(all_features, *config["views"][shot])
        rule = config["selected"][shot]
        predictions[shot] = predict_split(x, task, [rule])[rule]
        if diagnostics:
            independent = evaluate(x, task, "support_only", "cuda")
            support_only[shot] = summarize(correct_by_task(independent, task[3]))
    return predictions, support_only


def validate(args, plan, records, prior, release):
    out = args.output
    val, tasks = get_tasks(args, "val", release)
    archived = json_from(args.previous / "val_predictions.json")
    sources = {"parent": args.weights, **{name: args.trained / name for name in BRANCHES}}
    scores, diagnostics, files, candidates = {}, {}, {}, {}
    base = load_archive(args.data_root, args.artifacts, "train")
    for name, directory in sources.items():
        if name != "parent":
            print("Fitting base-class whitening and spatial metric:", name, flush=True)
            ResNetFeatures(directory, "cuda", 32).fit_base(base, directory)
        pred, diagnostics[name] = predict_images(directory, val, tasks, prior, diagnostics=True)
        if name == "parent" and any(not np.array_equal(pred[s], archived[s]) for s in tasks):
            raise ValueError("Fixed inference did not reproduce archived parent validation predictions")
        candidates[name] = {s: p.tolist() for s, p in pred.items()}
        scores[name] = {s: measures(p, tasks[s][3], archived[s]) for s, p in pred.items()}
        file = f"val_{name}.json"
        save_json(out / file, candidates[name])
        files[file] = digest_file(out / file)
        print("validation", name, {s: r["accuracy"] for s, r in scores[name].items()}, flush=True)
    winner = max(sources, key=lambda name: sum(row["correct"] for row in scores[name].values()))
    comparison = {s: paired_stats(correct_by_task(candidates["distill"][s], t[3]),
                                  correct_by_task(candidates["control"][s], t[3])) for s, t in tasks.items()}
    for file in INFERENCE:
        shutil.copyfile(sources[winner] / file, out / file)
    training = {"plan": plan["training"], "teacher": json_from(args.trained / "teacher.json"), "branches": records}
    save_json(out / "training.json", training)
    files["training.json"] = digest_file(out / "training.json")
    lock = {"winner": winner, "views": prior["views"], "selected": prior["selected"], "candidates": scores,
            "validation": scores[winner], "validation_distill_minus_control": comparison,
            "support_only_validation": diagnostics, "plan_sha256": digest_file(out / "plan.json"),
            "validation_files": files, "inference_files": {n: digest_file(out / n) for n in INFERENCE}}
    save_json(out / "validation_lock.json", lock)
    print("Locked validation choice:", winner, flush=True)


def study(args):
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(2)
    previous = json_from(args.previous / "results.json")
    verified_files(args.previous, {name: previous["files"][name] for name in
                                   ("plan.json", "validation_lock.json", "val_predictions.json")})
    prior = json_from(args.previous / "validation_lock.json")
    verified_files(args.weights, {n: prior["inference_files"][n] for n in INFERENCE})
    training = json_from(args.trained / "training_plan.json")
    if (previous["external_pretraining"] or training["external_weights"] or
            training["branches"] != list(BRANCHES) or training["parent_sha256"] != prior["inference_files"]["medium.pt"]
            or training["train_ids"] != json_from(args.artifacts / "protocol.json")["train_selection_sha256"]
            or any(weight for _, weight in prior["views"].values())):
        raise ValueError("Expected matched base-only backbones and fixed global/spatial features")
    records = {name: verify_training(args.trained / name, name, training) for name in BRANCHES}
    release = json_from(args.artifacts / "continued_results.json")
    episode_hashes = {n: release["files"][n]["sha256"] for n in
                      ("continued_validation_tasks.jsonl", "continued_episodes.json")}
    if episode_hashes != json_from(args.previous / "plan.json")["episode_files"]:
        raise ValueError("Episode definitions differ from the parent experiment")
    sources = ("self_distill_study.py", "prototype_restarts.py", "multiscale_study.py", "evaluate_gpu_tracks.py",
               "gpu_metric.py", "rotation_study.py", "supervised_whitening.py", "study_cached_refinement.py")
    plan = {"track": "self_distill", "external_pretraining": False, "training": training,
            "selection": training["selection"], "frozen_views": prior["views"], "frozen_rules": prior["selected"],
            "parent_lock_sha256": digest_file(args.previous / "validation_lock.json"),
            "dataset_manifest_sha256": digest_file(args.artifacts / "dataset_manifest.json"),
            "episode_files": episode_hashes, "gpu": torch.cuda.get_device_name(),
            "assignment_solver": "scipy" if linear_sum_assignment is not None else "numpy",
            "eval_source_sha256": {n: digest_file(Path(__file__).with_name(n)) for n in sources},
            "trained_sha256": {n: record["medium_sha256"] for n, record in records.items()}}
    if (out / "plan.json").exists() and plan != json_from(out / "plan.json"):
        raise ValueError("Different study in output directory")
    if (out / "results.json").is_file():
        verified_files(out, json_from(out / "results.json")["files"])
        print("Verified", out / "results.json", flush=True)
        return
    save_json(out / "plan.json", plan)
    if not (out / "validation_lock.json").exists():
        validate(args, plan, records, prior, release)
    lock = json_from(out / "validation_lock.json")
    verified_files(out, {**lock["inference_files"], **lock["validation_files"], "plan.json": lock["plan_sha256"]})

    # The first test data access occurs only after validation selection has been persisted.
    verified_files(args.previous, {"test_predictions.json": previous["files"]["test_predictions.json"]})
    baseline = json_from(args.previous / "test_predictions.json")
    test, tasks = get_tasks(args, "test", release)
    if lock["winner"] == "parent":
        predictions = baseline
    else:
        pred, _ = predict_images(out, test, tasks, lock)
        predictions = {s: row.tolist() for s, row in pred.items()}
    test_scores = {s: measures(predictions[s], task[3], baseline[s]) for s, task in tasks.items()}
    save_json(out / "test_predictions.json", predictions)
    files = {**lock["inference_files"], **lock["validation_files"],
             **{n: digest_file(out / n) for n in ("plan.json", "validation_lock.json", "test_predictions.json")}}
    save_json(out / "results.json", {"track": "self_distill", "winner": lock["winner"], "external_pretraining": False,
              "val": lock["validation"], "test": test_scores, "views": lock["views"], "selected": lock["selected"],
              "files": files})
    print("test", lock["winner"], test_scores, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/lab04_gpu_restarts"))
    parser.add_argument("--weights", type=Path, default=Path("outputs/lab04_gpu_rotation"))
    parser.add_argument("--trained", type=Path, default=Path(".local/lab04-self-distill"))
    parser.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lab04_gpu_self_distill"))
    study(parser.parse_args())
