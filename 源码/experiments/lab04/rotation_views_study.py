"""Evaluate rotation-distilled checkpoints with the previously locked few-shot classifier."""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from .accuracy90_data import json_from, load_archive
from .data import digest_file
from .evaluate_gpu_tracks import ResNetFeatures, verified_files
from .rotation_study import INFERENCE, linear_sum_assignment
from .self_distill_study import correct_by_task, get_tasks, measures, predict_images
from .study_cached_refinement import paired_stats
from .train_gpu_metric import save_json


NAMES = ("parent", "last", "average")


def validate(args, plan, release, previous, prior):
    out = args.output
    val, tasks = get_tasks(args, "val", release)
    base = load_archive(args.data_root, args.artifacts, "train")
    parent = json_from(args.previous / "val_predictions.json")
    sources = {"parent": args.weights, "last": args.trained / "last", "average": args.trained / "average"}
    scores, support_only, files, per_task = {}, {}, {}, {}
    for name, source in sources.items():
        pred = parent
        if name == "parent":
            support_only[name] = prior["support_only_validation"]["parent"]
        else:
            print("Fitting train-only feature calibration:", name, flush=True)
            ResNetFeatures(source, "cuda", 32).fit_base(base, source)
            pred, support_only[name] = predict_images(source, val, tasks, plan, diagnostics=True)
        scores[name] = {s: measures(pred[s], task[3], parent[s]) for s, task in tasks.items()}
        per_task[name] = {s: correct_by_task(pred[s], task[3]) for s, task in tasks.items()}
        if name == "parent" and any(scores[name][s]["correct"] != previous["val"][s]["correct"] for s in tasks):
            raise ValueError("Archived parent scores do not match the fixed validation tasks")
        file = f"val_{name}.json"
        save_json(out / file, {s: np.asarray(p).tolist() for s, p in pred.items()})
        files[file] = digest_file(out / file)
        print("validation", name, {s: v["accuracy"] for s, v in scores[name].items()}, flush=True)
    winner = max(NAMES, key=lambda name: sum(v["correct"] for v in scores[name].values()))
    for file in INFERENCE:
        shutil.copyfile(sources[winner] / file, out / file)
    lock = {"winner": winner, "views": plan["views"], "selected": plan["selected"], "candidates": scores,
            "validation": scores[winner], "support_only_validation": support_only,
            "average_minus_last": {s: paired_stats(a, per_task["last"][s]) for s, a in per_task["average"].items()},
            "plan_sha256": digest_file(out / "plan.json"), "validation_files": files,
            "inference_files": {n: digest_file(out / n) for n in INFERENCE}}
    save_json(out / "validation_lock.json", lock)
    print("Locked single backbone:", winner, flush=True)


def study(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    previous = json_from(args.previous / "results.json")
    verified_files(args.previous, {n: previous["files"][n] for n in ("plan.json", "validation_lock.json", "val_predictions.json")})
    prior_result = json_from(args.weights / "results.json")
    verified_files(args.weights, {n: prior_result["files"][n] for n in (*INFERENCE, "validation_lock.json", "val_parent.json")})
    prior = json_from(args.weights / "validation_lock.json")
    original = json_from(args.previous / "validation_lock.json")
    schedule = json_from(args.trained / "training_plan.json")
    completed = json_from(args.trained / "training.json")
    verified_files(args.trained, {**completed["files"], "training_plan.json": completed["plan_sha256"]})
    if (previous["external_pretraining"] or prior_result["external_pretraining"] or schedule["external_weights"]
            or prior["winner"] != "parent" or prior["selected"] != original["selected"] or prior["views"] != original["views"]
            or any(prior_result["files"][n] != original["inference_files"][n] for n in INFERENCE)
            or prior_result["files"]["val_parent.json"] != previous["files"]["val_predictions.json"]
            or schedule["train_ids"] != json_from(args.artifacts / "protocol.json")["train_selection_sha256"]
            or schedule["parent_sha256"] != prior_result["files"]["medium.pt"]
            or set(completed["files"]) != {f"{n}/medium.pt" for n in NAMES[1:]}
            or [(r["epoch"], r["updates"]) for r in completed["history"]] !=
            [(i + 1, 300) for i in range(schedule["epochs"])]):
        raise ValueError("Training, parent weights, or fixed inference differ from the expected experiment")
    release = json_from(args.artifacts / "continued_results.json")
    episode_files = json_from(args.previous / "plan.json")["episode_files"]
    if any(release["files"][n]["sha256"] != sha for n, sha in episode_files.items()):
        raise ValueError("Episode definitions differ from the parent experiment")
    sources = ("rotation_views_study.py", "self_distill_study.py", "prototype_restarts.py", "multiscale_study.py",
               "rotation_study.py", "evaluate_gpu_tracks.py", "gpu_metric.py", "supervised_whitening.py")
    plan = {"track": "rotation_views", "external_pretraining": False, "training": schedule,
            "training_sha256": digest_file(args.trained / "training.json"), "selection": schedule["selection"],
            "parent_lock_sha256": digest_file(args.previous / "validation_lock.json"),
            "prior_study_sha256": digest_file(args.weights / "results.json"), "views": prior["views"],
            "selected": prior["selected"], "episode_files": episode_files, "gpu": torch.cuda.get_device_name(),
            "dataset_manifest_sha256": digest_file(args.artifacts / "dataset_manifest.json"),
            "assignment_solver": "scipy" if linear_sum_assignment is not None else "numpy",
            "eval_source": {n: digest_file(Path(__file__).with_name(n)) for n in sources}}
    if (out / "plan.json").exists() and json_from(out / "plan.json") != plan:
        raise ValueError("Output is already used for a different study")
    if (out / "results.json").exists():
        verified_files(out, json_from(out / "results.json")["files"])
        return
    save_json(out / "plan.json", plan)
    shutil.copyfile(args.trained / "training.json", out / "training.json")
    if not (out / "validation_lock.json").exists():
        validate(args, plan, release, previous, prior)
    lock = json_from(out / "validation_lock.json")
    verified_files(out, {**lock["validation_files"], **lock["inference_files"], "plan.json": lock["plan_sha256"],
                         "training.json": plan["training_sha256"]})

    # Test access follows persistence and verification of the validation decision.
    verified_files(args.previous, {"test_predictions.json": previous["files"]["test_predictions.json"]})
    old = json_from(args.previous / "test_predictions.json")
    test, tasks = get_tasks(args, "test", release)
    pred = old
    if lock["winner"] != "parent":
        pred, _ = predict_images(out, test, tasks, lock)
    test_scores = {s: measures(pred[s], task[3], old[s]) for s, task in tasks.items()}
    save_json(out / "test_predictions.json", {s: np.asarray(p).tolist() for s, p in pred.items()})
    files = {**lock["validation_files"], **lock["inference_files"], **{n: digest_file(out / n) for n in
             ("plan.json", "training.json", "validation_lock.json", "test_predictions.json")}}
    save_json(out / "results.json", {"track": plan["track"], "external_pretraining": False, "winner": lock["winner"],
              "views": lock["views"], "selected": lock["selected"], "val": lock["validation"], "test": test_scores,
              "files": files})
    print("test", lock["winner"], test_scores, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/lab04_gpu_restarts"))
    parser.add_argument("--weights", type=Path, default=Path("outputs/lab04_gpu_self_distill"))
    parser.add_argument("--trained", type=Path, default=Path(".local/lab04-rotation-views"))
    parser.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lab04_gpu_rotation_views"))
    study(parser.parse_args())
