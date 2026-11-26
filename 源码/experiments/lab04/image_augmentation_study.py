"""Compare matched continuation runs on fixed episodes, then score the validation winner."""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from .accuracy90_data import json_from, load_archive
from .data import digest_file
from .episode_metric_study import score
from .evaluate_gpu_tracks import ResNetFeatures, verified_files
from .rotation_study import INFERENCE, linear_sum_assignment
from .self_distill_study import correct_by_task, get_tasks, predict_images
from .study_cached_refinement import paired_stats
from .train_gpu_metric import save_json
from .train_image_augmentation import BRANCHES


def validate(args, plan, parent_result, release):
    out = args.output
    val, tasks = get_tasks(args, "val", release)
    base = load_archive(args.data_root, args.artifacts, "train")
    original = json_from(args.parent / f"val_{parent_result['winner']}.json")
    previous = {s: np.asarray(p) for s, p in original.items()}
    scores, rates, files = {}, {}, {}
    directories = {"parent": args.parent, **{n: args.trained / n for n in BRANCHES}}
    for name, directory in directories.items():
        predictions = previous
        if name != "parent":
            print("Fitting base-only feature calibration:", name, flush=True)
            ResNetFeatures(directory, "cuda", 32).fit_base(base, directory)
            predictions, _ = predict_images(directory, val, tasks, plan)
        scores[name] = {s: score(p, tasks[s][3], previous[s]) for s, p in predictions.items()}
        if name == "parent" and any(row["correct"] != parent_result["val"][s]["correct"]
                                    for s, row in scores[name].items()):
            raise ValueError("Archived predictions disagree with the validation reference")
        rates[name] = {s: correct_by_task(p, tasks[s][3]) for s, p in predictions.items()}
        file = f"val_{name}.npz"
        np.savez_compressed(out / file, **predictions)
        files[file] = digest_file(out / file)
        print("Validation", name, {s: r["accuracy"] for s, r in scores[name].items()}, flush=True)
    winner = max(directories, key=lambda n: sum(r["correct"] for r in scores[n].values()))
    for file in INFERENCE:
        shutil.copyfile(directories[winner] / file, out / file)
    paired = {s: paired_stats(rates["augmented"][s], rates["control"][s]) for s in tasks}
    lock = {"winner": winner, "views": plan["views"], "selected": plan["selected"], "candidates": scores,
            "validation": scores[winner], "augmented_minus_control": paired, "validation_files": files,
            "plan_sha256": digest_file(out / "plan.json"),
            "inference_files": {n: digest_file(out / n) for n in INFERENCE}}
    save_json(out / "validation_lock.json", lock)
    print("Locked single backbone:", winner, flush=True)


def study(args):
    torch.set_num_threads(2)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    parent = json_from(args.parent / "results.json")
    parent_val = f"val_{parent['winner']}.json"
    verified_files(args.parent, {n: parent["files"][n] for n in (*INFERENCE, "plan.json", "validation_lock.json", parent_val)})
    parent_plan = json_from(args.parent / "plan.json")
    training = json_from(args.trained / "training_plan.json")
    training_sha = digest_file(args.trained / "training_plan.json")
    if (parent["track"] != "rotation_views" or parent["external_pretraining"] or training["external_weights"]
            or training["branches"] != list(BRANCHES) or training["parent_sha256"] != parent["files"]["medium.pt"]
            or training["train_ids"] != json_from(args.artifacts / "protocol.json")["train_selection_sha256"]):
        raise ValueError("Expected the two matched continuations of the same base-only ResNet12")
    logs = {}
    for name in BRANCHES:
        record = json_from(args.trained / name / "training.json")
        if (record["plan_sha256"] != training_sha or record["branch"] != name or
                [(r["epoch"], r["updates"], r["images"]) for r in record["history"]] !=
                [(i + 1, 300, 38400) for i in range(training["epochs"])]):
            raise ValueError(f"Incomplete or mismatched training: {name}")
        verified_files(args.trained / name, record["files"])
        logs[name] = record
    release = json_from(args.artifacts / "continued_results.json")
    if (any(release["files"][n]["sha256"] != sha for n, sha in parent_plan["episode_files"].items())
            or digest_file(args.artifacts / "dataset_manifest.json") != parent_plan["dataset_manifest_sha256"]):
        raise ValueError("Dataset or fixed evaluation tasks changed")
    prior_lock = json_from(args.parent / "validation_lock.json")
    plan = {"track": "image_augmentation", "external_pretraining": False, "selection": training["selection"],
            "training": training, "branch_weights": {n: r["files"]["medium.pt"] for n, r in logs.items()},
            "views": prior_lock["views"], "selected": prior_lock["selected"],
            "parent_validation_lock_sha256": parent["files"]["validation_lock.json"],
            "dataset_manifest_sha256": parent_plan["dataset_manifest_sha256"], "episode_files": parent_plan["episode_files"],
            "gpu": torch.cuda.get_device_name(), "assignment_solver": "scipy" if linear_sum_assignment else "numpy",
            "eval_source_sha256": {n: digest_file(Path(__file__).with_name(n)) for n in
                                   ("image_augmentation_study.py", "self_distill_study.py", "episode_metric_study.py",
                                    "prototype_restarts.py", "multiscale_study.py", "evaluate_gpu_tracks.py")}}
    if (out / "plan.json").exists() and json_from(out / "plan.json") != plan:
        raise ValueError("A different study is already present in the output directory")
    if (out / "results.json").exists():
        verified_files(out, json_from(out / "results.json")["files"])
        return
    save_json(out / "plan.json", plan)
    save_json(out / "training.json", logs)
    if not (out / "validation_lock.json").exists():
        validate(args, plan, parent, release)
    lock = json_from(out / "validation_lock.json")
    files = {**lock["validation_files"], **lock["inference_files"], "plan.json": lock["plan_sha256"]}
    verified_files(out, files)
    verified_files(args.parent, {"test_predictions.json": parent["files"]["test_predictions.json"]})
    baseline = json_from(args.parent / "test_predictions.json")
    test, tasks = get_tasks(args, "test", release)
    if lock["winner"] == "parent":
        predictions = {s: np.asarray(p) for s, p in baseline.items()}
    else:
        predictions, _ = predict_images(out, test, tasks, lock)
    test_scores = {s: score(p, tasks[s][3], np.asarray(baseline[s])) for s, p in predictions.items()}
    save_json(out / "test_predictions.json", {s: p.tolist() for s, p in predictions.items()})
    files.update({n: digest_file(out / n) for n in ("test_predictions.json", "training.json", "validation_lock.json")})
    save_json(out / "results.json", {"track": plan["track"], "external_pretraining": False,
              "winner": lock["winner"], "val": lock["validation"], "test": test_scores, "files": files})
    print("Test", lock["winner"], {s: r["accuracy"] for s, r in test_scores.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    parser.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    parser.add_argument("--parent", type=Path, default=Path("outputs/lab04_gpu_rotation_views"))
    parser.add_argument("--trained", type=Path, default=Path(".local/lab04-image-augmentation"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lab04_gpu_image_augmentation"))
    study(parser.parse_args())
