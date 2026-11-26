"""Compare regional supervision and region matching on fixed mini-ImageNet episodes."""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from .accuracy90_data import json_from, load_archive
from .data import digest_file
from .evaluate_gpu_tracks import ResNetFeatures, verified_files
from .experiment import summarize
from .multiscale_study import ViewFeatures, representation
from .prototype_restarts import predict_split
from .regional_kernel import collect_regions, predict_regions
from .rotation_study import INFERENCE, linear_sum_assignment
from .self_distill_study import correct_by_task, get_tasks
from .study_cached_refinement import paired_stats
from .train_gpu_metric import save_json


CANDIDATES = ("augmented", "rotation_views", "regional", "augmented_kernel", "regional_kernel")


def score(predictions, target, reference):
    rates = correct_by_task(predictions, target)
    return {**summarize(rates), "correct": int((predictions == target).sum()), "queries": int(target.size),
            "paired_vs_augmented": paired_stats(rates, correct_by_task(reference, target))}


def infer(directory, split, tasks, plan, *, standard, kernel):
    extractor = ViewFeatures(directory, "cuda")
    rows = extractor.collect(split.images, plan["views"].values())
    regions = collect_regions(extractor.original.model, split.images, "cuda", plan["kernel"]["grid"]) if kernel else None
    predictions, matched = {}, {}
    for shot, task in tasks.items():
        x = representation(rows, *plan["views"][shot])
        rule = plan["selected"][shot]
        if standard:
            predictions[shot] = predict_split(x, task, [rule])[rule]
        if kernel:
            matched[shot] = predict_regions(x, regions, task, plan["kernel"], rule, "cuda")
    return predictions, matched


def validate(args, plan, release):
    val, tasks = get_tasks(args, "val", release)
    archive = json_from(args.control / "validation_lock.json")
    predictions = {}
    for name, old_name in (("augmented", "augmented"), ("rotation_views", "parent")):
        with np.load(args.control / f"val_{old_name}.npz", allow_pickle=False) as data:
            predictions[name] = {s: data[s] for s in tasks}
    print("Fitting feature calibration using only base images", flush=True)
    base = load_archive(args.data_root, args.artifacts, "train")
    ResNetFeatures(args.trained, "cuda", 32).fit_base(base, args.trained)
    del base
    for name, directory in (("augmented", args.control), ("regional", args.trained)):
        print("Validation from images:", name, flush=True)
        standard, matched = infer(directory, val, tasks, plan, standard=True, kernel=True)
        if name == "augmented" and any(not np.array_equal(standard[s], predictions[name][s]) for s in tasks):
            raise ValueError("Image evaluation failed to reproduce archived augmentation predictions")
        predictions[name], predictions[name + "_kernel"] = standard, matched
        print(name, {s: (int((standard[s] == t[3]).sum()), int((matched[s] == t[3]).sum()))
                     for s, t in tasks.items()}, flush=True)

    scores, files = {}, {}
    for name in CANDIDATES:
        scores[name] = {s: score(predictions[name][s], t[3], predictions["augmented"][s]) for s, t in tasks.items()}
        file = f"val_{name}.npz"
        np.savez_compressed(args.output / file, **predictions[name])
        files[file] = digest_file(args.output / file)
    for name, old_name in (("augmented", "augmented"), ("rotation_views", "parent")):
        if any(scores[name][s]["correct"] != archive["candidates"][old_name][s]["correct"] for s in tasks):
            raise ValueError("Reference scores differ from archived validation")
    winner = max(CANDIDATES, key=lambda n: sum(r["correct"] for r in scores[n].values()))
    sources = {"augmented": args.control, "regional": args.trained, "rotation_views": args.parent}
    directory = sources[winner.removesuffix("_kernel")]
    for file in INFERENCE:
        shutil.copyfile(directory / file, args.output / file)
    paired = {}
    for name, other in (("regional", "augmented"), ("regional_kernel", "augmented_kernel"),
                        ("regional_kernel", "regional"), ("augmented_kernel", "augmented")):
        paired[f"{name}_minus_{other}"] = {s: paired_stats(correct_by_task(predictions[name][s], t[3]),
                                             correct_by_task(predictions[other][s], t[3])) for s, t in tasks.items()}
    lock = {"winner": winner, "views": plan["views"], "selected": plan["selected"],
            "kernel": plan["kernel"] if winner.endswith("_kernel") else None,
            "plan_sha256": digest_file(args.output / "plan.json"), "validation_files": files,
            "inference_files": {n: digest_file(args.output / n) for n in INFERENCE},
            "candidates": scores, "validation": scores[winner], "validation_paired": paired}
    save_json(args.output / "validation_lock.json", lock)
    print("Locked validation winner:", winner, flush=True)


def study(args):
    torch.set_num_threads(2)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    release = json_from(args.artifacts / "continued_results.json")
    control = json_from(args.control / "results.json")
    parent = json_from(args.parent / "results.json")
    training = json_from(args.trained / "training.json")
    training_plan = json_from(args.trained / "training_plan.json")
    control_plan = json_from(args.control / "plan.json")
    verified_files(args.control, {n: control["files"][n] for n in (*INFERENCE, "val_augmented.npz", "val_parent.npz",
                                                                   "validation_lock.json", "plan.json", "training.json")})
    verified_files(args.parent, {n: parent["files"][n] for n in (*INFERENCE, "validation_lock.json")})
    verified_files(args.trained, {**training["files"], "training_plan.json": training["plan_sha256"]})
    if (training_plan["matched_plan_sha256"] != control["files"]["plan.json"] or
            training_plan["matched_training"] != control_plan["training"] or
            control_plan["parent_validation_lock_sha256"] != parent["files"]["validation_lock.json"] or
            control["winner"] != "augmented" or control["external_pretraining"] or parent["external_pretraining"] or
            [r["region_updates"] for r in training["history"]] != [150] * control_plan["training"]["epochs"] or
            any(release["files"][n]["sha256"] != sha for n, sha in control_plan["episode_files"].items()) or
            digest_file(args.artifacts / "dataset_manifest.json") != control_plan["dataset_manifest_sha256"]):
        raise ValueError("Expected a matched regional continuation with the original dataset and episodes")
    plan = {"track": "regional", "external_pretraining": False, "candidates_in_tie_order": list(CANDIDATES),
            "training": training_plan, "selection": training_plan["selection"], "kernel": training_plan["kernel"],
            "views": control_plan["views"], "selected": control_plan["selected"], "branch_sha256": training["files"]["medium.pt"],
            "training_record_sha256": digest_file(args.trained / "training.json"), "episode_files": control_plan["episode_files"],
            "dataset_manifest_sha256": control_plan["dataset_manifest_sha256"], "gpu": torch.cuda.get_device_name(),
            "assignment_solver": "scipy" if linear_sum_assignment else "numpy",
            "source_sha256": {n: digest_file(Path(__file__).with_name(n)) for n in
                              ("regional_study.py", "regional_kernel.py", "prototype_restarts.py", "multiscale_study.py",
                               "evaluate_gpu_tracks.py", "rotation_study.py", "gpu_metric.py")}}
    if (out / "plan.json").exists() and json_from(out / "plan.json") != plan:
        raise ValueError("Another experiment uses this directory")
    if (out / "results.json").exists():
        verified_files(out, json_from(out / "results.json")["files"])
        return
    save_json(out / "plan.json", plan)
    shutil.copyfile(args.trained / "training.json", out / "training.json")
    if not (out / "validation_lock.json").exists():
        validate(args, plan, release)
    lock = json_from(out / "validation_lock.json")
    files = {**lock["validation_files"], **lock["inference_files"], "plan.json": lock["plan_sha256"]}
    verified_files(out, files)
    winner = lock["winner"]
    test, tasks = get_tasks(args, "test", release)
    histories = {}
    for name, directory, record in (("augmented", args.control, control), ("rotation_views", args.parent, parent)):
        verified_files(directory, {"test_predictions.json": record["files"]["test_predictions.json"]})
        histories[name] = {s: np.asarray(p) for s, p in json_from(directory / "test_predictions.json").items()}
    if winner in histories:
        predictions = histories[winner]
    else:
        standard, matched = infer(out, test, tasks, plan, standard=lock["kernel"] is None, kernel=lock["kernel"] is not None)
        predictions = standard if lock["kernel"] is None else matched
    scores = {s: score(predictions[s], t[3], histories["augmented"][s]) for s, t in tasks.items()}
    for s, t in tasks.items():
        scores[s]["paired_vs_rotation_views"] = paired_stats(correct_by_task(predictions[s], t[3]),
                                                             correct_by_task(histories["rotation_views"][s], t[3]))
    save_json(out / "test_predictions.json", {s: p.tolist() for s, p in predictions.items()})
    files.update({n: digest_file(out / n) for n in ("test_predictions.json", "training.json", "validation_lock.json")})
    save_json(out / "results.json", {"track": "regional", "winner": winner, "external_pretraining": False,
                                     "val": lock["validation"], "test": scores, "files": files})
    print("Test", winner, {s: r["correct"] for s, r in scores.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    parser.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    parser.add_argument("--parent", type=Path, default=Path("outputs/lab04_gpu_rotation_views"))
    parser.add_argument("--control", type=Path, default=Path("outputs/lab04_gpu_image_augmentation"))
    parser.add_argument("--trained", type=Path, default=Path(".local/lab04-regional"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lab04_gpu_regional"))
    study(parser.parse_args())
