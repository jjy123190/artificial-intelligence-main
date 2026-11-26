"""Select and confirm task-conditioned feature calibration before evaluating the test split."""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from .accuracy90_data import json_from
from .data import digest_file, sample_episode, selection_digest
from .episode_metric_study import score
from .evaluate_gpu_tracks import verified_files
from .multiscale_study import ViewFeatures, representation
from .rotation_study import INFERENCE, linear_sum_assignment
from .self_distill_study import get_tasks
from .task_subspace import ALPHA, CONFIGS, NEIGHBORS, RANK, predict_batch
from .train_gpu_metric import save_json


SEEDS = {"1": 516247, "5": 516251}
KEYS = ("support_indices", "support_labels", "query_indices", "query_labels")
PAPERS = {
    "TAFSSL": "https://arxiv.org/abs/2003.06670",
    "iLPC": "https://openaccess.thecvf.com/content/ICCV2021/html/Lazarou_Iterative_Label_Cleaning_for_Transductive_and_Semi-Supervised_Few-Shot_Learning_ICCV_2021_paper.html",
}


def confirmation_tasks(labels, shot):
    rng = np.random.default_rng(SEEDS[shot])
    rows = [sample_episode(labels, rng, shots=int(shot), queries=15) for _ in range(600)]
    return tuple(np.stack([r[key] for r in rows]).astype(np.int64, copy=False) for key in KEYS)


def predictions(features, task, rule, names):
    # Ground truth is used by the caller after prediction.
    si, sl, qi, _ = task
    by_name = {name: [] for name in names}
    for start in range(0, len(qi), 16):
        slc = slice(start, start + 16)
        s, y, q = [torch.from_numpy(a).cuda() for a in (features[si[slc]], sl[slc], features[qi[slc]])]
        for name, answers in predict_batch(s, y, q, rule, names).items():
            by_name[name].append(answers)
    return {name: np.concatenate(rows) for name, rows in by_name.items()}


def validate(args, plan, release, val_file):
    out = args.output
    val, tasks = get_tasks(args, "val", release)
    extractor = ViewFeatures(args.previous, "cuda")
    encoded = extractor.collect(val.images, plan["views"].values())
    reference = json_from(args.previous / val_file)
    scores, candidates, files = {}, {}, {}
    for shot, task in tasks.items():
        x = representation(encoded, *plan["views"][shot])
        all_pred = predictions(x, task, plan["rules"][shot], CONFIGS)
        parent = all_pred["parent"]
        if not np.array_equal(parent, reference[shot]):
            raise ValueError("Reference model did not reproduce all archived validation predictions")
        scores[shot] = {n: score(p, task[3], parent) for n, p in all_pred.items()}
        candidates[shot] = max(CONFIGS, key=lambda n: scores[shot][n]["correct"])
        file = f"fixed_val_{shot}.npz"
        np.savez_compressed(out / file, **all_pred)
        files[file] = digest_file(out / file)
        print("Fixed validation", shot, {n: r["correct"] for n, r in scores[shot].items()}, flush=True)
    save_json(out / "shortlist.json", {"candidates": candidates, "fixed": scores, "files": files,
              "plan_sha256": digest_file(out / "plan.json")})
    # Both top candidates have been fixed before looking at either confirmation task bank.
    confirmed, selected = {}, {}
    for shot, name in candidates.items():
        task = confirmation_tasks(val.labels, shot)
        x = representation(encoded, *plan["views"][shot])
        pred = predictions(x, task, plan["rules"][shot], list(dict.fromkeys(("parent", name))))
        confirmed[shot] = {n: score(p, task[3], pred["parent"]) for n, p in pred.items()}
        diff = confirmed[shot][name]["paired_vs_rotation_views"]
        selected[shot] = name if diff["mean"] > diff["ci95"] else "parent"
        file = f"confirmation_{shot}.npz"
        np.savez_compressed(out / file, **dict(zip(KEYS, task)), **pred)
        files[file] = digest_file(out / file)
        print("Confirmation", shot, name, confirmed[shot][name]["correct"],
              "parent", confirmed[shot]["parent"]["correct"], "keep", selected[shot], flush=True)
    files["shortlist.json"] = digest_file(out / "shortlist.json")
    inference = {n: digest_file(args.previous / n) for n in INFERENCE}
    for n in INFERENCE:
        shutil.copyfile(args.previous / n, out / n)
    lock = {"selected": selected, "validation": {s: scores[s][n] for s, n in selected.items()},
            "confirmation": confirmed, "confirmation_id_order_sha256": selection_digest(val),
            "views": plan["views"], "rules": plan["rules"], "validation_files": files,
            "inference_files": inference, "plan_sha256": digest_file(out / "plan.json")}
    save_json(out / "validation_lock.json", lock)
    print("Locked", selected, flush=True)


def study(args):
    torch.set_num_threads(2)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    previous = json_from(args.previous / "results.json")
    if previous["external_pretraining"] or previous["track"] != "rotation_views":
        raise ValueError("Expected the previous self-trained ResNet12")
    val_file = f"val_{previous['winner']}.json"
    verified_files(args.previous, {n: previous["files"][n] for n in
                                   (*INFERENCE, "validation_lock.json", "plan.json", val_file)})
    parent_plan = json_from(args.previous / "plan.json")
    parent = json_from(args.previous / "validation_lock.json")
    release = json_from(args.artifacts / "continued_results.json")
    hashes = parent_plan["episode_files"]
    if (parent_plan["training"]["train_ids"] != json_from(args.artifacts / "protocol.json")["train_selection_sha256"]
            or digest_file(args.artifacts / "dataset_manifest.json") != parent_plan["dataset_manifest_sha256"]
            or any(release["files"][n]["sha256"] != sha for n, sha in hashes.items())):
        raise ValueError("The training split or episode protocol changed")
    sources = ("task_subspace.py", "task_subspace_study.py", "prototype_restarts.py", "rotation_study.py",
               "multiscale_study.py", "gpu_metric.py", "episode_metric_study.py", "evaluate_gpu_tracks.py",
               "self_distill_study.py", "supervised_whitening.py")
    plan = {"track": "task_subspace", "external_pretraining": False, "papers": PAPERS,
            "configs": {n: list(w) for n, w in CONFIGS.items()}, "component_order": ["pca_weight", "graph_weight"],
            "rank": RANK, "neighbors": NEIGHBORS, "propagation_alpha": ALPHA, "similarity_power": 3,
            "selection": "Per shot, highest correct count on 600 fixed validation episodes; first candidate wins ties. "
                         "Enable that candidate only when the paired 95% lower bound exceeds zero on 600 fresh validation episodes.",
            "confirmation_seeds": SEEDS, "views": parent["views"], "rules": parent["selected"],
            "dataset_manifest_sha256": parent_plan["dataset_manifest_sha256"], "episode_files": hashes,
            "parent_lock_sha256": previous["files"]["validation_lock.json"],
            "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "assignment_solver": "scipy" if linear_sum_assignment is not None else "numpy",
            "source_sha256": {n: digest_file(Path(__file__).with_name(n)) for n in sources}}
    if (out / "plan.json").exists() and json_from(out / "plan.json") != plan:
        raise ValueError("Output directory already contains another experimental plan")
    if (out / "results.json").exists():
        verified_files(out, json_from(out / "results.json")["files"])
        return
    save_json(out / "plan.json", plan)
    if not (out / "validation_lock.json").is_file():
        validate(args, plan, release, val_file)
    lock = json_from(out / "validation_lock.json")
    verified_files(out, {**lock["inference_files"], **lock["validation_files"], "plan.json": lock["plan_sha256"]})
    # Only the locked per-shot model reaches this split. Parent predictions are archived.
    verified_files(args.previous, {"test_predictions.json": previous["files"]["test_predictions.json"]})
    reference = json_from(args.previous / "test_predictions.json")
    test, tasks = get_tasks(args, "test", release)
    encoded = ViewFeatures(out, "cuda").collect(test.images, lock["views"].values())
    results, raw = {}, {}
    for shot, name in lock["selected"].items():
        task = tasks[shot]
        x = representation(encoded, *lock["views"][shot])
        p = predictions(x, task, lock["rules"][shot], [name])[name]
        if name == "parent" and not np.array_equal(p, reference[shot]):
            raise ValueError("Archived parent test predictions changed")
        results[shot] = score(p, task[3], reference[shot])
        raw[shot] = p.tolist()
        print("Test", shot, name, results[shot]["correct"], flush=True)
    save_json(out / "test_predictions.json", raw)
    files = {**lock["validation_files"], **lock["inference_files"], **{n: digest_file(out / n) for n in
             ("plan.json", "validation_lock.json", "test_predictions.json")}}
    save_json(out / "results.json", {"track": plan["track"], "external_pretraining": False,
              "winner": lock["selected"], "val": lock["validation"], "test": results, "files": files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    parser.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/lab04_gpu_rotation_views"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lab04_gpu_task_subspace"))
    study(parser.parse_args())
