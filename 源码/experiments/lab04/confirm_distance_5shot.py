"""Confirm the 25% spatial feature on new five-shot episodes from validation classes."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from .accuracy90_data import json_from, load_archive
from .data import digest_file, sample_episode
from .distance_covariance import fuse, image_features
from .distance_covariance_study import HEAD_FILE, load_head
from .episode_metric_study import score
from .evaluate_gpu_tracks import verified_files
from .multiscale_study import ViewFeatures
from .prototype_restarts import predict_split
from .self_distill_study import get_tasks
from .train_gpu_metric import save_json


def compare(split, task, source, reference, evaluate_quarter):
    extractor = ViewFeatures(source, "cuda")
    rows = extractor.collect(split.images, [("corners", 0.)])["corners"]
    ref = (predict_split(rows, task, ["average"])["average"]
           if reference is None else np.asarray(reference))
    pred = ref
    if evaluate_quarter:
        d = image_features(extractor.original.model.encoder, load_head(source, "cuda"), split.images, "cuda")
        pred = predict_split(fuse(rows, d, .25), task, ["average"])["average"]
    return ref, pred


def study(args):
    torch.set_num_threads(2)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    previous = json_from(args.source / "validation_lock.json")
    verified_files(args.source, {**previous["inference_files"], "training.json": previous["training_sha256"],
                                 "plan.json": previous["plan_sha256"]})
    training = json_from(args.source / "training.json")
    verified_files(args.source, {HEAD_FILE: training["head_sha256"]})
    prior_plan = json_from(args.source / "plan.json")
    if (prior_plan["external_pretraining"] or previous["winner"] != "parent"
            or previous["views"]["5"] != ["corners", 0.] or previous["selected"]["5"] != "average"
            or digest_file(args.artifacts / "dataset_manifest.json") != prior_plan["dataset_manifest_sha256"]):
        raise ValueError("Unexpected spatial study or data")
    plan = {"shot": 5, "ways": 5, "queries_per_class": 15, "episodes": 600, "validation_seed": 840517,
            "weight": .25, "selection": "Use quarter only if its paired 95% lower bound exceeds zero on fresh validation episodes.",
            "scope": "Exploratory follow-up on same 16 validation classes. Original two-shot decision is separate.",
            "source_lock_sha256": digest_file(args.source / "validation_lock.json"),
            "source_code_sha256": digest_file(Path(__file__)), "head_sha256": training["head_sha256"],
            "dataset_manifest_sha256": prior_plan["dataset_manifest_sha256"], "episode_files": prior_plan["episode_files"]}
    if (out / "plan.json").exists() and json_from(out / "plan.json") != plan:
        raise ValueError("Different follow-up in output directory")
    if (out / "results.json").exists():
        verified_files(out, json_from(out / "results.json")["files"])
        return
    save_json(out / "plan.json", plan)
    if not (out / "confirmation.json").is_file():
        val = load_archive(args.data_root, args.artifacts, "val")
        rng = np.random.default_rng(plan["validation_seed"])
        episodes = [sample_episode(val.labels, rng, shots=5, queries=15) for _ in range(plan["episodes"])]
        task = tuple(np.stack([row[k] for row in episodes]).astype(np.int64, copy=False) for k in
                     ("support_indices", "support_labels", "query_indices", "query_labels"))
        ids = hashlib.sha256("\n".join(val.ids).encode("utf-8")).hexdigest()
        ref, pred = compare(val, task, args.source, None, True)
        record = score(pred, task[3], ref)
        pair = record["paired_vs_rotation_views"]
        winner = "quarter" if pair["mean"] - pair["ci95"] > 0 else "parent"
        np.savez_compressed(out / "confirmation_episodes.npz", support=task[0], support_labels=task[1],
                            query=task[2], targets=task[3], parent=ref, quarter=pred)
        save_json(out / "confirmation.json", {"plan_sha256": digest_file(out / "plan.json"), "winner": winner,
                  "parent": score(ref, task[3], ref), "quarter": record, "archive_ordered_ids_sha256": ids,
                  "episodes_sha256": digest_file(out / "confirmation_episodes.npz")})
        print("Independent episode draw", int((ref == task[3]).sum()), record, "locked", winner, flush=True)
    lock = json_from(out / "confirmation.json")
    files = {"plan.json": lock["plan_sha256"], "confirmation_episodes.npz": lock["episodes_sha256"]}
    verified_files(out, files)
    test = None
    if lock["winner"] == "quarter":
        prior_result = json_from(args.source / "results.json")
        verified_files(args.source, {"test_predictions.json": prior_result["files"]["test_predictions.json"]})
        release = json_from(args.artifacts / "continued_results.json")
        data, tasks = get_tasks(args, "test", release)
        task = tasks["5"]
        ref, pred = compare(data, task, args.source, json_from(args.source / "test_predictions.json")["5"], True)
        test = score(pred, task[3], ref)
        save_json(out / "test_predictions.json", {"5": pred.tolist()})
        files["test_predictions.json"] = digest_file(out / "test_predictions.json")
    files["confirmation.json"] = digest_file(out / "confirmation.json")
    save_json(out / "results.json", {"winner": lock["winner"], "validation": lock["quarter"],
              "test_5shot_quarter": test, "files": files})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    p.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    p.add_argument("--source", type=Path, default=Path("outputs/lab04_gpu_distance_covariance"))
    p.add_argument("--output", type=Path, default=Path("outputs/lab04_gpu_distance_5shot"))
    study(p.parse_args())
