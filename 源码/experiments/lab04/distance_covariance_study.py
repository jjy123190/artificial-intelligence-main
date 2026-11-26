"""Validate channel-distance embeddings against the same fixed ResNet12 classifier."""
import argparse
from pathlib import Path
import shutil

import numpy as np
import torch

from .accuracy90_data import json_from
from .data import digest_file
from .distance_covariance import (BATCH_TASKS, CHANNELS, DIMENSION, SEED, STEPS, DistanceHead,
                                 fuse, image_features, train_head)
from .episode_metric_study import score
from .evaluate_gpu_tracks import verified_files
from .multiscale_study import ViewFeatures
from .prototype_restarts import RestartClassifier, predict_split
from .rotation_study import INFERENCE, linear_sum_assignment
from .self_distill_study import get_tasks
from .train_episode_metric import data_args
from .train_gpu_metric import save_json


WEIGHTS = {"parent": 0., "quarter": .25, "half": .50}
HEAD_FILE = "distance_head.pt"


def load_head(directory, device):
    head = DistanceHead()
    head.load_state_dict(torch.load(directory / HEAD_FILE, map_location="cpu", weights_only=True))
    return head.eval().to(device)


class SpatialViews(ViewFeatures):
    def __init__(self, directory, device, weight):
        super().__init__(directory, device)
        self.device, self.weight = device, weight
        self.head = load_head(directory, device) if weight else None

    def collect(self, images, configs):
        configs = list(configs)
        rows = super().collect(images, configs)
        if self.weight:
            d = image_features(self.original.model.encoder, self.head, images, self.device)
            for scale in {scale for scale, _ in configs}:
                rows[scale] = fuse(rows[scale], d, self.weight)
        return rows


class DistanceClassifier(RestartClassifier):
    def __init__(self, directory, device="cpu"):
        directory = Path(directory)
        self.device = torch.device(device)
        result = json_from(directory / "results.json")
        verified_files(directory, {"validation_lock.json": result["files"]["validation_lock.json"]})
        lock = json_from(directory / "validation_lock.json")
        verified_files(directory, lock["inference_files"])
        self.views, self.rules = lock["views"], lock["selected"]
        self.extractor = SpatialViews(directory, self.device, lock["weight"])


def evaluate_split(args, split_name, candidates, plan, release, ref):
    split, tasks = get_tasks(args, split_name, release)
    extractor = ViewFeatures(args.previous, "cuda")
    rows = extractor.collect(split.images, plan["views"].values())
    distances = None
    if any(WEIGHTS[name] for name in candidates):
        distances = image_features(extractor.original.model.encoder, load_head(args.output, "cuda"),
                                   split.images, "cuda")
    scores, files = {}, {}
    for name in candidates:
        predictions, scores[name] = {}, {}
        for shot, task in tasks.items():
            scale, _ = plan["views"][shot]
            x = fuse(rows[scale], distances, WEIGHTS[name]) if WEIGHTS[name] else rows[scale]
            rule = plan["selected"][shot]
            pred = predict_split(x, task, [rule])[rule]
            if name == "parent" and not np.array_equal(pred, ref[shot]):
                raise ValueError("Fixed reference failed to reproduce")
            predictions[shot] = pred.tolist()
            scores[name][shot] = score(pred, task[3], ref[shot])
        file = f"val_{name}.json" if split_name == "val" else "test_predictions.json"
        save_json(args.output / file, predictions)
        files[file] = digest_file(args.output / file)
        print(split_name, name, {s: r["correct"] for s, r in scores[name].items()}, flush=True)
    return scores, files


def study(args):
    torch.set_num_threads(2)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    previous = json_from(args.previous / "results.json")
    val_file = "val_" + previous["winner"] + ".json"
    verified_files(args.previous, {n: previous["files"][n] for n in
                                   (*INFERENCE, "validation_lock.json", "plan.json", val_file)})
    parent = json_from(args.previous / "validation_lock.json")
    parent_plan = json_from(args.previous / "plan.json")
    source = ("distance_covariance.py", "distance_covariance_study.py", "episode_metric_study.py",
              "train_episode_metric.py", "prototype_restarts.py", "gpu_metric.py", "multiscale_study.py",
              "rotation_study.py", "evaluate_gpu_tracks.py", "self_distill_study.py")
    release = json_from(args.artifacts / "continued_results.json")
    episodes = parent_plan["episode_files"]
    expected_ids = json_from(args.artifacts / "protocol.json")["train_selection_sha256"]
    if (previous["external_pretraining"] or previous["track"] != "rotation_views"
            or any(w for _, w in parent["views"].values())
            or parent_plan["training"]["train_ids"] != expected_ids
            or digest_file(args.artifacts / "dataset_manifest.json") != parent_plan["dataset_manifest_sha256"]
            or any(release["files"][n]["sha256"] != sha for n, sha in episodes.items())):
        raise ValueError("Expected fixed views, base classes and episodes")
    plan = {"track": "distance_covariance", "external_pretraining": False, "weights": WEIGHTS,
            "selection": "One weight for both shots. Maximize combined validation correct; favor lower weight on ties.",
            "views": parent["views"], "selected": parent["selected"],
            "parent_lock_sha256": digest_file(args.previous / "validation_lock.json"), "episode_files": episodes,
            "train_ids": expected_ids, "dataset_manifest_sha256": parent_plan["dataset_manifest_sha256"],
            "training": {"steps": STEPS, "episodes_per_step": BATCH_TASKS, "alternating_shots": [1, 5],
                         "queries_per_class": 15, "seed": SEED, "projection_channels": CHANNELS, "dimension": DIMENSION,
                         "learning_rate": .0007, "alignment_weight": .25, "feature_maps": "5x5, original and flip"},
            "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "assignment_solver": "scipy" if linear_sum_assignment is not None else "numpy",
            "sources": {n: digest_file(Path(__file__).with_name(n)) for n in source}}
    if (out / "plan.json").exists() and json_from(out / "plan.json") != plan:
        raise ValueError("Different experiment in output directory")
    if (out / "results.json").is_file():
        verified_files(out, json_from(out / "results.json")["files"])
        return
    save_json(out / "plan.json", plan)
    if not (out / "training.json").is_file():
        net, history = train_head(args)
        torch.save(net.state_dict(), out / HEAD_FILE)
        save_json(out / "training.json", {"plan_sha256": digest_file(out / "plan.json"),
                  "maps": json_from(args.map_cache / "maps.json"), "history": history,
                  "parameters": sum(p.numel() for p in net.parameters()), "head_sha256": digest_file(out / HEAD_FILE)})
    training = json_from(out / "training.json")
    verified_files(out, {HEAD_FILE: training["head_sha256"], "plan.json": training["plan_sha256"]})
    if not (out / "validation_lock.json").is_file():
        scores, files = evaluate_split(args, "val", WEIGHTS, plan, release, json_from(args.previous / val_file))
        winner = max(WEIGHTS, key=lambda n: sum(v["correct"] for v in scores[n].values()))
        inference = dict(parent["inference_files"])
        for name in inference:
            shutil.copyfile(args.previous / name, out / name)
        if WEIGHTS[winner]:
            inference[HEAD_FILE] = training["head_sha256"]
        save_json(out / "validation_lock.json", {"winner": winner, "weight": WEIGHTS[winner],
                  "views": plan["views"], "selected": plan["selected"], "validation": scores[winner],
                  "candidates": scores, "inference_files": inference, "validation_files": files,
                  "training_sha256": digest_file(out / "training.json"), "plan_sha256": training["plan_sha256"]})
    lock = json_from(out / "validation_lock.json")
    verified_files(out, {**lock["validation_files"], **lock["inference_files"],
                         "plan.json": lock["plan_sha256"], "training.json": lock["training_sha256"]})
    print("Locked spatial weight", lock["weight"], flush=True)
    verified_files(args.previous, {"test_predictions.json": previous["files"]["test_predictions.json"]})
    winner = lock["winner"]
    scores, files = evaluate_split(args, "test", (winner,), plan, release, json_from(args.previous / "test_predictions.json"))
    files.update({**lock["validation_files"], **lock["inference_files"], HEAD_FILE: training["head_sha256"],
                  "training.json": lock["training_sha256"], "plan.json": lock["plan_sha256"],
                  "validation_lock.json": digest_file(out / "validation_lock.json")})
    save_json(out / "results.json", {"track": plan["track"], "winner": winner, "external_pretraining": False,
              "weight": lock["weight"], "views": plan["views"], "selected": plan["selected"],
              "val": lock["validation"], "test": scores[winner], "files": files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    data_args(parser)
    parser.add_argument("--map-cache", type=Path, default=Path(".local/lab04-distance-covariance"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lab04_gpu_distance_covariance"))
    study(parser.parse_args())
