"""Compare three base-only metrics on a fixed self-trained ResNet12."""
import argparse
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.nn import functional as F

from .accuracy90_data import json_from
from .data import digest_file
from .evaluate_gpu_tracks import verified_files
from .experiment import summarize
from .multiscale_study import ViewFeatures, representation
from .prototype_restarts import RestartClassifier, predict_split
from .rotation_study import INFERENCE, evaluate, linear_sum_assignment
from .self_distill_study import correct_by_task, get_tasks
from .study_cached_refinement import paired_stats
from .train_episode_metric import (BATCH_TASKS, BRANCHES, HIDDEN, SEED, STEPS, data_args,
                                   model_for, prepare_features, train_branch)
from .train_gpu_metric import save_json


CANDIDATES = ("parent", *BRANCHES, "crop_covariance")


def crop_transform(original, cropped):
    """Whiten directions that change across crops, regularized by isotropic noise."""
    changes = original.double() - cropped.double()
    changes -= changes.mean(0)
    covariance = changes.T @ changes / (len(changes) - 1)
    eig, basis = torch.linalg.eigh(covariance.cpu())
    gain = (1 + eig.clamp_min(0) / eig.mean().clamp_min(1e-12)).rsqrt()
    return ((basis * gain) @ basis.T).float()


def load_metric(out, name, device):
    if name == "parent":
        return None
    payload = torch.load(out / (name + ".pt"), map_location="cpu", weights_only=True)
    if name == "crop_covariance":
        return payload.to(device)
    if name not in BRANCHES:
        raise ValueError(name)
    net = model_for(payload["residual.0.weight"].shape[1])
    net.load_state_dict(payload)
    return net.to(device).eval()


@torch.inference_mode()
def apply_metric(features, metric, device):
    if metric is None:
        return features
    parts = []
    for start in range(0, len(features), 1024):
        x = torch.from_numpy(features[start:start + 1024]).to(device)
        if isinstance(metric, torch.Tensor):
            z = F.normalize(x @ metric)
        else:
            # Fixed blend protects the pretrained geometry on novel classes.
            z = F.normalize(x + metric(x))
        parts.append(z.cpu().numpy())
    return np.concatenate(parts)


class MetricViews(ViewFeatures):
    def __init__(self, directory, device, metric):
        super().__init__(directory, device)
        self.device, self.metric = device, metric

    def collect(self, images, configs):
        configs = list(configs)
        if any(weight != 0 for _, weight in configs):
            raise ValueError("This study uses the frozen global/spatial representation")
        rows = super().collect(images, configs)
        for scale in {scale for scale, _ in configs}:
            rows[scale] = apply_metric(rows[scale], self.metric, self.device)
        return rows


class EpisodeMetricClassifier(RestartClassifier):
    def __init__(self, directory, device="cpu"):
        directory = Path(directory)
        self.device = torch.device(device)
        result = json_from(directory / "results.json")
        verified_files(directory, {"validation_lock.json": result["files"]["validation_lock.json"]})
        lock = json_from(directory / "validation_lock.json")
        verified_files(directory, lock["inference_files"])
        self.views, self.rules = lock["views"], lock["selected"]
        metric = load_metric(directory, lock["winner"], self.device)
        self.extractor = MetricViews(directory, self.device, metric)


def fit_all(args, plan):
    out = args.output
    prepare_features(args)
    features_record = json_from(args.cache / "features.json")
    orig, crop = [torch.from_numpy(np.load(args.cache / n)).cuda() for n in ("orig.npy", "crop.npy")]
    labels = np.load(args.cache / "labels.npy")
    groups = np.stack([np.flatnonzero(labels == c) for c in np.unique(labels)])
    if (groups.shape != (64, 600) or orig.shape != crop.shape or orig.shape != (len(labels), 448)
            or features_record["plan"]["train_ids"] != plan["train_ids"]):
        raise ValueError("Expected matched base training views and labels")
    history = {}
    for name in BRANCHES:
        net, history[name] = train_branch(name, orig, crop, groups)
        torch.save(net.state_dict(), out / (name + ".pt"))
    torch.save(crop_transform(orig, crop), out / "crop_covariance.pt")
    save_json(out / "training.json", {
        "plan_sha256": digest_file(out / "plan.json"), "features": features_record, "history": history,
        "parameters_per_residual_head": sum(p.numel() for p in net.parameters()),
        "files": {n + ".pt": digest_file(out / (n + ".pt")) for n in CANDIDATES if n != "parent"}})


def score(pred, truth, baseline):
    rates = correct_by_task(pred, truth)
    return {**summarize(rates), "correct": int((pred == truth).sum()), "queries": int(truth.size),
            "paired_vs_rotation_views": paired_stats(rates, correct_by_task(baseline, truth))}


def run_split(args, split_name, release, names, baseline, plan):
    split, tasks = get_tasks(args, split_name, release)
    views, rules = plan["views"], plan["selected"]
    extractor = ViewFeatures(args.previous, "cuda")
    rows = extractor.collect(split.images, views.values())
    scores, diagnostic, files = {}, {}, {}
    for name in names:
        scores[name], diagnostic[name], answers = {}, {}, {}
        metric = load_metric(args.output, name, "cuda")
        for shot, task in tasks.items():
            x = apply_metric(representation(rows, *views[shot]), metric, "cuda")
            rule = rules[shot]
            pred = predict_split(x, task, [rule])[rule]
            if name == "parent" and not np.array_equal(pred, baseline[shot]):
                raise ValueError("Recomputed parent predictions differ from the archived reference")
            scores[name][shot] = score(pred, task[3], baseline[shot])
            if split_name == "val":
                independent = evaluate(x, task, "support_only", "cuda")
                diagnostic[name][shot] = summarize(correct_by_task(independent, task[3]))
            answers[shot] = pred.tolist()
        file = ("val_" + name if split_name == "val" else "test_predictions") + ".json"
        save_json(args.output / file, answers)
        files[file] = digest_file(args.output / file)
        print(split_name, name, {s: r["correct"] for s, r in scores[name].items()}, flush=True)
    return scores, diagnostic, files


def study(args):
    torch.set_num_threads(2)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    previous = json_from(args.previous / "results.json")
    val_file = "val_" + previous["winner"] + ".json"
    verified_files(args.previous, {n: previous["files"][n] for n in
                                   (*INFERENCE, "plan.json", "validation_lock.json", val_file)})
    parent_plan = json_from(args.previous / "plan.json")
    parent = json_from(args.previous / "validation_lock.json")
    protocol = json_from(args.artifacts / "protocol.json")
    release = json_from(args.artifacts / "continued_results.json")
    episodes = parent_plan["episode_files"]
    if (previous["track"] != "rotation_views" or previous["external_pretraining"]
            or parent_plan["training"]["train_ids"] != protocol["train_selection_sha256"]
            or parent_plan["dataset_manifest_sha256"] != digest_file(args.artifacts / "dataset_manifest.json")
            or any(release["files"][n]["sha256"] != h for n, h in episodes.items())
            or any(w for _, w in parent["views"].values())):
        raise ValueError("Expected the same self-trained parent, splits and fixed tasks")
    source_files = ("episode_metric_study.py", "train_episode_metric.py", "learned_calibration.py",
                    "prototype_restarts.py", "rotation_study.py", "multiscale_study.py", "gpu_metric.py",
                    "self_distill_study.py", "evaluate_gpu_tracks.py", "supervised_whitening.py")
    plan = {"track": "episode_metric", "external_pretraining": False,
            "candidates": list(CANDIDATES), "selection": "Maximize total correct on both validation shots; first candidate wins ties.",
            "views": parent["views"], "selected": parent["selected"], "episode_files": episodes,
            "train_ids": protocol["train_selection_sha256"], "dataset_manifest_sha256": parent_plan["dataset_manifest_sha256"],
            "parent_sha256": digest_file(args.previous / "validation_lock.json"),
            "training": {"seed": SEED, "steps": STEPS, "batch_tasks": BATCH_TASKS, "hidden": HIDDEN,
                         "shots": [1, 5], "ways": 5, "queries_per_class": 15, "residual_inference_blend": .5,
                         "refinement_steps": 3, "balance_steps": 4, "mean_std_alignment_weight": .25,
                         "identity_penalty": .20, "crop_covariance_ridge": "mean eigenvalue"},
            "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "assignment_solver": "scipy" if linear_sum_assignment is not None else "numpy",
            "source_sha256": {n: digest_file(Path(__file__).with_name(n)) for n in source_files}}
    if (out / "plan.json").exists() and json_from(out / "plan.json") != plan:
        raise ValueError("Different study in output directory")
    if (out / "results.json").is_file():
        verified_files(out, json_from(out / "results.json")["files"])
        return
    save_json(out / "plan.json", plan)
    if not (out / "training.json").is_file():
        fit_all(args, plan)
    training = json_from(out / "training.json")
    verified_files(out, {**training["files"], "plan.json": training["plan_sha256"]})
    if not (out / "validation_lock.json").is_file():
        val_ref = json_from(args.previous / val_file)
        print("Validating fixed candidates", CANDIDATES, flush=True)
        scores, diagnostics, files = run_split(args, "val", release, CANDIDATES, val_ref, plan)
        winner = max(CANDIDATES, key=lambda n: sum(r["correct"] for r in scores[n].values()))
        inference = dict(parent["inference_files"])
        for n in inference:
            shutil.copyfile(args.previous / n, out / n)
        if winner != "parent":
            inference[winner + ".pt"] = training["files"][winner + ".pt"]
        save_json(out / "validation_lock.json", {"winner": winner, "views": plan["views"], "selected": plan["selected"],
                  "validation": scores[winner], "candidates": scores, "support_only_validation": diagnostics,
                  "inference_files": inference, "validation_files": files, "plan_sha256": training["plan_sha256"],
                  "training_sha256": digest_file(out / "training.json")})
    lock = json_from(out / "validation_lock.json")
    verified_files(out, {**lock["inference_files"], **lock["validation_files"], "plan.json": lock["plan_sha256"],
                         "training.json": lock["training_sha256"]})
    winner = lock["winner"]
    print("Locked", winner, flush=True)
    verified_files(args.previous, {"test_predictions.json": previous["files"]["test_predictions.json"]})
    test_ref = json_from(args.previous / "test_predictions.json")
    scores, _, files = run_split(args, "test", release, (winner,), test_ref, plan)
    files.update({**lock["validation_files"], **training["files"], **lock["inference_files"],
                  "plan.json": lock["plan_sha256"], "training.json": lock["training_sha256"],
                  "validation_lock.json": digest_file(out / "validation_lock.json")})
    save_json(out / "results.json", {"track": plan["track"], "external_pretraining": False, "winner": winner,
              "views": plan["views"], "selected": plan["selected"], "val": lock["validation"],
              "test": scores[winner], "files": files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    data_args(parser)
    parser.add_argument("--output", type=Path, default=Path("outputs/lab04_gpu_episode_metric"))
    study(parser.parse_args())
