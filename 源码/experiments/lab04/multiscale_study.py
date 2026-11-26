"""Validate multiscale views and a base-trained local metric against the fixed ResNet12."""
import argparse
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from .accuracy90_data import json_from, load_archive
from .data import digest_file
from .evaluate_gpu_tracks import task_indices, verified_files
from .experiment import summarize
from .rotation_study import (INFERENCE, center_features, combine, evaluate, load_extractor,
                             predict_embeddings)
from .study_cached_refinement import paired_stats
from .train_gpu_metric import reference_rows, save_json
from .train_local_metric import LocalFeatures


CANDIDATES = tuple((scale, weight) for scale in ("center", "multiscale", "corners")
                   for weight in (0., .25, .5))
LOCAL_FILE = "local_metric.pt"


def resized_features(extractor, images, side, top, left):
    pieces = []
    for start in range(0, len(images), 256):
        crops = [np.asarray(Image.fromarray(img[top:top + side, left:left + side]).resize(
            (84, 84), Image.Resampling.BICUBIC)) for img in images[start:start + 256]]
        pieces.append(extractor(np.stack(crops)))
    return np.concatenate(pieces)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class ViewFeatures:
    def __init__(self, directory, device):
        self.original = load_extractor(directory, device)
        self.local = LocalFeatures(self.original.model, device)
        self.local_weights = directory / LOCAL_FILE

    def collect(self, images, configs):
        scales = {scale for scale, _ in configs}
        old = combine(self.original(images), center_features(self.original, images))
        rows = {"center": old}
        if "multiscale" in scales or "corners" in scales:
            small = resized_features(self.original, images, 58, 13, 13)
            rows["multiscale"] = unit(2 * old + small)
        if "corners" in scales:
            extra = sum(resized_features(self.original, images, 70, top, left)
                        for top, left in ((0, 0), (0, 14), (14, 0), (14, 14)))
            rows["corners"] = unit(2 * old + small + extra * .5)
        if any(weight > 0 for _, weight in configs):
            if self.local.head is None:
                self.local.load_head(self.local_weights)
            rows["local"] = combine(self.local(images), center_features(self.local, images))
        return rows


def representation(rows, scale, weight):
    if weight == 0:
        return rows[scale]
    return np.concatenate([np.sqrt(1 - weight) * rows[scale],
                           np.sqrt(weight) * rows["local"]], axis=-1).astype(np.float32)


class MultiscaleClassifier:
    def __init__(self, directory, device="cpu"):
        directory = Path(directory)
        self.device = torch.device(device)
        result = json_from(directory / "results.json")
        lock = json_from(directory / "validation_lock.json")
        verified_files(directory, {**lock["inference_files"],
                                   "validation_lock.json": result["files"]["validation_lock.json"]})
        self.selected = lock["selected"]
        self.extractor = ViewFeatures(directory, self.device)

    @torch.inference_mode()
    def predict(self, support, labels, query):
        labels = np.asarray(labels)
        classes, ids, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if (labels.ndim != 1 or len(labels) != len(support) or len(classes) != 5
                or np.any(counts != counts[0]) or counts[0] not in (1, 5) or len(query) == 0 or len(query) % 5):
            raise ValueError("Expected five balanced classes with one or five labeled supports")
        images = np.concatenate((support, query))
        if images.dtype != np.uint8 or images.shape[1:] != (84, 84, 3):
            raise ValueError("Expected 84x84 RGB uint8 images")
        config = self.selected[str(counts[0])]
        rows = self.extractor.collect(images, [config])
        features = torch.from_numpy(representation(rows, *config)).to(self.device)
        y = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        answers = predict_embeddings(features[None, :len(y)], y[None], features[None, len(y):], "diffusion")
        return classes[answers[0]]


def measures(pred, truth, baseline):
    if pred.shape != (600, 75) or truth.shape != pred.shape or baseline.shape != pred.shape:
        raise ValueError("Bad fixed episode shapes")
    values = (pred == truth).mean(axis=1)
    return {**summarize(values), "queries": int(truth.size), "correct": int((pred == truth).sum()),
            "paired_vs_rotation": paired_stats(values, (baseline == truth).mean(axis=1))}


def select(rows, tasks, parent_pred):
    candidates = {shot: [] for shot in tasks}
    best, predictions, selected = {}, {}, {}
    for scale, weight in CANDIDATES:
        features = representation(rows, scale, weight)
        for shot, task in tasks.items():
            pred = evaluate(features, task, "diffusion", "cuda")
            baseline = np.asarray(parent_pred[shot])
            if (scale, weight) == ("center", 0.) and not np.array_equal(pred, baseline):
                raise ValueError("Prior validation predictions failed to replay")
            score = measures(pred, task[3], baseline)
            candidates[shot].append({"config": [scale, weight], **score})
            if shot not in best or score["correct"] > best[shot]["correct"]:
                best[shot], predictions[shot], selected[shot] = score, pred.tolist(), [scale, weight]
        print("validation", scale, weight,
              {s: round(candidates[s][-1]["accuracy"] * 100, 3) for s in tasks}, flush=True)
    return candidates, best, predictions, selected


def study(args):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA; the exported classifier also runs on CPU")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    previous = json_from(args.previous / "results.json")
    training = json_from(args.trained / "training.json")
    training_plan = json_from(args.trained / "training_plan.json")
    if (previous["external_pretraining"] or training_plan["external_pretraining"]
            or training_plan["model_sha256"] != previous["files"]["medium.pt"]
            or training_plan["train_ids"] != json_from(args.artifacts / "protocol.json")["train_selection_sha256"]):
        raise ValueError("Expected matching base-only weights")
    verified_files(args.trained, training["files"])
    verified_files(args.previous, {k: previous["files"][k] for k in (*INFERENCE, "val_predictions.json")})
    release = json_from(args.artifacts / "continued_results.json")
    plan = {"track": "multiscale", "external_pretraining": False, "candidates": [list(c) for c in CANDIDATES],
            "selection": "For each shot choose the most correct validation queries. Ties favor earlier candidates.",
            "previous_results_sha256": digest_file(args.previous / "results.json"),
            "training_plan": training_plan, "training": training, "gpu": torch.cuda.get_device_name(),
            "files": {n: digest_file(Path(__file__).with_name(n)) for n in
                      ("multiscale_study.py", "rotation_study.py", "train_local_metric.py", "gpu_metric.py")},
            "dataset_manifest_sha256": digest_file(args.artifacts / "dataset_manifest.json"),
            "episodes": {n: release["files"][n]["sha256"] for n in
                         ("continued_validation_tasks.jsonl", "continued_episodes.json")}}
    if (out / "plan.json").exists() and json_from(out / "plan.json") != plan:
        raise ValueError("Output directory belongs to a different experiment")
    if (out / "results.json").exists():
        verified_files(out, json_from(out / "results.json")["files"])
        print("Verified", out / "results.json", flush=True)
        return
    save_json(out / "plan.json", plan)
    for file in INFERENCE:
        shutil.copyfile(args.previous / file, out / file)
    shutil.copyfile(args.trained / LOCAL_FILE, out / LOCAL_FILE)
    files = {n: previous["files"][n] for n in INFERENCE}
    files[LOCAL_FILE] = training["files"][LOCAL_FILE]
    verified_files(out, files)
    extractor = ViewFeatures(out, "cuda")

    if not (out / "validation_lock.json").exists():
        val = load_archive(args.data_root, args.artifacts, "val")
        episodes = reference_rows(args.artifacts, "val", release)
        tasks = {str(s): task_indices(val, episodes, s) for s in (1, 5)}
        print("Encoding the 16 validation classes", flush=True)
        rows = extractor.collect(val.images, CANDIDATES)
        scores, best, preds, selected = select(rows, tasks, json_from(args.previous / "val_predictions.json"))
        save_json(out / "val_predictions.json", preds)
        save_json(out / "validation_lock.json", {"selected": selected, "validation": best, "candidates": scores,
                  "inference_files": files, "plan_sha256": digest_file(out / "plan.json"),
                  "val_predictions_sha256": digest_file(out / "val_predictions.json")})
        del val, rows, preds
    lock = json_from(out / "validation_lock.json")
    verified_files(out, {**lock["inference_files"], "plan.json": lock["plan_sha256"],
                         "val_predictions.json": lock["val_predictions_sha256"]})
    selected = lock["selected"]
    print("Locked configurations", selected, flush=True)
    verified_files(args.previous, {"test_predictions.json": previous["files"]["test_predictions.json"]})
    reference = json_from(args.previous / "test_predictions.json")
    if all(c == ["center", 0.] for c in selected.values()):
        print("Validation retained the previous classifier", flush=True)
        test = {s: {**{k: v for k, v in row.items() if k != "paired_vs_medium"},
                    "paired_vs_rotation": paired_stats(np.zeros(600), np.zeros(600))}
                for s, row in previous["test"].items()}
        answers = reference
    else:
        test_data = load_archive(args.data_root, args.artifacts, "test")
        test_rows = reference_rows(args.artifacts, "test", release)
        rows = extractor.collect(test_data.images, selected.values())
        test, answers = {}, {}
        for shot, config in selected.items():
            task = task_indices(test_data, test_rows, int(shot))
            pred = evaluate(representation(rows, *config), task, "diffusion", "cuda")
            test[shot] = measures(pred, task[3], np.asarray(reference[shot]))
            answers[shot] = pred.tolist()
            print("test", shot, test[shot], flush=True)
    save_json(out / "test_predictions.json", answers)
    names = (*files, "plan.json", "validation_lock.json", "val_predictions.json", "test_predictions.json")
    save_json(out / "results.json", {"track": "multiscale", "external_pretraining": False, "selected": selected,
              "val": lock["validation"], "test": test, "files": {n: digest_file(out / n) for n in names}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    p.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    p.add_argument("--previous", type=Path, default=Path("outputs/lab04_gpu_rotation"))
    p.add_argument("--trained", type=Path, default=Path(".local/lab04-local-metric"))
    p.add_argument("--output", type=Path, default=Path("outputs/lab04_gpu_multiscale"))
    study(p.parse_args())
