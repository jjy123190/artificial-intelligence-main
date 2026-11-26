"""Select one base-only ResNet12 and its episode classifier on paired validation tasks."""
import argparse
from itertools import product
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from .accuracy90_data import json_from, load_archive
from .data import digest_file
from .evaluate_gpu_tracks import ResNetFeatures, RULES, predict_tasks, task_indices, verified_files
from .experiment import summarize
from .gpu_metric import SpatialMetric
from .multiview import capacity_predictions
from .study_cached_refinement import paired_stats
from .train_gpu_metric import reference_rows, save_json


VIEWS = ("full", "full_center")
DECISIONS = ("prototype", "diffusion", "spectral")
INFERENCE = ("medium.pt", "spatial_metric.pt", "base_transform.npz")


def load_extractor(directory, device, batch=32):
    extractor = ResNetFeatures(directory, device, batch)
    state = torch.load(directory / "spatial_metric.pt", map_location="cpu", weights_only=True)
    head = SpatialMetric(state["center"], state["scale"], state["projection.weight"].shape[0])
    head.load_state_dict(state)
    extractor.head = head.to(device).eval()
    with np.load(directory / "base_transform.npz") as data:
        extractor.transform = dict(data)
    return extractor


def center_features(extractor, images):
    # Same crop/resize and encoder for one live image or an archived split.
    features = []
    for start in range(0, len(images), 256):
        batch = images[start:start + 256]
        crops = [np.asarray(Image.fromarray(img[7:77, 7:77]).resize((84, 84), Image.Resampling.BICUBIC))
                 for img in batch]
        features.append(extractor(np.stack(crops)))
    return np.concatenate(features)


def combine(full, center):
    features = full + center
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def diffusion_predict(support, labels, query, *, spectral=False):
    # Graph smoothing uses only the images in each individual episode.
    x = F.normalize(torch.cat([support, query], dim=1), dim=-1)
    x = F.normalize(x - x.mean(1, keepdim=True), dim=-1)
    sim = x @ x.transpose(1, 2)
    sim.diagonal(dim1=1, dim2=2).fill_(-torch.inf)
    top, idx = sim.topk(min(7, sim.size(-1) - 1), dim=-1)
    affinity = torch.zeros_like(sim).scatter_(-1, idx, (top / .12).softmax(-1))
    affinity = (affinity + affinity.transpose(1, 2)) / 2
    weights = affinity / affinity.sum(-1, keepdim=True).clamp_min(1e-12)
    x = F.normalize(.65 * x + .35 * (weights @ x), dim=-1)
    if spectral:
        # The top nontrivial eigenvectors describe the five clusters in this task.
        # Concatenation makes cosine similarities invariant to eigenvector signs.
        inv_degree = affinity.sum(-1).clamp_min(1e-12).rsqrt()
        graph = affinity * inv_degree.unsqueeze(2) * inv_degree.unsqueeze(1)
        _, vectors = torch.linalg.eigh(graph.cpu().double())
        coordinates = F.normalize(vectors[:, :, -5:-1].to(x), dim=-1)
        x = F.normalize(torch.cat([x, .5 * coordinates], dim=-1), dim=-1)
    s, q = x.split([support.size(1), query.size(1)], dim=1)
    label_matrix = F.one_hot(labels, 5).to(x.dtype)
    prior = 3 * (label_matrix.transpose(1, 2) @ s)
    proto = F.normalize(prior, dim=-1)
    for _ in range(12):
        assignments = ((q @ proto.transpose(1, 2)) / .08).softmax(-1)
        for _ in range(32):
            assignments = F.normalize(assignments, p=1, dim=1) * (q.size(1) / 5)
            assignments = F.normalize(assignments, p=1, dim=-1)
        proto = F.normalize(prior + assignments.transpose(1, 2) @ q, dim=-1)
    logits = (q @ proto.transpose(1, 2)) / .08
    return logits.softmax(-1)


def balanced_labels(probabilities, query):
    # Use canonical feature order to break tied scores independently of API order.
    if linear_sum_assignment is None:
        return capacity_predictions(probabilities, query)
    count, ways = probabilities.shape
    if count % ways:
        raise ValueError("Expected an equal number of queries per class")
    slots = np.repeat(np.arange(ways), count // ways)
    order = np.lexsort(query.T[::-1])
    log_costs = -np.log(probabilities.astype(np.float64).clip(min=1e-15))
    rows, cols = linear_sum_assignment(log_costs[order][:, slots])
    predictions = np.empty(count, dtype=np.int64)
    predictions[order[rows]] = slots[cols]
    return predictions


def predict_embeddings(s, labels, q, decision):
    if decision == "support_only":
        return predict_tasks(s, labels, q, RULES[0]).cpu().numpy()
    if decision == "prototype":
        return predict_tasks(s, labels, q, RULES[1]).cpu().numpy()
    if decision not in ("diffusion", "spectral"):
        raise ValueError("Unknown decision rule")
    probabilities = diffusion_predict(s, labels, q, spectral=decision == "spectral").cpu().numpy()
    return np.stack([balanced_labels(p, x) for p, x in zip(probabilities, q.cpu().numpy())])


@torch.inference_mode()
def evaluate(features, tasks, decision, device):
    supports, labels, queries, _ = tasks
    answers = []
    for start in range(0, len(queries), 32):
        sl = slice(start, start + 32)
        inputs = [torch.from_numpy(x).to(device) for x in
                  (features[supports[sl]], labels[sl], features[queries[sl]])]
        answers.append(predict_embeddings(*inputs, decision))
    return np.concatenate(answers)


def metrics(answers, truth, baseline):
    if answers.shape != truth.shape or baseline.shape != truth.shape:
        raise ValueError("Episode predictions must match released task shapes")
    values = (answers == truth).mean(1)
    return {**summarize(values), "correct": int((answers == truth).sum()), "queries": int(truth.size),
            "paired_vs_medium": paired_stats(values, (baseline == truth).mean(1))}


class RotationClassifier:
    def __init__(self, directory, device):
        directory = Path(directory)
        result = json_from(directory / "results.json")
        verified_files(directory, {k: result["files"][k] for k in (*INFERENCE, "validation_lock.json")})
        self.config = json_from(directory / "validation_lock.json")["selected"]
        self.device = torch.device(device)
        self.extractor = load_extractor(directory, self.device)

    @torch.inference_mode()
    def predict(self, support, labels, query):
        labels = np.asarray(labels)
        classes, ids, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if (labels.ndim != 1 or len(support) != len(labels) or len(classes) != 5 or
                np.any(counts != counts[0]) or counts[0] not in (1, 5) or not len(query) or len(query) % 5):
            raise ValueError("Expected a balanced five-way task with one or five labeled supports per class")
        images = np.concatenate([support, query])
        if images.dtype != np.uint8 or images.shape[1:] != (84, 84, 3):
            raise ValueError("Expected 84x84 uint8 RGB images")
        shot = str(counts[0])
        config = self.config["shots"][shot]
        x = self.extractor(images)
        if config["view"] == "full_center":
            x = combine(x, center_features(self.extractor, images))
        x = torch.from_numpy(x).to(self.device)
        y = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        prediction = predict_embeddings(x[None, :len(y)], y[None], x[None, len(y):], config["decision"])
        return classes[prediction[0]]


def fit_and_select(args):
    out, previous = args.output, args.previous
    if not torch.cuda.is_available() or args.batch_size < 1:
        raise ValueError("Use a working CUDA device and a positive feature batch size")
    out.mkdir(parents=True, exist_ok=True)
    prior = json_from(previous / "results.json")
    verified_files(previous, {n: prior["files"][n] for n in (*INFERENCE, "plan.json", "val_predictions.json")})
    trained = args.trained
    training_plan = json_from(trained / "training_plan.json")
    history = json_from(trained / "training.json")
    if (training_plan["external_weights"] or training_plan["parent_sha256"] != prior["files"]["medium.pt"]
            or [row["epoch"] for row in history] != list(range(1, training_plan["epochs"] + 1))):
        raise ValueError("Expected a completed continuation of the published self-trained model")
    source = Path(__file__)
    release = json_from(args.artifacts / "continued_results.json")
    plan = {"track": "rotation", "external_pretraining": False, "gpu": torch.cuda.get_device_name(),
            "previous_sha256": digest_file(previous / "results.json"), "training": training_plan,
            "torch": str(torch.__version__), "feature_batch_size": args.batch_size,
            "assignment_solver": "scipy" if linear_sum_assignment is not None else "numpy",
            "continued_sha256": digest_file(trained / "medium.pt"),
            "eval_source_sha256": {p.name: digest_file(p) for p in
                                  (source, source.with_name("multiview.py"), source.with_name("gpu_metric.py"),
                                   source.with_name("evaluate_gpu_tracks.py"))},
            "views": list(VIEWS), "decisions": list(DECISIONS),
            "diagnostic": "Validation only: full view and support-only cosine prototypes for both backbones.",
            "selection": "For each backbone and shot maximize the number of correct validation predictions. Choose one backbone by the total across both equal-size shot groups. Ties favor the earlier configuration.",
            "archives": digest_file(args.artifacts / "dataset_manifest.json"),
            "episode_files": {n: release["files"][n]["sha256"] for n in
                              ("continued_validation_tasks.jsonl", "continued_episodes.json")}}
    if (out / "plan.json").exists() and json_from(out / "plan.json") != plan:
        raise ValueError("Use a separate output directory for a changed plan")
    if (out / "results.json").exists():
        verified_files(out, json_from(out / "results.json")["files"])
        print(out / "results.json", flush=True)
        return
    save_json(out / "plan.json", plan)
    device = torch.device("cuda")
    roots = {"medium": previous, "rotation": trained}
    if not (out / "validation_lock.json").exists():
        print("Fitting the continued backbone's train-only metric", flush=True)
        new = ResNetFeatures(trained, device, args.batch_size)
        new.fit_base(load_archive(args.data_root, args.artifacts, "train"), trained)
        split = load_archive(args.data_root, args.artifacts, "val")
        rows = reference_rows(args.artifacts, "val", release)
        tasks = {str(s): task_indices(split, rows, s) for s in (1, 5)}
        baseline = json_from(previous / "val_predictions.json")
        candidates, best_scores, best_configs, predictions, support_only = {}, {}, {}, {}, {}
        for name, root in roots.items():
            extractor = new if name == "rotation" else load_extractor(root, device, args.batch_size)
            full = extractor(split.images)
            with_center = combine(full, center_features(extractor, split.images))
            candidates[name], best_scores[name], best_configs[name], predictions[name] = {}, {}, {}, {}
            support_only[name] = {}
            for shot, task in tasks.items():
                independent = evaluate(full, task, "support_only", device)
                support_only[name][shot] = summarize((independent == task[3]).mean(axis=1))
                scores, answers = {}, {}
                for view, decision in product(VIEWS, DECISIONS):
                    key = f"{view}/{decision}"
                    answers[key] = evaluate(full if view == "full" else with_center, task, decision, device)
                    scores[key] = metrics(answers[key], task[3], np.asarray(baseline[shot]))
                if name == "medium" and not np.array_equal(answers["full/prototype"], baseline[shot]):
                    raise ValueError("Published validation predictions failed to replay")
                best = max(scores, key=lambda k: scores[k]["correct"])
                view, decision = best.split("/")
                candidates[name][shot] = scores
                best_scores[name][shot] = scores[best]
                best_configs[name][shot] = {"view": view, "decision": decision}
                predictions[name][shot] = answers[best].tolist()
                print("val", name, shot, best, scores[best]["accuracy"], flush=True)
        winner = max(roots, key=lambda name: sum(row["correct"] for row in best_scores[name].values()))
        selected = {"backbone": winner, "shots": best_configs[winner]}
        save_json(out / "val_predictions.json", predictions[winner])
        save_json(out / "validation_lock.json", {"selected": selected, "candidates": candidates,
                  "support_only_full_view": support_only,
                  "validation": best_scores[winner], "plan_sha256": digest_file(out / "plan.json"),
                  "val_predictions_sha256": digest_file(out / "val_predictions.json"),
                  "inference_files": {n: digest_file(roots[winner] / n) for n in INFERENCE}})
        del split
    lock = json_from(out / "validation_lock.json")
    if lock["plan_sha256"] != digest_file(out / "plan.json"):
        raise ValueError("Validation selection belongs to another experiment")
    verified_files(out, {"val_predictions.json": lock["val_predictions_sha256"]})
    selected = lock["selected"]
    winner = roots[selected["backbone"]]
    verified_files(winner, lock["inference_files"])
    for name in INFERENCE:
        shutil.copyfile(winner / name, out / name)
    shutil.copyfile(trained / "training.json", out / "continuation_training.json")
    verified_files(out, lock["inference_files"])
    print("Locked configuration", selected, "starting fixed test", flush=True)
    split = load_archive(args.data_root, args.artifacts, "test")
    rows = reference_rows(args.artifacts, "test", release)
    verified_files(previous, {"test_predictions.json": prior["files"]["test_predictions.json"]})
    baseline = json_from(previous / "test_predictions.json")
    extractor = load_extractor(out, device, args.batch_size)
    full = extractor(split.images)
    center = (combine(full, center_features(extractor, split.images))
              if any(c["view"] == "full_center" for c in selected["shots"].values()) else None)
    test, answers = {}, {}
    for shot, config in selected["shots"].items():
        task = task_indices(split, rows, int(shot))
        pred = evaluate(full if config["view"] == "full" else center, task, config["decision"], device)
        test[shot] = metrics(pred, task[3], np.asarray(baseline[shot]))
        answers[shot] = pred.tolist()
        print("test", shot, test[shot], flush=True)
    save_json(out / "test_predictions.json", answers)
    files = (*INFERENCE, "plan.json", "validation_lock.json", "val_predictions.json", "test_predictions.json",
             "continuation_training.json")
    save_json(out / "results.json", {"track": "rotation", "external_pretraining": False, "selected": selected,
              "val": lock["validation"], "test": test, "files": {n: digest_file(out / n) for n in files}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    p.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    p.add_argument("--previous", type=Path, default=Path("outputs/lab04_gpu_medium"))
    p.add_argument("--trained", type=Path, default=Path(".local/lab04-rotation"))
    p.add_argument("--output", type=Path, default=Path("outputs/lab04_gpu_rotation"))
    p.add_argument("--batch-size", type=int, default=32)
    torch.set_num_threads(2)
    fit_and_select(p.parse_args())
