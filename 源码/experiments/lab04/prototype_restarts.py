"""Compare single and multiple prototype initializations on the same frozen embeddings."""
import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .accuracy90_data import json_from, load_archive
from .data import digest_file
from .evaluate_gpu_tracks import task_indices, verified_files
from .multiscale_study import measures, representation, ViewFeatures
from .rotation_study import balanced_labels, predict_embeddings
from .train_gpu_metric import reference_rows, save_json


RULES = ("single", "best", "average")
PERMS = list(permutations(range(5)))


def smooth(s, q):
    x = F.normalize(torch.cat([s, q], dim=1), dim=-1)
    x = F.normalize(x - x.mean(1, keepdim=True), dim=-1)
    similarities = x @ x.transpose(-1, -2)
    similarities.diagonal(dim1=1, dim2=2).fill_(-torch.inf)
    val, idx = similarities.topk(min(7, x.size(1) - 1))
    w = torch.zeros_like(similarities).scatter_(-1, idx, (val / .12).softmax(-1))
    w = w + w.transpose(1, 2)
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    x = F.normalize(.65 * x + .35 * (w @ x), dim=-1)
    return x.split((s.size(1), q.size(1)), dim=1)


def initial_centers(q, anchors):
    # Farthest-first starts prefer dense regions of the query cloud.
    b, n, _ = q.shape
    sim = q @ q.transpose(1, 2)
    density = sim.topk(min(6, n)).values.mean(-1)
    density = .6 + .4 * (density - density.amin(-1, keepdim=True)) / (
        density.amax(-1, keepdim=True) - density.amin(-1, keepdim=True)).clamp_min(1e-8)
    batch = torch.arange(b, device=q.device)
    first = density.argmax(1)
    seeds = [first]
    far = 1 - sim[batch, first]
    for _ in range(min(6, n) - 1):
        nxt = (far * density).argmax(-1)
        seeds.append(nxt)
        far = torch.minimum(far, 1 - sim[batch, nxt])
    starts = torch.stack(seeds, dim=1)
    chosen = [starts]
    far = 1 - sim[batch[:, None], starts]
    for _ in range(4):
        nxt = (far * density[:, None]).argmax(-1)
        chosen.append(nxt)
        far = torch.minimum(far, 1 - sim[batch[:, None], nxt])
    centers = q[batch[:, None, None], torch.stack(chosen, dim=2)]
    perm = torch.tensor(PERMS, device=q.device)
    matching = anchors[:, None] @ centers.transpose(-1, -2)
    scores = torch.einsum("brij,pij->brp", matching, F.one_hot(perm, 5).to(q.dtype))
    indices = perm[scores.argmax(-1)]
    centers = centers.gather(2, indices.unsqueeze(-1).expand_as(centers))
    return torch.cat((anchors[:, None], centers), dim=1)


def balanced_weights(logits):
    weights = logits.softmax(-1)
    for _ in range(32):
        weights = F.normalize(weights, p=1, dim=-2) * (weights.size(-2) / 5)
        weights = F.normalize(weights, p=1, dim=-1)
    return weights


def predict_with_starts(s, labels, q, choices):
    answers = {}
    if "single" in choices:
        answers["single"] = predict_embeddings(s, labels, q, "diffusion")
    if all(c == "single" for c in choices):
        return answers
    support, query = smooth(s, q)
    prior = 3 * (F.one_hot(labels, 5).to(s.dtype).transpose(1, 2) @ support)
    centers = initial_centers(query, F.normalize(prior, dim=-1))
    for _ in range(12):
        scores = query[:, None] @ centers.transpose(-1, -2)
        w = balanced_weights(scores / .08)
        centers = F.normalize(prior[:, None] + w.transpose(-1, -2) @ query[:, None], dim=-1)
    scores = query[:, None] @ centers.transpose(-1, -2)
    w = balanced_weights(scores / .08)
    # Compare the balanced, entropy-regularized clustering objective, including
    # the labeled support term. Every scalar is local to one episode and start.
    objective = (w * (scores - .08 * w.clamp_min(1e-12).log())).sum((-1, -2))
    objective = objective + (prior[:, None] * centers).sum((-1, -2))
    probs = (scores / .08).softmax(-1)
    for name in choices:
        if name == "single":
            continue
        if name == "best":
            p = probs[torch.arange(len(s), device=s.device), objective.argmax(-1)]
        elif name == "average":
            p = ((objective / .35).softmax(-1)[:, :, None, None] * probs).sum(1)
        else:
            raise ValueError(f"Unknown rule: {name}")
        answers[name] = np.stack([balanced_labels(a, x) for a, x in zip(p.cpu().numpy(), q.cpu().numpy())])
    return answers


@torch.inference_mode()
def predict_split(features, task, rules):
    si, labels, qi, _ = task
    results = {r: [] for r in rules}
    for start in range(0, len(qi), 16):
        sl = slice(start, start + 16)
        values = [torch.from_numpy(a).cuda() for a in (features[si[sl]], labels[sl], features[qi[sl]])]
        for name, pred in predict_with_starts(*values, rules).items():
            results[name].append(pred)
    return {name: np.concatenate(v) for name, v in results.items()}


class RestartClassifier:
    def __init__(self, directory, device="cpu", weights=None):
        directory = Path(directory)
        self.device = torch.device(device)
        result = json_from(directory / "results.json")
        verified_files(directory, {"validation_lock.json": result["files"]["validation_lock.json"]})
        lock = json_from(directory / "validation_lock.json")
        self.rules, self.views = lock["selected"], lock["views"]
        source = Path(weights) if weights is not None else directory.parent / "lab04_gpu_multiscale"
        verified_files(source, lock["inference_files"])
        self.extractor = ViewFeatures(source, self.device)

    @torch.inference_mode()
    def predict(self, support, labels, query):
        labels = np.asarray(labels)
        classes, ids, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if (labels.ndim != 1 or len(support) != len(labels) or len(classes) != 5
                or np.any(counts != counts[0]) or counts[0] not in (1, 5) or not len(query) or len(query) % 5):
            raise ValueError("Expected a balanced five-way task with one or five supports")
        images = np.concatenate([support, query])
        if images.dtype != np.uint8 or images.shape[1:] != (84, 84, 3):
            raise ValueError("Expected 84x84 RGB uint8 images")
        view, rule = self.views[str(counts[0])], self.rules[str(counts[0])]
        features = representation(self.extractor.collect(images, [view]), *view)
        x = torch.from_numpy(features).to(self.device)
        y = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        pred = predict_with_starts(x[None, :len(y)], y[None], x[None, len(y):], (rule,))[rule]
        return classes[pred[0]]


def study(args):
    torch.set_num_threads(2)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    parent = json_from(args.previous / "validation_lock.json")
    parent_plan = json_from(args.previous / "plan.json")
    if parent_plan["external_pretraining"]:
        raise ValueError("Expected self-trained ResNet12")
    verified_files(args.previous, {**parent["inference_files"], "plan.json": parent["plan_sha256"],
                                   "val_predictions.json": parent["val_predictions_sha256"]})
    old_result = json_from(args.rotation / "results.json")
    release = json_from(args.artifacts / "continued_results.json")
    views = parent["selected"]
    plan = {"rules": list(RULES), "views": views, "selection": "Maximum correct validation queries per shot; first rule wins ties.",
            "source_sha256": digest_file(Path(__file__)),
            "parent_lock_sha256": digest_file(args.previous / "validation_lock.json"),
            "rotation_result_sha256": digest_file(args.rotation / "results.json"),
            "episode_files": parent_plan["episodes"], "gpu": torch.cuda.get_device_name(), "external_pretraining": False}
    if (out / "plan.json").exists() and json_from(out / "plan.json") != plan:
        raise ValueError("Different study in output directory")
    save_json(out / "plan.json", plan)
    if (out / "results.json").exists():
        verified_files(out, json_from(out / "results.json")["files"])
        return
    extractor = ViewFeatures(args.previous, "cuda")
    if not (out / "validation_lock.json").exists():
        val = load_archive(args.data_root, args.artifacts, "val")
        rows = reference_rows(args.artifacts, "val", release)
        features = extractor.collect(val.images, views.values())
        parent_vals = json_from(args.previous / "val_predictions.json")
        verified_files(args.rotation, {"val_predictions.json": old_result["files"]["val_predictions.json"]})
        original = json_from(args.rotation / "val_predictions.json")
        best, selected, candidates, predictions = {}, {}, {}, {}
        for shot, view in views.items():
            task = task_indices(val, rows, int(shot))
            preds = predict_split(representation(features, *view), task, RULES)
            if not np.array_equal(preds["single"], parent_vals[shot]):
                raise ValueError("Single-start rule did not reproduce the previous validation predictions")
            candidates[shot] = {r: measures(a, task[3], np.asarray(original[shot])) for r, a in preds.items()}
            selected[shot] = max(RULES, key=lambda r: candidates[shot][r]["correct"])
            best[shot] = candidates[shot][selected[shot]]
            predictions[shot] = preds[selected[shot]].tolist()
            print("val", shot, {k: v["accuracy"] for k, v in candidates[shot].items()}, flush=True)
        save_json(out / "val_predictions.json", predictions)
        save_json(out / "validation_lock.json", {"selected": selected, "views": views, "candidates": candidates,
                  "validation": best, "inference_files": parent["inference_files"],
                  "plan_sha256": digest_file(out / "plan.json"), "val_predictions_sha256": digest_file(out / "val_predictions.json")})
        del features, val, predictions
    lock = json_from(out / "validation_lock.json")
    verified_files(out, {"plan.json": lock["plan_sha256"], "val_predictions.json": lock["val_predictions_sha256"]})
    selected = lock["selected"]
    print("Locked", selected, flush=True)
    # Identical selected classifiers reuse previously locked predictions verbatim.
    parent_result = json_from(args.previous / "results.json")
    verified_files(args.previous, {"test_predictions.json": parent_result["files"]["test_predictions.json"],
                                   "validation_lock.json": parent_result["files"]["validation_lock.json"]})
    parent_test = json_from(args.previous / "test_predictions.json")
    verified_files(args.rotation, {"test_predictions.json": old_result["files"]["test_predictions.json"]})
    original = json_from(args.rotation / "test_predictions.json")
    changed = [s for s in selected if selected[s] != "single"]
    if changed:
        test = load_archive(args.data_root, args.artifacts, "test")
        rows = reference_rows(args.artifacts, "test", release)
        features = extractor.collect(test.images, [views[s] for s in changed])
    report, predictions = {}, {}
    for s, rule in selected.items():
        if s in changed:
            task = task_indices(test, rows, int(s))
            pred = predict_split(representation(features, *views[s]), task, [rule])[rule]
            report[s] = measures(pred, task[3], np.asarray(original[s]))
            predictions[s] = pred.tolist()
        else:
            report[s], predictions[s] = parent_result["test"][s], parent_test[s]
        print("test", s, report[s], flush=True)
    save_json(out / "test_predictions.json", predictions)
    files = ("plan.json", "validation_lock.json", "val_predictions.json", "test_predictions.json")
    save_json(out / "results.json", {"track": "restarts", "external_pretraining": False, "selected": selected,
              "views": views, "val": lock["validation"], "test": report, "files": {n: digest_file(out / n) for n in files}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    p.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    p.add_argument("--previous", type=Path, default=Path("outputs/lab04_gpu_multiscale"))
    p.add_argument("--rotation", type=Path, default=Path("outputs/lab04_gpu_rotation"))
    p.add_argument("--output", type=Path, default=Path("outputs/lab04_gpu_restarts"))
    study(p.parse_args())
