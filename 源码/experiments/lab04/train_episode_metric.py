"""Train a small metric on base-class tasks using frozen ResNet12 features."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .accuracy90_data import json_from, load_archive
from .data import digest_file
from .evaluate_gpu_tracks import verified_files
from .learned_calibration import LearnedCalibration, LearnedCalibrationConfig
from .rotation_study import center_features, load_extractor
from .train_gpu_metric import save_json


BRANCHES = ("support", "refined")
STEPS = 1200
BATCH_TASKS = 8
SEED = 40719
HIDDEN = 64


def prepare_features(args):
    cache = args.cache
    cache.mkdir(parents=True, exist_ok=True)
    parent = json_from(args.previous / "results.json")
    verified_files(args.previous, {"validation_lock.json": parent["files"]["validation_lock.json"]})
    lock = json_from(args.previous / "validation_lock.json")
    verified_files(args.previous, lock["inference_files"])
    if parent["external_pretraining"]:
        raise ValueError("Expected a self-trained ResNet12")
    plan = {"parent_files": lock["inference_files"],
            "dataset_manifest": digest_file(args.artifacts / "dataset_manifest.json"),
            "train_ids": json_from(args.artifacts / "protocol.json")["train_selection_sha256"],
            "views": ["original with flip", "70px center crop with flip"]}
    if (cache / "features.json").is_file():
        record = json_from(cache / "features.json")
        if record["plan"] != plan:
            raise ValueError("Feature cache belongs to a different backbone or dataset")
        verified_files(cache, record["files"])
        return
    base = load_archive(args.data_root, args.artifacts, "train")
    if base.images.shape[0] != 38400 or len(base.class_names) != 64:
        raise ValueError("Expected all 64 base classes")
    enc = load_extractor(args.previous, "cuda")
    print("Encoding 38,400 base images on", torch.cuda.get_device_name(), flush=True)
    for name, transform in (("orig.npy", enc), ("crop.npy", lambda x: center_features(enc, x))):
        pieces = []
        for start in range(0, len(base.images), 2048):
            pieces.append(transform(base.images[start:start + 2048]))
            if start % 8192 == 0:
                print(name, start + len(pieces[-1]), "/", len(base.images), flush=True)
        np.save(cache / name, np.concatenate(pieces))
    np.save(cache / "labels.npy", base.labels)
    save_json(cache / "features.json", {"plan": plan, "files": {
        n: digest_file(cache / n) for n in ("orig.npy", "crop.npy", "labels.npy")}})


def model_for(dimension):
    cfg = LearnedCalibrationConfig(dimension=dimension, hidden=HIDDEN, residual_scale=.5)
    return LearnedCalibration(cfg)


def sample_indices(groups, rng, shots):
    classes = rng.random((BATCH_TASKS, len(groups))).argsort(1)[:, :5]
    # The first shots images form support, the next 15 distinct images form query.
    offsets = rng.random((BATCH_TASKS, 5, groups.shape[1])).argsort(2)[:, :, :shots + 15]
    return groups[classes[:, :, None], offsets]


def prototype_logits(support, query, refine):
    """Query features can update centers. Targets never enter an update."""
    anchor = support.sum(2)
    centers = F.normalize(anchor, dim=-1)
    if refine:
        for _ in range(3):
            assignment = (query @ centers.transpose(1, 2) / .10).softmax(-1)
            for _ in range(4):
                assignment = F.normalize(assignment, p=1, dim=1) * (query.shape[1] / 5)
                assignment = F.normalize(assignment, p=1, dim=2)
            centers = F.normalize(3 * anchor + assignment.transpose(1, 2) @ query, dim=-1)
    return query @ centers.transpose(1, 2) / .10


def train_branch(name, orig, crop, groups):
    if name not in BRANCHES:
        raise ValueError(name)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = model_for(orig.shape[1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=.00005)
    targets = torch.arange(5, device=orig.device).repeat_interleave(15).repeat(BATCH_TASKS)
    history = []
    totals = torch.zeros(3, device=orig.device)
    for step in range(STEPS):
        shots = (1, 5)[step % 2]
        ix = torch.as_tensor(sample_indices(groups, rng, shots), device=orig.device)
        a, b = orig[ix].flatten(0, 2), crop[ix].flatten(0, 2)
        x = F.normalize(a + b)
        z = model(x)
        by_class = z.reshape(BATCH_TASKS, 5, shots + 15, -1)
        support = by_class[:, :, :shots]
        query = by_class[:, :, shots:].flatten(1, 2)
        logits = prototype_logits(support, query, name == "refined")
        ce = F.cross_entropy(logits.flatten(0, 1), targets, label_smoothing=.04)
        a, b = model(a), model(b)
        align = ((a.mean(0) - b.mean(0)).square().sum()
                 + (a.std(0, unbiased=False) - b.std(0, unbiased=False)).square().sum())
        drift = (z - x).square().sum(-1).mean()
        loss = ce + .25 * align + .20 * drift
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        totals += torch.stack((ce, align, drift)).detach()
        if (step + 1) % 200 == 0:
            values = (totals / 200).tolist()
            if not np.isfinite(values).all():
                raise FloatingPointError("Non-finite training losses")
            history.append(dict(zip(("step", "cross_entropy", "alignment", "drift"), [step + 1, *values])))
            totals.zero_()
            print(name, history[-1], flush=True)
    return model.cpu().eval(), history


def data_args(p):
    p.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    p.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    p.add_argument("--previous", type=Path, default=Path("outputs/lab04_gpu_rotation_views"))
    p.add_argument("--cache", type=Path, default=Path(".local/lab04-episode-metric"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cache the original and center-crop base-class embeddings")
    data_args(p)
    torch.set_num_threads(2)
    prepare_features(p.parse_args())
