"""Compare a base-class ResNet12 and a frozen external encoder on the released episodes."""
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.nn import functional as F

from .accuracy90_data import json_from, load_archive
from .data import digest_file, normalized_tensor
from .experiment import summarize
from .gpu_metric import encode, embed, fit_metric, fuse_features
from .study_cached_refinement import paired_stats
from .supervised_whitening import fit
from .train_gpu_metric import reference_rows, save_json
from .train_medium_gpu import medium_model


DINO_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
DINO_FILES = {
    "config.json": "f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70",
    "model.safetensors": "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841",
}


@dataclass(frozen=True)
class PrototypeRule:
    name: str
    temperature: float
    updates: int
    support_weight: float


# The first rule uses only labeled supports to classify each query independently.
# The other two exploit the stated equal query counts using soft assignments.
RULES = (PrototypeRule("cosine", .10, 0, 1.),
         PrototypeRule("balanced", .08, 12, 3.),
         PrototypeRule("balanced_sharp", .035, 20, 3.))


def verified_files(directory, expected):
    for name, checksum in expected.items():
        if digest_file(directory / name) != checksum:
            raise ValueError(f"File differs from the recorded SHA-256: {directory / name}")


@torch.inference_mode()
def dino_features(model, images, batch_size, device):
    """Use all original pixels, upsampled to 224, with CLS and mean patch descriptors."""
    features = []
    for start in range(0, len(images), batch_size):
        x = normalized_tensor(images[start:start + batch_size]).to(device)
        x = F.interpolate(x, (224, 224), mode="bicubic", align_corners=False, antialias=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=torch.device(device).type == "cuda"):
            tokens = model(pixel_values=x).last_hidden_state.float()
        cls = F.normalize(tokens[:, 0])
        patches = F.normalize(tokens[:, 1:].mean(1))
        features.append(F.normalize(torch.cat([cls, patches], dim=1)).cpu())
    return torch.cat(features).numpy()


def load_dino(directory, device):
    from transformers import Dinov2Model
    verified_files(directory, DINO_FILES)
    return Dinov2Model.from_pretrained(directory, local_files_only=True).eval().to(device)


def predict_tasks(support, labels, query, rule):
    """Batch of five-way tasks. Inputs are image embeddings and local support labels only."""
    support, query = F.normalize(support, dim=-1), F.normalize(query, dim=-1)
    assigned = F.one_hot(labels, num_classes=5).to(support.dtype)
    support_sum = assigned.transpose(1, 2) @ support
    prototypes = F.normalize(support_sum, dim=-1)
    if not rule.updates:
        return (query @ prototypes.transpose(1, 2)).argmax(-1)
    # Center each transductive task independently before estimating its prototypes.
    center = torch.cat([support, query], dim=1).mean(1, keepdim=True)
    support = F.normalize(support - center, dim=-1)
    query = F.normalize(query - center, dim=-1)
    prior = rule.support_weight * (assigned.transpose(1, 2) @ support)
    prototypes = F.normalize(prior, dim=-1)
    for step in range(rule.updates + 1):
        logits = (query @ prototypes.transpose(1, 2)) / rule.temperature
        # Sinkhorn normalization enforces equal expected query mass per class.
        weights = (logits - logits.amax(-1, keepdim=True)).exp()
        mass = query.shape[1] / 5
        for _ in range(40):
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-30) * mass
            weights = weights / weights.sum(2, keepdim=True).clamp_min(1e-30)
        if step < rule.updates:
            prototypes = F.normalize(prior + weights.transpose(1, 2) @ query, dim=-1)
    return weights.argmax(-1)


def task_indices(split, rows, shots):
    lookup = {name: i for i, name in enumerate(split.ids)}
    rows = sorted((row for row in rows if row["shots"] == shots), key=lambda row: row["episode"])
    if [row["episode"] for row in rows] != list(range(600)):
        raise ValueError("Expected the 600 distinct released episodes per shot")
    supports, support_labels, queries, targets = [], [], [], []
    for row in rows:
        si = [lookup[name] for name in row["support_ids"]]
        qi = [lookup[name] for name in row["query_ids"]]
        classes = list(dict.fromkeys(split.labels[si].tolist()))
        sl = [classes.index(c) for c in split.labels[si]]
        target = [classes.index(c) for c in split.labels[qi]]
        if (len(set(si + qi)) != 5 * (shots + 15) or len(classes) != 5
                or np.bincount(sl).tolist() != [shots] * 5
                or np.bincount(target).tolist() != [15] * 5 or target != row["query_labels"]):
            raise ValueError("Support/query data differ from the released task")
        supports.append(si)
        support_labels.append(sl)
        queries.append(qi)
        targets.append(target)
    return tuple(np.asarray(x, dtype=np.int64) for x in (supports, support_labels, queries, targets))


@torch.inference_mode()
def score(features, tasks, rule, device):
    si, labels, qi, targets = tasks
    predictions = []
    for start in range(0, len(si), 24):
        batch = slice(start, start + 24)
        inputs = [torch.from_numpy(x).to(device) for x in
                  (features[si[batch]], labels[batch], features[qi[batch]])]
        predictions.append(predict_tasks(*inputs, rule).cpu().numpy())
    predictions = np.concatenate(predictions)
    correct = (predictions == targets).sum(1)
    return {**summarize(correct / 75), "correct": int(correct.sum()), "queries": int(targets.size)}, predictions


class ResNetFeatures:
    def __init__(self, directory, device, batch):
        self.model = medium_model()
        self.model.load_state_dict(torch.load(directory / "medium.pt", map_location="cpu", weights_only=True))
        self.device, self.batch = device, batch
        self.head, self.transform = None, None

    def fit_base(self, base, out):
        raw, details = encode(self.model, base.images, self.batch, self.device)
        self.transform = fit(raw.numpy(), base.labels)
        np.savez_compressed(out / "base_transform.npz", **self.transform)
        self.head, _ = fit_metric(details, base.labels, device=self.device, steps=1500, seed=20261127)
        torch.save(self.head.cpu().state_dict(), out / "spatial_metric.pt")
        self.head.to(self.device)

    def __call__(self, images):
        raw, detail = encode(self.model, images, self.batch, self.device)
        return fuse_features(raw.numpy(), self.transform, embed(self.head, detail, self.device), .5).astype(np.float32)


class EpisodeClassifier:
    """Restore the validation-selected rule and predict from RGB support and query images."""

    def __init__(self, directory, device="cpu", weights=None, batch_size=32):
        from .gpu_metric import SpatialMetric
        directory = Path(directory)
        plan = json_from(directory / "plan.json")
        lock = json_from(directory / "validation_lock.json")
        verified_files(directory, {**lock["inference_files"], "plan.json": lock["plan_sha256"]})
        self.device = torch.device(device)
        self.rules = {int(shot): next(PrototypeRule(**r) for r in plan["rules"] if r["name"] == name)
                      for shot, name in lock["selected"].items()}
        if plan["external_pretraining"]:
            weights = Path(weights) if weights else Path(__file__).resolve().parents[2] / ".local/dinov2-base"
            dino = load_dino(weights, self.device)
            self.extract = lambda images: dino_features(dino, images, batch_size, self.device)
        else:
            self.extract = ResNetFeatures(directory, self.device, batch_size)
            state = torch.load(directory / "spatial_metric.pt", map_location="cpu", weights_only=True)
            head = SpatialMetric(state["center"], state["scale"], state["projection.weight"].shape[0])
            head.load_state_dict(state)
            self.extract.head = head.to(self.device).eval()
            with np.load(directory / "base_transform.npz") as data:
                self.extract.transform = dict(data)

    @torch.inference_mode()
    def predict(self, support_images, support_labels, query_images):
        labels = np.asarray(support_labels)
        if labels.ndim != 1 or len(labels) != len(support_images):
            raise ValueError("Expected one label per support image")
        classes, ids, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if len(classes) != 5 or np.any(counts != counts[0]) or counts[0] not in self.rules:
            raise ValueError("Expected five classes with one or five supports each")
        images = np.concatenate([support_images, query_images])
        if not len(query_images) or images.dtype != np.uint8 or images.shape[1:] != (84, 84, 3):
            raise ValueError("Expected nonempty uint8 RGB images at 84x84")
        rule = self.rules[counts[0]]
        if rule.updates and len(query_images) % len(classes):
            raise ValueError("The balanced rule requires equally sized query classes")
        x = torch.from_numpy(self.extract(images)).to(self.device)
        y = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        prediction = predict_tasks(x[None, :len(ids)], y[None], x[None, len(ids):], rule)
        return classes[prediction[0].cpu().numpy()]


def run(args):
    device = torch.device("cuda")
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or args.batch_size <= 0:
        raise ValueError("Select a working CUDA device and positive batch size")
    out = args.output or Path("outputs") / ("lab04_gpu_" + args.track)
    out.mkdir(parents=True, exist_ok=True)
    release = json_from(args.artifacts / "continued_results.json")
    fresh = json_from(args.previous / "results.json")
    plan = {"track": args.track, "rules": [asdict(rule) for rule in RULES],
            "selection": "Highest validation accuracy separately for each shot; ties use first rule.",
            "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "seed": 20261127, "batch_size": args.batch_size,
            "episode_files": {name: release["files"][name]["sha256"] for name in
                              ("continued_validation_tasks.jsonl", "continued_episodes.json")},
            "dataset_manifest_sha256": digest_file(args.artifacts / "dataset_manifest.json"),
            "implementation_sha256": digest_file(Path(__file__))}
    if args.track == "dino":
        plan.update(model_id="facebook/dinov2-base", revision=DINO_REVISION, files=DINO_FILES,
                    external_pretraining=True, representation="224px, normalized CLS + mean patch, no fine-tuning")
    else:
        plan.update(external_pretraining=False, backbone_sha256=digest_file(args.trained / "medium.pt"),
                    training_plan=json_from(args.trained / "training_plan.json"),
                    representation="84px, two views, 50% supervised whitening + 50% spatial metric")
    if (out / "plan.json").exists() and json_from(out / "plan.json") != plan:
        raise ValueError("Output belongs to a different run")
    if (out / "results.json").is_file():
        result = json_from(out / "results.json")
        verified_files(out, result["files"])
        print(out / "results.json", flush=True)
        return
    save_json(out / "plan.json", plan)
    inference_files = {}
    if args.track == "dino":
        model = load_dino(args.weights, device)

        def extract(images):
            return dino_features(model, images, args.batch_size, device)
    else:
        if len(json_from(args.trained / "training.json")) != plan["training_plan"]["epochs"]:
            raise ValueError("Backbone training has not completed")
        for name in ("training.json", "medium.pt"):
            shutil.copyfile(args.trained / name, out / name)
        extract = ResNetFeatures(out, device, args.batch_size)
        extract.fit_base(load_archive(args.data_root, args.artifacts, "train"), out)
        inference_files = {n: digest_file(out / n) for n in ("medium.pt", "spatial_metric.pt", "base_transform.npz")}
    result = {"track": args.track, "external_pretraining": plan["external_pretraining"]}
    for split_name in ("val", "test"):
        print("Encoding", split_name, "images on", plan["gpu"], flush=True)
        split = load_archive(args.data_root, args.artifacts, split_name)
        features = extract(split.images)
        rows = reference_rows(args.artifacts, split_name, release)
        selected, candidates, scores, evidence = {}, {}, {}, {}
        if split_name == "test":
            selected = json_from(out / "validation_lock.json")["selected"]
        previous_file = split_name + "_predictions.json"
        verified_files(args.previous, {previous_file: fresh["files"][previous_file]})
        previous = {(r["shots"], r["episode"]): r["correct"] for r in json_from(args.previous / previous_file)}
        for shot in (1, 5):
            key = str(shot)
            tasks = task_indices(split, rows, shot)
            rules = RULES if split_name == "val" else [next(r for r in RULES if r.name == selected[key])]
            trials = {r.name: score(features, tasks, r, device) for r in rules}
            candidates[key] = {name: pair[0] for name, pair in trials.items()}
            best = max(trials, key=lambda name: trials[name][0]["accuracy"])
            stats, answers = trials[best]
            selected[key] = best
            by_episode = (answers == tasks[3]).mean(1)
            baseline = np.array([previous[shot, i] for i in range(600)]) / 75
            scores[key] = {**stats, "paired_vs_fresh": paired_stats(by_episode, baseline)}
            evidence[key] = answers.tolist()
            print(split_name, shot, best, scores[key], flush=True)
        if split_name == "val":
            save_json(out / "validation_lock.json", {"selected": selected, "candidates": candidates,
                      "plan_sha256": digest_file(out / "plan.json"), "inference_files": inference_files})
        save_json(out / previous_file, evidence)
        result[split_name] = scores
        del split, features, rows
    files = ["plan.json", "validation_lock.json", "val_predictions.json", "test_predictions.json", *inference_files]
    if args.track == "medium":
        files.append("training.json")
    result["files"] = {name: digest_file(out / name) for name in files}
    save_json(out / "results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", choices=("medium", "dino"))
    parser.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/lab04_gpu_fresh"))
    parser.add_argument("--trained", type=Path, default=Path(".local/lab04-medium"))
    parser.add_argument("--weights", type=Path, default=Path(".local/dinov2-base"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())
