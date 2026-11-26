"""Train a local covariance descriptor on the base split of the frozen self-trained ResNet12."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .accuracy90_data import json_from, load_archive
from .data import digest_file, normalized_tensor
from .evaluate_gpu_tracks import verified_files
from .gpu_metric import embed, fit_metric, SpatialMetric
from .train_gpu_backbone import atomic_save
from .train_gpu_metric import save_json
from .train_medium_gpu import medium_model


def describe_local(model, x, projections, upper):
    parts = []
    for depth, block in enumerate(model.encoder):
        x = block(x)
        if depth < 2:
            continue
        pixels = x.flatten(2)
        # Retain the magnitude of strong activations and coarse vertical position.
        halves = F.adaptive_avg_pool2d(x, (2, 1)).flatten(1)
        power = pixels.clamp_min(1e-6).pow(3).mean(-1).pow(1 / 3)
        parts.extend([F.normalize(halves), F.normalize(power)])
        reduced = projections[depth - 2] @ pixels
        reduced = reduced - reduced.mean(-1, keepdim=True)
        cov = reduced @ reduced.transpose(1, 2) / pixels.shape[-1]
        values = cov[:, upper[0], upper[1]]
        parts.append(F.normalize(values.sign() * values.abs().sqrt()))
    return F.normalize(torch.cat(parts, dim=1))


class LocalFeatures:
    def __init__(self, model, device, batch=32):
        self.device = torch.device(device)
        self.model = model.eval().to(self.device, memory_format=torch.channels_last)
        self.batch = batch
        rng = torch.Generator().manual_seed(20261129)
        self.projections = [
            torch.linalg.qr(torch.randn(channels, 48, generator=rng))[0].T.contiguous().to(self.device)
            for channels in (128, 256)]
        self.upper = torch.triu_indices(48, 48, device=self.device)
        self.head = None

    @torch.inference_mode()
    def descriptors(self, images):
        if self.device.type == "cuda":
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        pieces = []
        for start in range(0, len(images), self.batch):
            x = normalized_tensor(images[start:start + self.batch]).to(self.device, memory_format=torch.channels_last)
            full = describe_local(self.model, x, self.projections, self.upper)
            flip = describe_local(self.model, x.flip(-1), self.projections, self.upper)
            pieces.append(F.normalize(full + flip).cpu())
        return torch.cat(pieces)

    def load_head(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        head = SpatialMetric(state["center"], state["scale"], len(state["projection.weight"]))
        head.load_state_dict(state)
        self.head = head.eval().to(self.device)

    def __call__(self, images):
        return embed(self.head, self.descriptors(images), self.device)


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Use a CUDA-enabled PyTorch environment to fit the local descriptor")
    torch.set_num_threads(2)
    parent = json_from(args.previous / "results.json")
    if parent["external_pretraining"] or parent["selected"]["backbone"] != "rotation":
        raise ValueError("Expected the base-only rotation continuation")
    verified_files(args.previous, {"medium.pt": parent["files"]["medium.pt"]})
    path = args.output
    path.mkdir(parents=True, exist_ok=True)
    seed, steps = 20261129, 2000
    plan = {"model_sha256": parent["files"]["medium.pt"], "external_pretraining": False,
            "train_ids": json_from(args.artifacts / "protocol.json")["train_selection_sha256"],
            "steps": steps, "seed": seed, "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "source_sha256": {p.name: digest_file(p) for p in
                              (Path(__file__), Path(__file__).with_name("gpu_metric.py"))}}
    if (path / "training_plan.json").exists() and json_from(path / "training_plan.json") != plan:
        raise ValueError("Training directory already holds a different plan")
    save_json(path / "training_plan.json", plan)
    if (path / "training.json").exists():
        recorded = json_from(path / "training.json")
        verified_files(path, recorded["files"])
        print("Verified trained local metric", flush=True)
        return
    base = load_archive(args.data_root, args.artifacts, "train")
    model = medium_model()
    model.load_state_dict(torch.load(args.previous / "medium.pt", map_location="cpu", weights_only=True))
    extractor = LocalFeatures(model, "cuda")
    print("Encoding", len(base.labels), "base images on", plan["gpu"], flush=True)
    descr = extractor.descriptors(base.images)
    print("Fitting local descriptor projection", list(descr.shape), flush=True)
    head, loss = fit_metric(descr, base.labels, device="cuda", steps=steps, seed=seed)
    if not all(np.isfinite(loss)):
        raise FloatingPointError("Non-finite metric training loss")
    atomic_save(path / "local_metric.pt", head.cpu().state_dict())
    save_json(path / "training.json", {"images": len(base.labels), "classes": len(set(base.labels)),
              "descriptor_dimension": descr.size(1), "loss_first_last": loss, "steps": steps,
              "files": {n: digest_file(path / n) for n in ("local_metric.pt", "training_plan.json")}})
    print("Saved", path / "local_metric.pt", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/lab04"))
    parser.add_argument("--data-root", type=Path, default=Path(".local/mini-imagenet"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/lab04_gpu_rotation"))
    parser.add_argument("--output", type=Path, default=Path(".local/lab04-local-metric"))
    train(parser.parse_args())
