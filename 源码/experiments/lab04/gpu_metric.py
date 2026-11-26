"""Few-shot ResNet12 with frozen multilevel descriptors and a trainable metric."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import digest_file, normalized_tensor
from .experiment import load_checkpoint
from .graph_calibration import GraphConfig, apply_transform, predict


def describe(encoder, images, upper):
    """Pool spatial moments and channel correlations before discarding feature maps."""
    parts = []
    maps = images
    for depth, block in enumerate(encoder, 1):
        maps = block(maps)
        if depth < 2:
            continue
        spatial = maps.flatten(2)
        mean = spatial.mean(-1)
        parts.extend((F.normalize(mean), F.normalize(spatial.std(-1, unbiased=False))))
        # Keep the covariance vector compact at all three depths.
        compact = spatial.reshape(len(maps), 32, maps.shape[1] // 32, -1).mean(2)
        centered = compact - compact.mean(-1, keepdim=True)
        covariance = centered @ centered.transpose(1, 2) / centered.shape[-1]
        entries = covariance[:, upper[0], upper[1]]
        parts.append(F.normalize(entries.sign() * entries.abs().sqrt()))
    return mean, F.normalize(torch.cat(parts, dim=1))


@torch.inference_mode()
def encode(backbone, images, batch_size, device, *, clean_features=None, spatial=True):
    if batch_size < 1 or len(images) < 1:
        raise ValueError("A nonempty image array and positive batch size are required")
    if torch.device(device).type == "cuda":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    backbone.eval().to(device, memory_format=torch.channels_last)
    upper = torch.triu_indices(32, 32, device=device) if spatial else None
    raw, descriptors = [], []
    for start in range(0, len(images), batch_size):
        x = normalized_tensor(images[start:start + batch_size]).to(device, memory_format=torch.channels_last)
        if spatial:
            first, detail = describe(backbone.encoder, x, upper)
            flipped, flipped_detail = describe(backbone.encoder, x.flip(-1), upper)
            descriptors.append(F.normalize(detail + flipped_detail).cpu())
        else:
            first, flipped = backbone.features(x), backbone.features(x.flip(-1))
        if clean_features is not None:
            clean_features.append(first.cpu())
        raw.append(F.normalize(F.normalize(first) + F.normalize(flipped)).cpu())
    return torch.cat(raw), torch.cat(descriptors) if spatial else None


class SpatialMetric(nn.Module):
    def __init__(self, center, scale, dimension=192):
        super().__init__()
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.projection = nn.Linear(len(center), dimension, bias=False)

    def forward(self, descriptors):
        centered = (descriptors - self.center) / self.scale
        return F.normalize(self.projection(centered), dim=-1)


def fit_metric(descriptors, labels, *, device, steps, seed):
    """Train exclusively on episodic support/query pairs from the 64 base classes."""
    torch.manual_seed(seed)
    x = descriptors.to(device)
    by_class = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    rng = np.random.default_rng(seed)
    scale = x.std(0).clamp_min(1e-4).sqrt()
    head = SpatialMetric(x.mean(0), scale).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0005)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=0.00005)
    targets = torch.arange(16, device=device).repeat_interleave(8)
    losses = []
    for step in range(steps):
        shots = 1 if step % 2 == 0 else 5
        classes = rng.choice(len(by_class), 16, replace=False)
        samples = np.stack([rng.choice(by_class[c], shots + 8, replace=False) for c in classes])
        features = head(x[torch.as_tensor(samples, device=device)])
        centers = F.normalize(features[:, :shots].mean(1))
        logits = features[:, shots:].flatten(0, 1) @ centers.T / 0.08
        loss = F.cross_entropy(logits, targets, label_smoothing=0.05)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step in (0, steps - 1):
            losses.append(float(loss.detach()))
    return head.eval(), losses


def embed(head, descriptors, device, batch_size=512):
    with torch.inference_mode():
        return torch.cat([head(batch.to(device)).cpu()
                          for batch in descriptors.split(batch_size)]).numpy()


def fuse_features(raw, transform, spatial, weight):
    gap = apply_transform(raw, transform)
    if weight == 0:
        return gap
    return np.concatenate([np.sqrt(1 - weight) * gap, np.sqrt(weight) * spatial], axis=1)


class FewShotClassifier:
    """Load the selected checkpoint and predict one balanced five-way episode."""

    def __init__(self, directory, device="cpu"):
        directory = Path(directory)
        lock = json.loads((directory / "validation_lock.json").read_bytes())
        for name, expected in lock["inference_files"].items():
            if digest_file(directory / name) != expected:
                raise ValueError("Inference file changed: " + name)
        self.device = torch.device(device)
        self.backbone, _ = load_checkpoint(directory / "fresh.pt")
        self.configs = lock["configs"]
        self.weight = lock["spatial_weight"]
        with np.load(directory / "base_transform.npz", allow_pickle=False) as data:
            self.transform = {key: data[key] for key in data}
        self.head = None
        if self.weight:
            state = torch.load(directory / "spatial_metric.pt", map_location="cpu", weights_only=True)
            self.head = SpatialMetric(state["center"], state["scale"], len(state["projection.weight"]))
            self.head.load_state_dict(state)
            self.head.to(self.device).eval()

    def predict(self, support, support_labels, query, batch_size=64):
        """Inputs are uint8 RGB arrays at 84x84. Labels follow the support class order."""
        labels = np.asarray(support_labels)
        classes, counts = np.unique(labels, return_counts=True)
        if (len(classes) != 5 or len(set(counts)) != 1 or counts[0] not in (1, 5)
                or len(support) != len(labels) or not len(query) or len(query) % 5):
            raise ValueError("Expected five classes with one or five supports each and balanced queries")
        images = np.concatenate([support, query])
        if images.dtype != np.uint8 or images.shape[1:] != (84, 84, 3):
            raise ValueError("Expected uint8 RGB images shaped (N, 84, 84, 3)")
        raw, detail = encode(self.backbone, images, batch_size, self.device, spatial=bool(self.weight))
        spatial = None if self.head is None else embed(self.head, detail, self.device)
        x = fuse_features(raw.numpy(), self.transform, spatial, self.weight)
        result = predict(x[:len(labels)], np.searchsorted(classes, labels), x[len(labels):],
                         GraphConfig(**self.configs[str(counts[0])]))
        return classes[result["predictions"]]
