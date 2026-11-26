"""Mix the global image embedding with a kernel comparing sets of ResNet12 regions."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .accuracy90_data import json_from
from .data import normalized_tensor
from .multiscale_study import representation
from .prototype_restarts import RestartClassifier, predict_with_starts


@torch.inference_mode()
def collect_regions(model, images, device, grid):
    # Match corresponding locations across an image and its horizontal flip.
    encoder = model.encoder[:3].eval().to(device, memory_format=torch.channels_last)
    rows = []
    for start in range(0, len(images), 32):
        x = normalized_tensor(images[start:start + 32]).to(device, memory_format=torch.channels_last)
        a = F.adaptive_avg_pool2d(encoder(x), grid)
        b = F.adaptive_avg_pool2d(encoder(x.flip(-1)), grid).flip(-1)
        rows.append(((a + b) / 2).flatten(2).transpose(1, 2).contiguous().cpu())
    return torch.cat(rows).numpy()


def kernel_features(global_features, regions, config):
    # Each batch member is a separate episode. No category labels enter the kernel.
    b, n, p, d = regions.shape
    patches = F.normalize(regions - regions.mean(dim=(1, 2), keepdim=True), dim=-1).reshape(b, n*p, d)
    similarities = patches @ patches.transpose(-1, -2)
    pairwise = ((similarities.clamp(max=1) - 1) / config["temperature"]).exp()
    kernel = pairwise.reshape(b, n, p, n, p).mean(dim=(2, 4))
    inv_norm = kernel.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).rsqrt()
    kernel = inv_norm.unsqueeze(-1) * kernel * inv_norm.unsqueeze(-2)
    # A Gram factor permits the existing cosine/prototype code to use the set kernel.
    values, vectors = torch.linalg.eigh((kernel + kernel.transpose(-1, -2)) * .5)
    local = (vectors * values.clamp_min(0).sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
    w = config["weight"]
    return torch.cat(((1 - w)**.5 * F.normalize(global_features, dim=-1), w**.5 * local), dim=-1)


@torch.inference_mode()
def predict_regions(embeddings, patches, task, config, rule, device):
    support, labels, query = task[:3]
    predictions = []
    for start in range(0, len(support), 8):
        end = start + 8
        indices = np.concatenate((support[start:end], query[start:end]), axis=1)
        x = torch.from_numpy(embeddings[indices]).to(device)
        r = torch.from_numpy(patches[indices]).to(device)
        joint = kernel_features(x, r, config)
        y = torch.from_numpy(labels[start:end]).long().to(device)
        pred = predict_with_starts(joint[:, :y.size(1)], y, joint[:, y.size(1):], [rule])[rule]
        predictions.append(pred)
    return np.concatenate(predictions)


class RegionalClassifier(RestartClassifier):
    def __init__(self, directory, device="cpu"):
        directory = Path(directory)
        super().__init__(directory, device, weights=directory)
        self.kernel = json_from(directory / "validation_lock.json")["kernel"]

    @torch.inference_mode()
    def predict(self, support, labels, query):
        if self.kernel is None:
            return super().predict(support, labels, query)
        labels = np.asarray(labels)
        classes, local_labels, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if (labels.ndim != 1 or len(support) != len(labels) or len(classes) != 5 or
                np.any(counts != counts[0]) or counts[0] not in (1, 5) or not len(query) or len(query) % 5):
            raise ValueError("Expected a balanced five-way episode with one or five supports")
        images = np.concatenate((support, query))
        if images.dtype != np.uint8 or images.shape[1:] != (84, 84, 3):
            raise ValueError("Expected 84x84 RGB uint8 images")
        shot = str(counts[0])
        view = self.views[shot]
        rows = self.extractor.collect(images, [view])
        embedding = representation(rows, *view)
        regions = collect_regions(self.extractor.original.model, images, self.device, self.kernel["grid"])
        indices = np.arange(len(images))[None]
        task = (indices[:, :len(labels)], local_labels[None], indices[:, len(labels):])
        return classes[predict_regions(embedding, regions, task, self.kernel, self.rules[shot], self.device)[0]]
