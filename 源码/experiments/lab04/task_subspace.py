"""Task-local PCA and support-seeded label propagation for the self-trained ResNet12."""
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .accuracy90_data import json_from
from .evaluate_gpu_tracks import verified_files
from .multiscale_study import ViewFeatures, representation
from .prototype_restarts import predict_with_starts


# (weight of the normalized PCA projection, weight of the propagated-label embedding)
CONFIGS = {"parent": (0., 0.), "pca_half": (.5, 0.), "pca_only": (1., 0.),
           "propagation": (0., .20), "pca_propagation": (.5, .20)}
RANK = 7
NEIGHBORS = 7
ALPHA = .8


def principal_features(centered):
    # The sample Gram matrix is at most 100x100 in the evaluation protocol.
    # Project back to the original coordinate axes to remove eigenvector sign ambiguity.
    gram = centered.double() @ centered.double().transpose(1, 2)
    _, basis = torch.linalg.eigh(gram)
    basis = basis[:, :, -min(RANK, centered.size(1) - 1):].to(centered)
    return F.normalize(basis @ (basis.transpose(1, 2) @ centered), dim=-1)


def label_features(centered, labels):
    similarity = (centered @ centered.transpose(1, 2)).clamp_min(0)
    similarity.diagonal(dim1=1, dim2=2).zero_()
    value, index = similarity.topk(min(NEIGHBORS, similarity.size(-1) - 1))
    affinity = torch.zeros_like(similarity).scatter_(2, index, value.pow(3))
    affinity = .5 * (affinity + affinity.transpose(1, 2))
    inv = affinity.sum(-1).clamp_min(1e-12).rsqrt()
    graph = inv[:, :, None] * affinity * inv[:, None, :]
    eye = torch.eye(graph.size(1), dtype=graph.dtype, device=graph.device)
    seeds = centered.new_zeros(centered.size(0), centered.size(1), 5)
    seeds[:, :labels.size(1)] = F.one_hot(labels, 5).to(centered.dtype)
    raw = torch.linalg.solve(eye - ALPHA * graph, seeds).clamp_min(0)
    # Remove the five-class uniform component; a disconnected query contributes zero.
    normalized = F.normalize(raw, p=1, dim=-1)
    return F.normalize(normalized - normalized.mean(-1, keepdim=True), dim=-1)


@torch.inference_mode()
def predict_batch(support, labels, query, rule, names):
    answers = {}
    if "parent" in names:
        answers["parent"] = predict_with_starts(support, labels, query, [rule])[rule]
    alternatives = [n for n in names if n != "parent"]
    if not alternatives:
        return answers
    original = F.normalize(torch.cat((support, query), 1), dim=-1)
    centered = original - original.mean(1, keepdim=True)
    x = F.normalize(centered, dim=-1)
    pca = principal_features(centered) if any(CONFIGS[n][0] for n in alternatives) else None
    prop = label_features(x, labels) if any(CONFIGS[n][1] for n in alternatives) else None
    for name in alternatives:
        p, g = CONFIGS[name]
        parts = []
        for block, weight in ((original, (1-p)*(1-g)), (pca, p*(1-g)), (prop, g)):
            if weight:
                parts.append(math.sqrt(weight) * block)
        z = F.normalize(torch.cat(parts, dim=-1), dim=-1)
        s, q = z.split((support.size(1), query.size(1)), dim=1)
        answers[name] = predict_with_starts(s, labels, q, [rule])[rule]
    return answers


class SubspaceClassifier:
    def __init__(self, directory, device="cpu"):
        directory = Path(directory)
        result = json_from(directory / "results.json")
        verified_files(directory, {"validation_lock.json": result["files"]["validation_lock.json"]})
        lock = json_from(directory / "validation_lock.json")
        verified_files(directory, lock["inference_files"])
        self.chosen, self.views, self.rules = lock["selected"], lock["views"], lock["rules"]
        self.device = torch.device(device)
        self.extractor = ViewFeatures(directory, self.device)

    @torch.inference_mode()
    def predict(self, support, labels, query):
        labels = np.asarray(labels)
        classes, ids, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if (labels.ndim != 1 or len(support) != len(labels) or len(classes) != 5
                or np.any(counts != counts[0]) or counts[0] not in (1, 5) or not len(query) or len(query) % 5):
            raise ValueError("Expected balanced five-way support and query with one or five supports per class")
        images = np.concatenate((support, query))
        if images.dtype != np.uint8 or images.shape[1:] != (84, 84, 3):
            raise ValueError("Expected 84x84 uint8 RGB images")
        shot = str(counts[0])
        view = self.views[shot]
        values = representation(self.extractor.collect(images, [view]), *view)
        x = torch.from_numpy(values).to(self.device)
        y = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        name = self.chosen[shot]
        pred = predict_batch(x[None, :len(y)], y[None], x[None, len(y):], self.rules[shot], [name])[name]
        return classes[pred[0]]
