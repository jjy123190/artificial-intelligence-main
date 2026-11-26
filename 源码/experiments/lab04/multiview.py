"""Optional multi-view features and capacity-constrained transductive decisions.

Views are deterministic original/left-right-flip embeddings. A capacity prior
uses a DECLARED equal-query-count protocol, never hidden query labels. Soft
scores are returned unmodified and are not falsely called calibrated certainty.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import torch

from .calibration import unit_features
from .data import normalized_tensor
from .graph_calibration import GraphConfig, predict as graph_predict


@dataclass(frozen=True)
class MultiViewConfig:
    views: str = "mean"
    assignment: str = "soft"

    def __post_init__(self):
        if self.views not in {"original", "mean", "concat"}:
            raise ValueError("Unsupported view combination")
        if self.assignment not in {"soft", "balanced_capacity"}:
            raise ValueError("Unsupported decision rule")

    def to_dict(self):
        return asdict(self)


def combine_views(original, flipped, mode):
    first = unit_features(original)
    if mode == "original":
        return first
    second = unit_features(flipped)
    if first.shape != second.shape:
        raise ValueError("View feature matrices must have identical shapes")
    if mode == "mean":
        return unit_features((first + second) * .5)
    if mode == "concat":
        return unit_features(np.concatenate((first, second), axis=1))
    raise ValueError("Unknown view aggregation")


def encode_views(model, images, *, mode="mean", batch_size=64):
    if mode not in {"original", "mean", "concat"}:
        raise ValueError("Unsupported view combination")
    if model.training:
        raise ValueError("Inference must use a frozen evaluation-mode model")
    output = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            tensor = normalized_tensor(images[start:start + batch_size])
            first = model.features(tensor).cpu().numpy()
            second = model.features(tensor.flip(-1)).cpu().numpy() if mode != "original" else first
            output.append(combine_views(first, second, mode))
    if not output:
        raise ValueError("No images to encode")
    return np.concatenate(output)


def minimum_assignment(cost):
    """Square linear assignment via primal-dual augmenting paths (NumPy only).

    No downloaded solver, label inputs, or platform-specific dependencies.
    Return a column per row. O(n^3), bounded to one <=75-query classroom task.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError("Expected a nonempty square assignment matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Assignment costs must be finite")
    n = len(matrix)
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p, way = np.zeros(n + 1, dtype=np.int64), np.zeros(n + 1, dtype=np.int64)
    for row in range(1, n + 1):
        p[0] = row
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            available = np.flatnonzero(~used[1:]) + 1
            cur = matrix[i0 - 1, available - 1] - u[i0] - v[available]
            improved = cur < minv[available]
            minv[available[improved]] = cur[improved]
            way[available[improved]] = j0
            j1 = int(available[np.argmin(minv[available])])
            delta = minv[j1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = int(j1)
            if j0 == 0:
                break
    columns = np.empty(n, dtype=np.int64)
    columns[p[1:] - 1] = np.arange(n)
    return columns


def capacity_predictions(probabilities, query_features):
    scores = np.asarray(probabilities, dtype=np.float64)
    features = np.asarray(query_features, dtype=np.float64)
    if (scores.ndim != 2 or len(scores) != len(features) or not len(scores)
            or not np.isfinite(scores).all() or (scores < 0).any()
            or scores.shape[1] < 2 or len(scores) % scores.shape[1]):
        raise ValueError("Capacity assignment requires finite scores and balanced whole-class query counts")
    ways = scores.shape[1]
    slots = np.repeat(np.arange(ways), len(scores) // ways)
    # Order by features, not labels, so arbitrary API query ordering cannot
    # break a tie differently. Identical-feature samples are interchangeable.
    order = np.lexsort(features.T[::-1])
    cost = -np.log(np.maximum(scores[order][:, slots], 1e-15))
    labels = slots[minimum_assignment(cost)]
    result = np.empty(len(labels), dtype=np.int64)
    result[order] = labels
    return result


def predict(support, support_labels, query, graph_config, assignment="soft"):
    if assignment not in {"soft", "balanced_capacity"}:
        raise ValueError("Unknown assignment rule")
    result = graph_predict(support, support_labels, query, graph_config)
    if assignment == "balanced_capacity":
        if not graph_config.balanced:
            raise ValueError("Explicit balanced protocol is required for capacity decisions")
        result["predictions"] = capacity_predictions(result["probabilities"], query)
    result["decision_rule"] = assignment
    result["assigned_scores"] = result["probabilities"][np.arange(len(query)), result["predictions"]]
    return result
