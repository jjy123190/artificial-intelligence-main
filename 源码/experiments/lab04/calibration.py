"""Mean/variance shrinkage and optional unlabelled-query prototype updates.

There is intentionally no query-label argument in any prediction function.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class CalibrationConfig:
    mean_weight: float = 0.15
    variance_prior_strength: float = 4.0
    neighbors: int = 3
    prototype_weight: float = 0.25
    prototype_steps: int = 2
    confidence_threshold: float = 0.25
    temperature: float = 0.5

    def to_dict(self):
        return asdict(self)


def unit_features(features):
    features = np.asarray(features, dtype=np.float64)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)


def base_statistics(features, labels):
    features = unit_features(features)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    means = np.stack([features[labels == label].mean(0) for label in classes])
    variances = np.stack([features[labels == label].var(0) for label in classes])
    return {"classes": classes, "means": means, "variances": variances}


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    exponentials = np.exp(values - np.max(values, axis=1, keepdims=True))
    return exponentials / np.maximum(exponentials.sum(1, keepdims=True), 1e-12)


def calibrated_predict(support, support_labels, query, prior,
                       *, mean=False, variance=False, prototype=False,
                       config: CalibrationConfig | None = None):
    """Predict an episode; query labels cannot influence any computation.

    Query-assisted prototype refinement is explicitly transductive. With
    prototype=False this function is inductive: predictions for one query
    are independent of all other query examples.
    """
    config = config or CalibrationConfig()
    support, query = unit_features(support), unit_features(query)
    labels = np.asarray(support_labels, dtype=np.int64)
    classes = np.unique(labels)
    if len(support) != len(labels) or support.shape[1] != query.shape[1]:
        raise ValueError("Inconsistent support/query feature shapes")
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Support labels must be consecutive episode labels")
    prototypes = np.stack([support[labels == label].mean(0) for label in classes])
    raw_prototypes = prototypes.copy()
    shot_counts = np.array([(labels == label).sum() for label in classes])
    prior_means = np.asarray(prior["means"], dtype=np.float64)
    prior_variances = np.asarray(prior["variances"], dtype=np.float64)
    neighbor_ids = np.argsort(((prototypes[:, None] - prior_means[None]) ** 2).sum(2),
                             axis=1)[:, :min(config.neighbors, len(prior_means))]
    neighborhood_means = prior_means[neighbor_ids].mean(1)
    neighborhood_variances = (prior_variances[neighbor_ids] +
                              (prior_means[neighbor_ids] -
                               neighborhood_means[:, None]) ** 2).mean(1)
    # Floor is tied to training statistics and independent of the query set.
    variance_floor = np.maximum(prior_variances.mean(0) * 0.1, 1e-5)
    if mean:
        prototypes = ((1 - config.mean_weight) * prototypes +
                      config.mean_weight * neighborhood_means)
    # A shared training-derived scale leaves Euclidean baseline decisions
    # unchanged while keeping softmax temperatures comparable across modes.
    variances = np.full_like(prototypes, max(float(prior_variances.mean()), 1e-5))
    if variance:
        for label in classes:
            selected = support[labels == label]
            observed_dof = max(len(selected) - 1, 0)
            empirical = selected.var(0, ddof=1) if observed_dof else np.zeros(support.shape[1])
            variances[label] = (
                observed_dof * empirical + config.variance_prior_strength *
                neighborhood_variances[label]
            ) / (observed_dof + config.variance_prior_strength)
        variances = np.maximum(variances, variance_floor)

    def score(current):
        distance = ((query[:, None] - current[None]) ** 2 / variances[None]).mean(2)
        # A common temperature changes confidence but not nearest-prototype
        # predictions. The variance version is diagonal Mahalanobis distance,
        # not a generative density claim.
        return -distance / max(config.temperature, 1e-6)

    initial_prototypes = prototypes.copy()
    if prototype:
        for _ in range(config.prototype_steps):
            weights = softmax(score(prototypes))
            confidence = weights.max(1)
            accepted = confidence >= config.confidence_threshold
            weights = weights * accepted[:, None]
            masses = weights.sum(0)
            prototypes = (
                shot_counts[:, None] * initial_prototypes +
                config.prototype_weight * weights.T @ query
            ) / (shot_counts[:, None] + config.prototype_weight * masses[:, None])
    logits = score(prototypes)
    return {
        "predictions": logits.argmax(1).astype(int),
        "probabilities": softmax(logits),
        "prototypes_before": raw_prototypes,
        "prototypes_after": prototypes,
        "variances": variances,
        "neighbor_ids": neighbor_ids,
        "transductive": bool(prototype),
    }


METHODS = {
    "baseline": {"mean": False, "variance": False, "prototype": False},
    "alignment_only": {"mean": False, "variance": False, "prototype": False},
    "mean_only": {"mean": True, "variance": False, "prototype": False},
    "variance_only": {"mean": False, "variance": True, "prototype": False},
    "prototype_only": {"mean": False, "variance": False, "prototype": True},
    "mean_variance": {"mean": True, "variance": True, "prototype": False},
    "full": {"mean": True, "variance": True, "prototype": True},
    "no_alignment_full": {"mean": True, "variance": True, "prototype": True},
}
