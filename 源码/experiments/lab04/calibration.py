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
    # Defaults preserve the published v1 predictions exactly.
    variance_blend: float = 1.0
    balanced_queries: bool = False
    normalize_prototypes: bool = False

    def __post_init__(self):
        for name in ("mean_weight", "variance_blend", "confidence_threshold"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if (not np.isfinite(self.variance_prior_strength) or self.variance_prior_strength <= 0
                or not np.isfinite(self.prototype_weight) or self.prototype_weight < 0
                or not np.isfinite(self.temperature) or self.temperature <= 0):
            raise ValueError("Invalid calibration scale")
        if (isinstance(self.neighbors, bool) or not isinstance(self.neighbors, int) or self.neighbors < 1
                or isinstance(self.prototype_steps, bool) or not isinstance(self.prototype_steps, int)
                or not 0 <= self.prototype_steps <= 30):
            raise ValueError("Invalid iteration or neighborhood count")

    def to_dict(self):
        return asdict(self)


def unit_features(features):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] == 0 or not np.isfinite(features).all():
        raise ValueError("Features must be a finite matrix with a nonzero embedding dimension")
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


def balanced_soft_assignments(logits, iterations=50):
    """Entropy-regularized uniform class-mass assignment, with no labels.

    This optional transductive prior is appropriate only for the explicitly
    balanced 5-way benchmark. It is NOT a general class-imbalance solution.
    Log-space Sinkhorn iterations avoid underflow on confident logits.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not logits.shape[0] or not logits.shape[1] or not np.isfinite(logits).all():
        raise ValueError("Expected nonempty finite logits")
    log_weights = logits - logits.max()
    def logsumexp(x, axis):
        maximum = x.max(axis=axis, keepdims=True)
        return maximum + np.log(np.exp(x - maximum).sum(axis=axis, keepdims=True))
    target_mass = np.log(logits.shape[0] / logits.shape[1])
    for _ in range(iterations):
        log_weights -= logsumexp(log_weights, 1)
        log_weights += target_mass - logsumexp(log_weights, 0)
    return softmax(log_weights)


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
    raw_labels = np.asarray(support_labels)
    if raw_labels.ndim != 1 or not np.issubdtype(raw_labels.dtype, np.integer):
        raise ValueError("Support labels must be a one-dimensional integer array")
    labels = raw_labels.astype(np.int64)
    if (support.ndim != 2 or query.ndim != 2 or not len(support) or not len(query)
            or not np.isfinite(support).all() or not np.isfinite(query).all()):
        raise ValueError("Features must be nonempty finite matrices")
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
    if (prior_means.ndim != 2 or prior_means.shape != prior_variances.shape
            or not len(prior_means) or prior_means.shape[1] != support.shape[1]
            or not np.isfinite(prior_means).all() or not np.isfinite(prior_variances).all()
            or (prior_variances < 0).any()):
        raise ValueError("Prior statistics must be finite matching feature matrices with nonnegative variances")
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
        # Damping towards the shared scalar prevents a noisy transferred
        # diagonal from dominating Euclidean geometry in unseen categories.
        shared = max(float(prior_variances.mean()), 1e-5)
        variances = (1 - config.variance_blend) * shared + config.variance_blend * variances

    def score(current):
        distance = ((query[:, None] - current[None]) ** 2 / variances[None]).mean(2)
        # A common temperature changes confidence but not nearest-prototype
        # predictions. The variance version is diagonal Mahalanobis distance,
        # not a generative density claim.
        return -distance / max(config.temperature, 1e-6)

    if config.normalize_prototypes:
        prototypes = unit_features(prototypes)
    initial_prototypes = prototypes.copy()
    if prototype:
        for _ in range(config.prototype_steps):
            logits = score(prototypes)
            weights = (balanced_soft_assignments(logits) if config.balanced_queries else softmax(logits))
            confidence = weights.max(1)
            accepted = confidence >= config.confidence_threshold
            weights = weights * accepted[:, None]
            masses = weights.sum(0)
            prototypes = (
                shot_counts[:, None] * initial_prototypes +
                config.prototype_weight * weights.T @ query
            ) / (shot_counts[:, None] + config.prototype_weight * masses[:, None])
            if config.normalize_prototypes:
                prototypes = unit_features(prototypes)
    logits = score(prototypes)
    probabilities = (balanced_soft_assignments(logits)
                     if prototype and config.balanced_queries else softmax(logits))
    return {
        "predictions": probabilities.argmax(1).astype(int),
        "probabilities": probabilities,
        "prototypes_before": raw_prototypes,
        "prototypes_after": prototypes,
        "variances": variances,
        "neighbor_ids": neighbor_ids,
        "transductive": bool(prototype),
        "balanced_queries": bool(prototype and config.balanced_queries),
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
