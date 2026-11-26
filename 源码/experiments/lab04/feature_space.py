"""Training-only feature centering, power normalization, and reproducible priors."""
from __future__ import annotations

import numpy as np
from .calibration import base_statistics, unit_features


def transform_features(features, transform):
    x = unit_features(features)
    power = float(np.asarray(transform.get("power", 1.0)))
    center = np.asarray(transform.get("center", np.zeros(x.shape[1])), dtype=np.float64)
    if (not 0 < power <= 1 or center.shape != (x.shape[1],) or not np.isfinite(center).all()):
        raise ValueError("Invalid training-derived feature transform")
    x = x - center
    x = np.sign(x) * np.abs(x) ** power
    return unit_features(x)


def fit_feature_space(training_features, training_labels, *, center_strength=0.0, power=1.0):
    if not 0 <= center_strength <= 1:
        raise ValueError("Center strength must be in [0, 1]")
    transform = {"center": unit_features(training_features).mean(0) * center_strength,
                 "power": np.array(power), "center_strength": np.array(center_strength)}
    features = transform_features(training_features, transform)
    return {**base_statistics(features, training_labels), **transform}


def select_prior(prior, shots, *, transformed=False):
    """Read shot-specific training statistics without breaking v1 artifacts."""
    prefix = f"shot{shots}_"
    if transformed and prefix + "means" in prior:
        return {key[len(prefix):]: value for key, value in prior.items() if key.startswith(prefix)}
    return {key: value for key, value in prior.items() if not key.startswith("shot")}
