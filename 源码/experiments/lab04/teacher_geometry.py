"""CPU-friendly transfer of a fixed external teacher geometry, not a new backbone.

RFS motivates representation distillation, but this is NOT the RFS algorithm:
it is an explicitly separate ridge-regression feature-calibration experiment.
Only matching base-train image features are fitted. The teacher never encodes
validation/test queries at deployment; a compact 128x128 metric factor suffices.
Do not mix this external-teacher lineage with the no-external-weight baseline.
"""
from __future__ import annotations

import math
import numpy as np


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite feature matrix")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def fit_teacher_geometry(student_train, teacher_train, *, ridge_ratio=.1, mix=1.0):
    """Ridge min ||XW-Y||² + lambda||W||²; compress WW^T to a PSD metric.

    Identity order MUST be checked by the caller. Center/scale are train-only.
    Trace-normalized geometry is mixed with identity for a declared safeguard.
    Returned matrices need no teacher, training images, or query labels at use.
    """
    if isinstance(ridge_ratio, bool) or not isinstance(ridge_ratio, (float, int)) or not math.isfinite(ridge_ratio) or not 0 < ridge_ratio <= 10:
        raise ValueError("Invalid ridge ratio")
    if isinstance(mix, bool) or not isinstance(mix, (float, int)) or not math.isfinite(mix) or not 0 <= mix <= 1:
        raise ValueError("Invalid teacher mix")
    student, teacher = unit(student_train), unit(teacher_train)
    if len(student) != len(teacher) or len(student) < 2:
        raise ValueError("Paired training samples are required")
    center = student.mean(0)
    x, y = student - center, teacher - teacher.mean(0)
    covariance = x.T @ x / len(x)
    cross = x.T @ y / len(x)
    regularization = ridge_ratio * max(float(np.trace(covariance) / len(covariance)), 1e-12)
    regression = np.linalg.solve(covariance + regularization * np.eye(len(covariance)), cross)
    teacher_metric = regression @ regression.T
    scale = float(np.trace(teacher_metric) / len(teacher_metric))
    if scale <= 1e-15:
        raise ValueError("Teacher/student training covariance is degenerate")
    metric = (1 - mix) * np.eye(len(covariance)) + mix * teacher_metric / scale
    eigenvalues, vectors = np.linalg.eigh(metric)
    if eigenvalues.min() < -1e-9:
        raise ValueError("Unexpected non-PSD metric")
    # A symmetric square root preserves the embedding dimension and all scores.
    matrix = (vectors * np.sqrt(np.maximum(eigenvalues, 0))[None]) @ vectors.T
    return {"center": center, "matrix": matrix, "ridge_ratio": np.array(ridge_ratio),
            "mix": np.array(mix), "regression": regression, "metric_scale": np.array(scale)}


def apply_teacher_geometry(features, transform):
    x = unit(features)
    center, matrix = np.asarray(transform["center"]), np.asarray(transform["matrix"])
    if center.shape != (x.shape[1],) or matrix.shape != (x.shape[1], x.shape[1]):
        raise ValueError("Geometry dimensions disagree")
    if not np.isfinite(center).all() or not np.isfinite(matrix).all():
        raise ValueError("Non-finite geometry")
    return unit((x - center) @ matrix)
