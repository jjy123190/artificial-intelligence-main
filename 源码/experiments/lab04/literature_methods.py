"""Literature-guided, label-isolated few-shot inference candidates.

Independent equation-level implementations, not vendored upstream code:
* TIM-GD: Boudiaf et al., NeurIPS 2020, arXiv:2008.11297.
* PT-MAP: Hu et al., arXiv:2006.03806, Algorithm 1.

Only support labels enter these functions. Query features interact transductively.
PT-MAP additionally assumes DECLARED equal query class masses. Signed power is
an explicit adaptation for our LeakyReLU features, not an exact paper reproduction.
There is no checkpoint loading, training, validation selection, or test access here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch.nn import functional as F


def _finite_number(value, name, low, high, *, include_low=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or value > high or (value < low if include_low else value <= low):
        raise ValueError(f"Invalid {name}")


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Invalid {name}")


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _batch_inputs(support, support_labels, query):
    s = np.asarray(support, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    y = np.asarray(support_labels)
    if s.ndim != 3 or q.ndim != 3 or y.ndim != 2:
        raise ValueError("Expected batched support/query matrices and support labels")
    if min(s.shape) < 1 or min(q.shape) < 1 or s.shape[0] != q.shape[0] or s.shape[2] != q.shape[2]:
        raise ValueError("Empty or incompatible episode features")
    if y.shape != s.shape[:2] or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Support labels must be an integer matrix aligned with support")
    if not np.isfinite(s).all() or not np.isfinite(q).all():
        raise ValueError("Features must be finite")
    classes = np.unique(y[0])
    if len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("At least two contiguous support classes are required")
    if any(not np.array_equal(np.unique(row), classes) for row in y):
        raise ValueError("All episodes must contain the same number of support classes")
    return s, y.astype(np.int64, copy=False), q, len(classes)


def prepare_episode_batch(support, query, *, signed_power=None, task_center=False):
    """No labels or across-episode statistics; support/query means stay separate.

    The upstream PT-MAP QR step is omitted: an orthonormal change of coordinates
    preserves all norms, means and distances used here. Signed power is named
    explicitly because vanilla sqrt would produce NaNs on negative features.
    """
    s = np.asarray(support, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    if s.ndim != 3 or q.ndim != 3 or min(s.shape) < 1 or min(q.shape) < 1:
        raise ValueError("Expected nonempty batched feature matrices")
    if s.shape[0] != q.shape[0] or s.shape[2] != q.shape[2]:
        raise ValueError("Incompatible feature matrices")
    if not np.isfinite(s).all() or not np.isfinite(q).all():
        raise ValueError("Features must be finite")
    if type(task_center) is not bool:
        raise ValueError("task_center must be boolean")
    if signed_power is not None:
        _finite_number(signed_power, "signed power", 0, 1, include_low=False)
        s = np.sign(s) * np.abs(s) ** signed_power
        q = np.sign(q) * np.abs(q) ** signed_power
    s, q = _unit(s), _unit(q)
    if task_center:
        s = _unit(s - s.mean(axis=1, keepdims=True))
        q = _unit(q - q.mean(axis=1, keepdims=True))
    return s, q


@dataclass(frozen=True)
class MAPConfig:
    inverse_temperature: float = 10.0
    learning_rate: float = 0.2
    steps: int = 20
    sinkhorn_steps: int = 300
    sinkhorn_tolerance: float = 1e-7

    def __post_init__(self):
        _finite_number(self.inverse_temperature, "inverse temperature", 0, 100, include_low=False)
        _finite_number(self.learning_rate, "learning rate", 0, 1, include_low=False)
        _integer(self.steps, "steps", 0, 100)
        _integer(self.sinkhorn_steps, "Sinkhorn steps", 1, 2000)
        _finite_number(self.sinkhorn_tolerance, "Sinkhorn tolerance", 0, 0.01, include_low=False)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TIMConfig:
    temperature: float = 15.0
    learning_rate: float = 1e-4
    steps: int = 1000
    support_weight: float = 0.1
    marginal_weight: float = 1.0
    conditional_weight: float = 0.1

    def __post_init__(self):
        _finite_number(self.temperature, "temperature", 0, 200, include_low=False)
        _finite_number(self.learning_rate, "learning rate", 0, 0.1, include_low=False)
        _integer(self.steps, "steps", 0, 5000)
        _finite_number(self.support_weight, "support weight", 0, 100, include_low=False)
        _finite_number(self.marginal_weight, "marginal weight", 0, 100)
        _finite_number(self.conditional_weight, "conditional weight", 0, 100)

    def to_dict(self):
        return asdict(self)


def _logsumexp(x, axis, keepdims=True):
    peak = x.max(axis=axis, keepdims=True)
    result = peak + np.log(np.exp(x - peak).sum(axis=axis, keepdims=True))
    return result if keepdims else result.squeeze(axis)


def balanced_transport(logits, *, steps=300, tolerance=1e-7):
    """Log-domain Sinkhorn with row mass 1 and declared column mass N/K.

    No query labels, inferred class counts, or hidden ground-truth proportions.
    Fractional N/K is valid for a soft transport plan, unlike hard assignments.
    """
    z = np.asarray(logits, dtype=np.float64)
    _integer(steps, "Sinkhorn steps", 1, 2000)
    _finite_number(tolerance, "Sinkhorn tolerance", 0, 0.01, include_low=False)
    if z.ndim != 3 or min(z.shape) < 1 or z.shape[-1] < 2 or not np.isfinite(z).all():
        raise ValueError("Invalid batched transport logits")
    z = z - z.max(axis=(1, 2), keepdims=True)
    target = z.shape[1] / z.shape[2]
    for iteration in range(steps):
        z = z - _logsumexp(z, axis=2)
        z = z + math.log(target) - _logsumexp(z, axis=1)
        if iteration % 5 == 4:
            probability = np.exp(z)
            residual = max(float(np.abs(probability.sum(2) - 1).max()),
                           float(np.abs(probability.sum(1) - target).max()))
            if residual <= tolerance:
                break
    probability = np.exp(z)
    residual = max(float(np.abs(probability.sum(2) - 1).max()),
                   float(np.abs(probability.sum(1) - target).max()))
    return probability, residual


def map_predict_batch(support, support_labels, query, config=None):
    """Damped prototype updates using entropic optimal-transport assignments."""
    config = config or MAPConfig()
    s, y, q, ways = _batch_inputs(support, support_labels, query)
    s, q = _unit(s), _unit(q)
    one_hot = np.eye(ways)[y]
    counts = one_hot.sum(1)
    support_sum = np.einsum("bnk,bnd->bkd", one_hot, s)
    centers = support_sum / counts[:, :, None]
    before = centers.copy()
    worst_residual = 0.0
    for iteration in range(config.steps + 1):
        distances = (np.sum(q * q, axis=2, keepdims=True)
                     + np.sum(centers * centers, axis=2)[:, None, :]
                     - 2 * np.einsum("bnd,bkd->bnk", q, centers))
        probability, residual = balanced_transport(
            -config.inverse_temperature * distances,
            steps=config.sinkhorn_steps, tolerance=config.sinkhorn_tolerance)
        worst_residual = max(worst_residual, residual)
        if iteration < config.steps:
            estimates = (support_sum + np.einsum("bnk,bnd->bkd", probability, q))
            estimates /= (counts + probability.sum(1))[:, :, None]
            centers += config.learning_rate * (estimates - centers)
    if not np.isfinite(probability).all() or worst_residual > 1e-3:
        raise ValueError("Transport failed to converge; do not silently use bad masses")
    return {"predictions": probability.argmax(2), "probabilities": probability,
            "prototypes_before": before, "prototypes_after": centers,
            "transductive": True, "balanced_queries": True,
            "max_transport_residual": worst_residual}


def tim_objective(logits_s, labels_s, logits_q, config):
    """Per-task .1 CE - H(mean p_q) + .1 mean H(p_q); sum, never mix tasks."""
    log_s, log_q = logits_s.log_softmax(-1), logits_q.log_softmax(-1)
    cross_entropy = -log_s.gather(-1, labels_s.unsqueeze(-1)).squeeze(-1).mean(1)
    probability = log_q.exp()
    conditional = -(probability * log_q).sum(-1).mean(1)
    marginal = probability.mean(1)
    marginal_entropy = -(marginal * marginal.clamp_min(1e-15).log()).sum(-1)
    return (config.support_weight * cross_entropy - config.marginal_weight * marginal_entropy
            + config.conditional_weight * conditional).sum()


def tim_predict_batch(support, support_labels, query, config=None):
    """TIM-GD with a frozen embedding and task-local Adam-updated prototypes."""
    config = config or TIMConfig()
    s_np, y_np, q_np, ways = _batch_inputs(support, support_labels, query)
    # Explicit float64 and detached copies: no gradients into cached embeddings.
    with torch.inference_mode(False), torch.enable_grad():
        s = torch.tensor(_unit(s_np), dtype=torch.float64)
        q = torch.tensor(_unit(q_np), dtype=torch.float64)
        y = torch.tensor(y_np, dtype=torch.long)
        one_hot = F.one_hot(y, ways).to(s.dtype)
        centers = (one_hot.transpose(1, 2) @ s) / one_hot.sum(1).unsqueeze(-1)
        before = centers.detach().numpy().copy()
        centers = centers.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([centers], lr=config.learning_rate)

        def logits(x):
            return config.temperature * (x @ centers.transpose(1, 2)
                   - 0.5 * centers.square().sum(2).unsqueeze(1)
                   - 0.5 * x.square().sum(2).unsqueeze(2))

        losses = []
        for iteration in range(config.steps):
            loss = tim_objective(logits(s), y, logits(q), config)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite TIM objective")
            if iteration == 0:
                losses.append(float(loss.detach()) / len(s))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            losses.append(float(tim_objective(logits(s), y, logits(q), config)) / len(s))
            probability = logits(q).softmax(2).numpy().copy()
            after = centers.numpy().copy()
    return {"predictions": probability.argmax(2), "probabilities": probability,
            "prototypes_before": before, "prototypes_after": after,
            "objective_before": losses[0], "objective_after": losses[-1],
            "transductive": True, "balanced_queries": False,
            "marginal_entropy_prior": "uniform-favoring, not a hard count constraint"}
