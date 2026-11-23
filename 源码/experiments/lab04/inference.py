"""Portable live inference used by the course website's lab04 adapter.

All methods sample exactly the same episode for the same shot count and seed.
The accuracy of one live episode is not a benchmark confidence interval.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from threading import RLock
import time

import numpy as np
import torch

from .calibration import CalibrationConfig, calibrated_predict, unit_features
from .data import digest_file, normalized_tensor, sample_episode
from .experiment import load_checkpoint
from .models import maml_adapt

_INFERENCE_LOCK = RLock()
_ARTIFACT_VERSIONS = {}


def load_demo_pool(artifact_dir):
    """Return the fixed, held-out image pool without pickle deserialization."""
    directory = str(Path(artifact_dir).resolve())
    info = (Path(directory) / "demo_pool.npz").stat()
    return _load_pool_cached(directory, info.st_mtime_ns, info.st_size)


@lru_cache(maxsize=3)
def _load_pool_cached(artifact_dir, modified_ns, size):
    path = Path(artifact_dir) / "demo_pool.npz"
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


@lru_cache(maxsize=6)
def _model(artifact_dir, name):
    model, _ = load_checkpoint(Path(artifact_dir) / f"{name}.pt")
    return model


@lru_cache(maxsize=6)
def _prior(artifact_dir, name):
    with np.load(Path(artifact_dir) / f"{name}_prior.npz", allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def clear_cache():
    with _INFERENCE_LOCK:
        _load_pool_cached.cache_clear()
        _model.cache_clear()
        _prior.cache_clear()
        _ARTIFACT_VERSIONS.clear()


def _ensure_artifact_version(directory, metadata, manifest_digest):
    filenames = ["baseline.pt", "aligned.pt", "maml.pt", "demo_pool.npz",
                 "baseline_prior.npz", "aligned_prior.npz"]
    observations = []
    for name in filenames:
        stat = (Path(directory) / name).stat()
        observations.append((name, stat.st_mtime_ns, stat.st_size))
    version = (manifest_digest, tuple(observations))
    if _ARTIFACT_VERSIONS.get(directory) == version:
        return
    for name in ("baseline", "aligned", "maml"):
        actual = digest_file(Path(directory) / f"{name}.pt")
        if actual != metadata["weights"][name]["sha256"]:
            raise ValueError(f"Checksum mismatch for {name}.pt; artifacts are being replaced or are inconsistent")
    if digest_file(Path(directory) / "demo_pool.npz") != metadata["demo_pool"]["sha256"]:
        raise ValueError("Checksum mismatch for demo_pool.npz")
    for name, record in metadata.get("priors", {}).items():
        if name not in {"baseline", "aligned"}:
            raise ValueError("Unexpected prior name in artifact manifest")
        if digest_file(Path(directory) / f"{name}_prior.npz") != record["sha256"]:
            raise ValueError(f"Checksum mismatch for {name}_prior.npz")
    _load_pool_cached.cache_clear()
    _model.cache_clear()
    _prior.cache_clear()
    _ARTIFACT_VERSIONS[directory] = version


def _prototypes(features, labels, ways=5):
    return np.stack([features[labels == label].mean(0) for label in range(ways)])


def run_episode(artifact_dir, shots=1, seed=20261122, method="calibrated", calibration="full"):
    """Run a measured 5-way episode on actual images using saved weights.

    Accepted methods: baseline, maml, calibrated.
    Calibration modes: none, mean, variance, prototype, full.
    Samples and prototypes are 32-dimensional unit-feature coordinates.
    For calibration, feature vectors do not move; only the prototype and
    distance metric change. For MAML, before/after features reflect actual
    updates to all network parameters using support images and labels.
    """
    if shots not in (1, 5):
        raise ValueError("shots must be 1 or 5")
    if not isinstance(seed, int) or not 0 <= seed <= 2 ** 32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    if method not in {"baseline", "maml", "calibrated"}:
        raise ValueError("Unknown method")
    mode_flags = {
        "none": dict(mean=False, variance=False, prototype=False),
        "mean": dict(mean=True, variance=False, prototype=False),
        "variance": dict(mean=False, variance=True, prototype=False),
        "prototype": dict(mean=False, variance=False, prototype=True),
        "full": dict(mean=True, variance=True, prototype=True),
    }
    if calibration not in mode_flags:
        raise ValueError("Unknown calibration strategy")
    directory = str(Path(artifact_dir).resolve())
    result_file = Path(directory) / "results.json"
    manifest_bytes = result_file.read_bytes()
    metadata = json.loads(manifest_bytes)
    if metadata.get("status") != "complete":
        raise RuntimeError("Measured experiment artifacts are not complete")
    # torch.func.functional_call temporarily swaps module parameters. It is
    # unsafe to interleave on a shared cached model without serialization.
    with _INFERENCE_LOCK:
        _ensure_artifact_version(directory, metadata, hashlib.sha256(manifest_bytes).hexdigest())
        torch.set_num_threads(min(int(metadata.get("runtime", {}).get("threads", 6)), 8))
        pool = load_demo_pool(directory)
        task = sample_episode(pool["labels"], np.random.default_rng(seed), shots=shots, queries=15)
        support_indices, query_indices = task["support_indices"], task["query_indices"]
        support_labels = task["support_labels"]
        support_images = normalized_tensor(pool["images"][support_indices])
        query_images = normalized_tensor(pool["images"][query_indices])
        source = {"baseline": "baseline", "maml": "maml", "calibrated": "aligned"}[method]
        network = _model(directory, source)
        config = CalibrationConfig(**metadata["calibration"][str(shots)])
        started = time.perf_counter()
        with torch.inference_mode():
            support_before = unit_features(network.features(support_images).numpy())
            query_before = unit_features(network.features(query_images).numpy())
        prototype_before = _prototypes(support_before, support_labels)
        if method == "maml":
            with torch.enable_grad():
                adapted = maml_adapt(network, support_images, torch.from_numpy(support_labels),
                                     steps=3, learning_rate=0.1, second_order=False)
            with torch.no_grad():
                logits, query_features = torch.func.functional_call(
                    network, adapted, (query_images,), {"return_features": True})
                _, support_features = torch.func.functional_call(
                    network, adapted, (support_images,), {"return_features": True})
                probabilities = logits.softmax(1).numpy()
            support_after = unit_features(support_features.numpy())
            query_after = unit_features(query_features.numpy())
            prototype_after = _prototypes(support_after, support_labels)
            prediction = probabilities.argmax(1)
            transductive, metric = False, "adapted_classifier_logits"
            actual_config = {"inner_steps": 3, "inner_learning_rate": 0.1,
                             "meta_training": "second_order_MAML", "adapt_all_parameters": True}
        else:
            flags = mode_flags[calibration] if method == "calibrated" else mode_flags["none"]
            prediction_result = calibrated_predict(support_before, support_labels, query_before,
                                                    _prior(directory, source), config=config, **flags)
            probabilities = prediction_result["probabilities"]
            prediction = prediction_result["predictions"]
            prototype_before = prediction_result["prototypes_before"]
            prototype_after = prediction_result["prototypes_after"]
            support_after, query_after = support_before, query_before
            transductive = prediction_result["transductive"]
            metric = "diagonal_mahalanobis" if flags["variance"] else "squared_euclidean"
            actual_config = {**config.to_dict(), **flags}
        inference_ms = (time.perf_counter() - started) * 1000
        # Query labels enter only here, after all predictions are final.
        query_labels = task["query_labels"]
        classes = [{"label": label, "wnid": str(pool["class_wnids"][original]),
                    "name": str(pool["class_names"][original])}
                   for label, original in enumerate(task["classes"])]
        support_rows = [
            {"id": str(pool["ids"][index]), "label": int(support_labels[position]),
             "class_name": classes[int(support_labels[position])]["name"],
             "feature": support_after[position].tolist(),
             "feature_before": support_before[position].tolist(),
             "feature_after": support_after[position].tolist(), "image_index": int(index)}
            for position, index in enumerate(support_indices)
        ]
        query_rows = [
            {"id": str(pool["ids"][index]), "label": int(query_labels[position]),
             "class_name": classes[int(query_labels[position])]["name"],
             "prediction": int(prediction[position]),
             "prediction_name": classes[int(prediction[position])]["name"],
             "correct": bool(prediction[position] == query_labels[position]),
             "confidence": float(probabilities[position].max()),
             "probabilities": probabilities[position].tolist(),
             "feature": query_after[position].tolist(),
             "feature_before": query_before[position].tolist(),
             "feature_after": query_after[position].tolist(), "image_index": int(index)}
            for position, index in enumerate(query_indices)
        ]
        return {"schema_version": 1, "shots": shots, "seed": seed, "method": method,
                "calibration": calibration if method == "calibrated" else "none",
                "ways": 5, "queries_per_class": 15,
                "accuracy": float(np.mean(prediction == query_labels)),
                "correct": int(np.sum(prediction == query_labels)), "total": len(query_labels),
                "inference_ms": inference_ms, "transductive": transductive, "metric": metric,
                "classes": classes, "class_names": [row["name"] for row in classes],
                "support": support_rows, "query": query_rows,
                "prototypes_before": prototype_before.tolist(),
                "prototypes_after": prototype_after.tolist(),
                "prototype_role": "descriptive_class_centers" if method == "maml" else "decision_prototypes",
                "feature_motion": "real_support_adaptation" if method == "maml" else "unchanged",
                "config": actual_config,
                "checkpoint_sha256": metadata["weights"][source]["sha256"],
                "pool_hash": metadata["demo_pool"]["sha256"],
                "probability_note": "softmax model scores; not a calibrated probability guarantee"}
