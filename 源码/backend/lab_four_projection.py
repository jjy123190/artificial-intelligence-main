"""Post-prediction PCA with a shared basis and isotropic display coordinates."""
from __future__ import annotations


def project(before, after, prototypes_before, prototypes_after):
    import numpy as np
    arrays = [np.asarray(x, dtype=np.float64) for x in (before, after, prototypes_before, prototypes_after)]
    first, second, pa, pb = arrays
    if (any(x.ndim != 2 or not len(x) or not np.isfinite(x).all() for x in arrays)
            or first.shape != second.shape or pa.shape != pb.shape
            or any(x.shape[1] != first.shape[1] for x in arrays) or first.shape[1] < 2):
        raise ValueError("Projection needs finite matching feature matrices")
    # Fit to actual sample features, not prototype locations. Changing only a
    # prototype cannot invent a new PCA basis for unchanged image features.
    samples = np.concatenate((first, second))
    center = samples.mean(axis=0)
    _, singular, vectors = np.linalg.svd(samples - center, full_matrices=False)
    basis = vectors[:2].T
    for col in range(2):
        if basis[np.argmax(np.abs(basis[:, col])), col] < 0:
            basis[:, col] *= -1
    all_points = np.concatenate(arrays)
    projected = (all_points - center) @ basis
    low, high = projected.min(0), projected.max(0)
    span = max(float((high - low).max()), 1e-8)
    # One scale for both axes preserves the geometry of this 2D projection.
    # This must not be mistaken for full high-dimensional decision geometry.
    projected = .5 + .88 * (projected - (low + high) / 2) / span
    energy = singular ** 2
    variance = energy[:2] / max(float(energy.sum()), 1e-12)
    n, k = len(first), len(pa)
    return {"before":projected[:n], "after":projected[n:2*n],
            "prototypes_before":projected[2*n:2*n+k], "prototypes_after":projected[2*n+k:],
            "explained_variance":variance.tolist(), "basis_source":"sample features before and after prediction only",
            "isotropic":True}
