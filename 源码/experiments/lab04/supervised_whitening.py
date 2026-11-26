"""Train-label-only residual covariance: distinguish within-class nuisance from class separation."""
from __future__ import annotations
import numpy as np
from .calibration import unit_features


def fit(training_features,training_labels,*,strength=1.,exponent=.5,residual_mix=.5):
    if not 0<=strength<=1 or not 0<=exponent<=1 or not 0<=residual_mix<=1:
        raise ValueError("Whitening coefficients must lie in [0,1]")
    x=unit_features(training_features);labels=np.asarray(training_labels)
    if labels.ndim!=1 or len(labels)!=len(x) or len(x)<2:raise ValueError("Invalid train labels")
    center=x.mean(0)*strength
    total=np.cov(x,rowvar=False)
    residual=x.copy()
    for cls in np.unique(labels):
        selected=labels==cls
        residual[selected]-=x[selected].mean(0)
    within=(residual.T@residual)/max(len(x)-len(np.unique(labels)),1)
    # Rescale to match total variance: residual_mix controls directions rather
    # than an arbitrary overall distance scale. All statistics are base-train.
    within*=np.trace(total)/max(float(np.trace(within)),1e-12)
    covariance=(1-residual_mix)*total+residual_mix*within
    eigenvalues,vectors=np.linalg.eigh(covariance)
    floor=max(float(eigenvalues.mean())*.1,1e-6)
    matrix=(vectors*np.maximum(eigenvalues,floor)[None]**(-exponent/2))@vectors.T
    return {"center":center,"matrix":matrix,"strength":np.array(strength),
            "exponent":np.array(exponent),"residual_mix":np.array(residual_mix)}
