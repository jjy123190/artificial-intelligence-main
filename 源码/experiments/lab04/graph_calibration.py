"""Experimental label propagation and balanced prototype classification.

Prediction functions deliberately have no query-label argument. All query
interaction is transductive and must be disclosed, including class balance.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from .calibration import unit_features, softmax, balanced_soft_assignments


@dataclass(frozen=True)
class GraphConfig:
    neighbors:int=10
    alpha:float=.9
    power:float=3.
    temperature:float=.1
    prototype_weight:float=.2
    prototype_steps:int=8
    graph_weight:float=.5
    balanced:bool=True

    def __post_init__(self):
        if not 0<=self.alpha<1 or not 0<=self.graph_weight<=1:
            raise ValueError("Invalid propagation coefficient")
        if self.neighbors<1 or self.temperature<=0 or self.power<=0 or self.prototype_weight<0:
            raise ValueError("Invalid graph or prototype setting")
        if not 0<=self.prototype_steps<=50:
            raise ValueError("Invalid prototype iterations")

    def to_dict(self):return asdict(self)


def centered_whitening(training_features,strength=0.,exponent=0.):
    if not 0<=strength<=1 or not 0<=exponent<=1:raise ValueError("Invalid transform")
    x=unit_features(training_features);center=x.mean(0)*strength
    covariance=np.cov(x,rowvar=False)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    floor=max(float(np.mean(eigenvalues))*.1,1e-6)
    scale=np.maximum(eigenvalues,floor)**(-exponent/2)
    matrix=(eigenvectors*scale[None])@eigenvectors.T
    return {"center":center,"matrix":matrix,"strength":np.array(strength),"exponent":np.array(exponent)}


def apply_transform(features,transform):
    x=unit_features(features)
    return unit_features((x-transform["center"])@transform["matrix"])


def predict(support,support_labels,query,config=None):
    config=config or GraphConfig()
    support,query=unit_features(support),unit_features(query)
    labels=np.asarray(support_labels)
    classes=np.unique(labels);ways=len(classes)
    if not np.array_equal(classes,np.arange(ways)) or len(labels)!=len(support):
        raise ValueError("Support labels must be contiguous task labels")
    prototypes=np.stack([support[labels==c].mean(0) for c in classes])
    initial=prototypes.copy();counts=np.bincount(labels,minlength=ways)
    for _ in range(config.prototype_steps):
        logits=-((query[:,None]-prototypes[None])**2).sum(2)/config.temperature
        weights=balanced_soft_assignments(logits) if config.balanced else softmax(logits)
        prototypes=(counts[:,None]*initial+config.prototype_weight*weights.T@query)/(counts[:,None]+config.prototype_weight*weights.sum(0)[:,None])
    logits=-((query[:,None]-prototypes[None])**2).sum(2)/config.temperature
    probabilities=balanced_soft_assignments(logits) if config.balanced else softmax(logits)
    if config.graph_weight>0:
        features=np.concatenate((support,query));n=len(features)
        affinity=np.maximum(features@features.T,0.)**config.power
        np.fill_diagonal(affinity,0.)
        k=min(config.neighbors,n-1)
        selected=np.argpartition(affinity,n-k,axis=1)[:,-k:]
        mask=np.zeros_like(affinity,dtype=bool);np.put_along_axis(mask,selected,True,axis=1)
        affinity=affinity*mask;affinity=np.maximum(affinity,affinity.T)
        degrees=affinity.sum(1);scale=np.sqrt(np.maximum(degrees,1e-12))
        normalized=affinity/scale[:,None]/scale[None,:]
        target=np.zeros((n,ways));target[np.arange(len(support)),labels]=1.
        # Solve the label propagation fixed point once; only supports seed it.
        propagated=np.linalg.solve(np.eye(n)-config.alpha*normalized,target)[len(support):]
        propagated=np.maximum(propagated,1e-12)
        propagated=propagated/propagated.sum(1,keepdims=True)
        probabilities=(1-config.graph_weight)*probabilities+config.graph_weight*propagated
        if config.balanced:probabilities=balanced_soft_assignments(np.log(np.maximum(probabilities,1e-12)))
    return {"predictions":probabilities.argmax(1),"probabilities":probabilities,
            "transductive":bool(config.graph_weight or config.prototype_steps or config.balanced),
            "balanced_queries":config.balanced,"prototypes_before":initial,"prototypes_after":prototypes}
