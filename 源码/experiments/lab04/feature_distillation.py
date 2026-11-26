"""Train-only, same-view feature/relational distillation for a light ResNet12.

This implements a PROPOSED training primitive, not a measured improvement and
not an exact RFS reproduction. The teacher supplies features, not an invented
64-class classifier. All validation/test feature distillation must be forbidden
by the future dataset runner. The external lineage must remain disclosed.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FeatureDistillationConfig:
    feature_weight: float = 1.0
    relation_weight: float = .5
    alignment_weight: float = .5
    metric_weight: float = .25
    smoothing: float = .05

    def __post_init__(self):
        for name in ('feature_weight','relation_weight','alignment_weight','metric_weight'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=10:
                raise ValueError(f'Invalid {name}')
        if isinstance(self.smoothing,bool) or not isinstance(self.smoothing,(int,float)) or not math.isfinite(self.smoothing) or not 0<=self.smoothing<1:
            raise ValueError('Invalid label smoothing')

    def to_dict(self):return asdict(self)


class FeatureProjector(nn.Module):
    """Training-only projection; remove from the deployed student's checkpoint."""
    def __init__(self, student_dim=128, teacher_dim=640):
        super().__init__()
        if any(type(v) is not int or v<1 for v in (student_dim,teacher_dim)):
            raise ValueError('Positive integer embedding dimensions required')
        self.linear=nn.Linear(student_dim,teacher_dim,bias=False)

    def forward(self, features):return self.linear(features)


def distillation_terms(student, teacher, projector):
    if student.ndim!=2 or teacher.ndim!=2 or len(student)!=len(teacher) or len(student)<2:
        raise ValueError('At least two paired train features required')
    if student.device!=teacher.device or not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError('Invalid paired feature tensors')
    # This detach is required even if the caller forgot inference_mode/no_grad.
    target=F.normalize(teacher.detach(),dim=1)
    projected=F.normalize(projector(student),dim=1)
    if target.shape!=projected.shape:raise ValueError('Projection and teacher dimensions disagree')
    feature=(1-(projected*target).sum(1)).mean()
    original=F.normalize(student,dim=1)
    delta=(original@original.T-target@target.T).square()
    n=len(student)
    relation=(delta.sum()-delta.diagonal().sum())/(n*(n-1))
    return {'feature':feature,'relation':relation}


def training_loss(logits, labels, student_a, student_b, teacher_a, teacher_b,
                  projector, metric_loss, config=None):
    """CE + feature KD + relational KD + two-view mean/variance + metric.

    Both teacher views must correspond to the same underlying augmentation as
    their student views. Do NOT supervise a random crop with an unrelated cache.
    This function has no query labels and must only receive base-train minibatches.
    """
    config=config or FeatureDistillationConfig()
    if logits.ndim!=2 or labels.ndim!=1 or len(logits)!=len(labels) or len(logits)!=len(student_a):
        raise ValueError('Supervised training tensors disagree')
    if student_a.shape!=student_b.shape or teacher_a.shape!=teacher_b.shape:
        raise ValueError('Both views must have matching shapes')
    if labels.dtype!=torch.long or torch.any(labels<0) or torch.any(labels>=logits.shape[1]):
        raise ValueError('Invalid base-train classification labels')
    if not torch.is_tensor(metric_loss) or metric_loss.ndim!=0 or not torch.isfinite(metric_loss):
        raise ValueError('A finite scalar metric loss is required')
    a=distillation_terms(student_a,teacher_a,projector)
    b=distillation_terms(student_b,teacher_b,projector)
    ce=F.cross_entropy(logits,labels,label_smoothing=config.smoothing)
    x,y=F.normalize(student_a,dim=1),F.normalize(student_b,dim=1)
    alignment=(x.mean(0)-y.mean(0)).square().mean()+(x.var(0,unbiased=False)-y.var(0,unbiased=False)).square().mean()
    feature=(a['feature']+b['feature'])/2;relation=(a['relation']+b['relation'])/2
    total=(ce+config.feature_weight*feature+config.relation_weight*relation
           +config.alignment_weight*alignment+config.metric_weight*metric_loss)
    if not torch.isfinite(total):raise ValueError('Non-finite distillation loss')
    return {'total':total,'ce':ce,'feature':feature,'relation':relation,'alignment':alignment,'metric':metric_loss}
