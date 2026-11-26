"""Small trainable calibration on a frozen, locally-trained ResNet12 embedding.

No external weights. A meta-baseline-inspired episode loss coexists with base
CE and two-view mean/variance alignment. This is NOT MAML and does not replace
that required comparison. The existing backbone is unchanged; extra parameters
and train budget must be disclosed. Query labels never enter prediction.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LearnedCalibrationConfig:
    architecture:str='residual'
    dimension:int=128
    hidden:int=128
    residual_scale:float=.5
    ce_weight:float=.25
    episode_weight:float=1.
    alignment_weight:float=.5
    anchor_weight:float=.1
    learning_rate:float=.001
    epochs:int=24
    def __post_init__(self):
        if self.architecture not in ('linear','residual'):raise ValueError('Unsupported adapter')
        for name in ('dimension','hidden','epochs'):
            v=getattr(self,name)
            if type(v) is not int or not 1<=v<=512:raise ValueError('Invalid '+name)
        for name in ('residual_scale','ce_weight','episode_weight','alignment_weight','anchor_weight','learning_rate'):
            v=getattr(self,name)
            if type(v) not in (float,int) or not math.isfinite(v) or v<0:raise ValueError('Invalid '+name)
        if not 0<self.learning_rate<=.01 or not 0<self.residual_scale<=1:raise ValueError('Invalid scale/rate')
    def to_dict(self):return asdict(self)


class LearnedCalibration(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or LearnedCalibrationConfig();c=self.config
        if c.architecture=='linear':
            self.residual=nn.Linear(c.dimension,c.dimension,bias=False)
            nn.init.zeros_(self.residual.weight)
        else:
            self.residual=nn.Sequential(nn.Linear(c.dimension,c.hidden),nn.GELU(),nn.Linear(c.hidden,c.dimension,bias=False))
            nn.init.zeros_(self.residual[-1].weight)
    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=self.config.dimension:raise ValueError('Invalid calibration features')
        return F.normalize(x+self.config.residual_scale*self.residual(x),dim=1)


def episode_objective(z,labels,*,shots,ways=5,generator=None):
    """Episode tasks formed EXCLUSIVELY from known base-train labels.

    Query labels here are base-train supervision; never call this on held-out
    validation/test query labels. Prediction itself has no label truth argument.
    """
    if type(shots) is not int or shots not in (1,5) or ways<2:raise ValueError('Invalid episode')
    classes=torch.unique(labels)
    if len(classes)<ways:raise ValueError('Not enough train classes')
    selected=classes[torch.randperm(len(classes),generator=generator)[:ways]]
    support=[];queries=[];targets=[]
    for i,c in enumerate(selected):
        ids=torch.nonzero(labels==c,as_tuple=False).flatten()
        if len(ids)<=shots:raise ValueError('Not enough independent support/query train samples')
        ids=ids[torch.randperm(len(ids),generator=generator)]
        support.append(z[ids[:shots]].mean(0));queries.append(z[ids[shots:]])
        targets.append(torch.full((len(ids)-shots,),i,dtype=torch.long,device=z.device))
    prototypes=F.normalize(torch.stack(support),dim=1)
    logits=10.*torch.cat(queries)@prototypes.T
    return F.cross_entropy(logits,torch.cat(targets))


def objective(adapter,classifier,x,x_first,x_second,labels,*,shots,generator=None):
    c=adapter.config;z=adapter(x);a=adapter(x_first);b=adapter(x_second)
    ce=F.cross_entropy(10.*z@F.normalize(classifier,dim=1).T,labels,label_smoothing=.05)
    meta=episode_objective(z,labels,shots=shots,generator=generator)
    align=(a.mean(0)-b.mean(0)).square().mean()+(a.var(0,unbiased=False)-b.var(0,unbiased=False)).square().mean()
    # Bound departure from the already useful representation, especially for novel classes.
    anchor=(z-x).square().sum(1).mean()
    total=c.ce_weight*ce+c.episode_weight*meta+c.alignment_weight*align+c.anchor_weight*anchor
    if not torch.isfinite(total):raise ValueError('Non-finite calibration training loss')
    return {'total':total,'ce':ce,'episode':meta,'alignment':align,'anchor':anchor}


def encode_calibration(adapter,features,batch_size=512):
    x=np.asarray(features)
    if x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all():raise ValueError('Expected finite nonempty features')
    if adapter.training:raise ValueError('Calibration prediction requires eval mode')
    result=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):result.append(adapter(torch.tensor(x[start:start+batch_size],dtype=torch.float32)).numpy())
    return np.concatenate(result)
