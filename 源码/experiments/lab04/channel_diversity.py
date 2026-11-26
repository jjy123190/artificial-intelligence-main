"""Fan-out diversification for a previously duplicated lightweight ResNet12.

Net2Net-inspired engineering intervention, not an exact paper reproduction.
For equal adjacent INPUT channels, adding +noise/-noise to their fan-out keeps
the forward function identical but makes upstream gradients different. Current
trained pairs are near-equal, not exact, so the real-model change must be measured
and the unmodified epoch-zero checkpoint must remain eligible. No external data.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def diversify_fanout(model,*,relative_scale=.25,seed=20261204):
 if type(relative_scale) not in (float,int) or not math.isfinite(relative_scale) or not 0<relative_scale<=1:raise ValueError('Invalid fan-out scale')
 generator=torch.Generator().manual_seed(seed);changed=[]
 with torch.no_grad():
  for name,module in model.named_modules():
   if not isinstance(module,(nn.Conv2d,nn.Linear)):continue
   weights=module.weight
   if weights.shape[1]%2:continue # RGB layer is not duplicated.
   even=weights[:,0::2];odd=weights[:,1::2]
   # Zero-sum over each duplicated pair; do not scale existing information.
   sigma=float(weights.std(unbiased=False).clamp_min(1e-8))*relative_scale
   noise=torch.randn(even.shape,generator=generator,dtype=weights.dtype).to(weights.device)*sigma
   even.add_(noise);odd.sub_(noise);changed.append(name)
 return changed


def pair_correlation_penalty(features):
 """Remove redundant adjacent output directions, with a variance safeguard.

 This is a targeted diagnostic penalty, not proof that all correlated features
 are useless. Do not use held-out labels/features for regularizer fitting.
 """
 if features.ndim!=2 or features.shape[0]<4 or features.shape[1]%2:raise ValueError('Need finite even-channel train features')
 centered=features-features.mean(0)
 std=(centered.square().mean(0)+1e-4).sqrt()
 normalized=centered/std
 correlation=(normalized[:,0::2]*normalized[:,1::2]).mean(0)
 return correlation.square().mean()


def cosine_episode_loss(features,labels,*,shots,generator):
 if features.ndim!=2 or labels.ndim!=1 or len(features)!=len(labels) or shots not in (1,5):raise ValueError('Invalid base episode')
 classes=torch.unique(labels)
 if len(classes)<5:raise ValueError('Need five base classes')
 classes=classes[torch.randperm(len(classes),generator=generator)[:5]]
 z=F.normalize(features,dim=1);prototypes=[];queries=[];targets=[]
 for i,c in enumerate(classes):
  ids=torch.nonzero(labels==c,as_tuple=False).flatten()
  if len(ids)<=shots:raise ValueError('Insufficient independent train support/query')
  ids=ids[torch.randperm(len(ids),generator=generator)]
  prototypes.append(z[ids[:shots]].mean(0));queries.append(z[ids[shots:]])
  targets.append(torch.full((len(ids)-shots,),i,dtype=torch.long,device=z.device))
 centers=F.normalize(torch.stack(prototypes),dim=1)
 return F.cross_entropy(10.*torch.cat(queries)@centers.T,torch.cat(targets))
