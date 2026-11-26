"""Experimental training primitives; the published v2 models remain unchanged."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .models import ResNet12


def widen_resnet12(source, factor=2):
    """Channel duplication preserves convolutions and GroupNorm group moments."""
    if type(factor) is not int or factor < 1:
        raise ValueError("factor must be a positive integer")
    wider = ResNet12(classes=source.classifier.out_features,
                     widths=[n * factor for n in source.widths])
    originals = dict(source.named_modules())
    with torch.no_grad():
        for name, layer in wider.named_modules():
            old = originals.get(name)
            if isinstance(layer, nn.Conv2d):
                out_factor = layer.out_channels // old.out_channels
                in_factor = layer.in_channels // old.in_channels
                weights = old.weight.repeat_interleave(out_factor, 0)
                layer.weight.copy_(weights.repeat_interleave(in_factor, 1) / in_factor)
            elif isinstance(layer, nn.GroupNorm):
                repeat = layer.num_channels // old.num_channels
                layer.weight.copy_(old.weight.repeat_interleave(repeat))
                layer.bias.copy_(old.bias.repeat_interleave(repeat))
            elif isinstance(layer, nn.Linear):
                layer.weight.copy_(old.weight.repeat_interleave(factor, 1) / factor)
                layer.bias.copy_(old.bias)
    return wider


def break_channel_symmetry(model, scale=0.01, generator=None):
    if not 0 <= scale <= .1:
        raise ValueError("Invalid perturbation scale")
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                noise = torch.randn(layer.weight.shape, generator=generator)
                layer.weight.add_(noise * layer.weight.std().clamp_min(1e-6) * scale)


def balanced_batches(labels, rng, ways=16, samples=4):
    """Use each base image exactly once, with equal samples per selected class."""
    import numpy as np
    classes = np.unique(labels)
    if len(classes) % ways:
        raise ValueError("Class count must be divisible by ways")
    pools = [rng.permutation(np.flatnonzero(labels == c)) for c in classes]
    lengths = [len(p) for p in pools]
    if len(set(lengths)) != 1 or lengths[0] % samples:
        raise ValueError("Classes require equal divisible image counts")
    for offset in range(0, lengths[0], samples):
        order = rng.permutation(len(classes))
        for start in range(0, len(order), ways):
            yield np.concatenate([pools[int(i)][offset:offset+samples] for i in order[start:start+ways]])


def episode_loss(features, labels, ways=16, samples=4, temperature=.1):
    """Prototype loss within grouped BASE-class minibatches, no test classes."""
    if len(features) != ways * samples or samples < 2 or samples % 2:
        raise ValueError("Expected complete even class groups")
    grouped = labels.reshape(ways, samples)
    if not bool((grouped == grouped[:, :1]).all()) or len(torch.unique(grouped[:, 0])) != ways:
        raise ValueError("Batch must be grouped by distinct labels")
    z = F.normalize(features, dim=1).reshape(ways, samples, -1)
    centers = F.normalize(z[:, :samples//2].mean(1), dim=1)
    query = z[:, samples//2:].reshape(-1, z.shape[-1])
    target = torch.arange(ways, device=features.device).repeat_interleave(samples//2)
    return F.cross_entropy(query @ centers.T / temperature, target)


def augment_images(images, generator):
    """Independent crop/flip/color perturbation per training image."""
    inputs = torch.from_numpy(images).permute(0,3,1,2).float() / 255
    n, _, h, w = inputs.shape
    padded = F.pad(inputs, (8,8,8,8), mode="reflect")
    offsets = torch.randint(0,17,(n,2),generator=generator)
    rows = offsets[:,0,None,None] + torch.arange(h)[None,:,None]
    cols = offsets[:,1,None,None] + torch.arange(w)[None,None,:]
    batch = torch.arange(n)[:,None,None]
    inputs = padded.permute(0,2,3,1)[batch,rows,cols].permute(0,3,1,2).contiguous()
    flip = torch.rand(n,generator=generator) < .5
    inputs[flip] = inputs[flip].flip(-1)
    brightness = .8 + .4 * torch.rand(n,1,1,1,generator=generator)
    contrast = .8 + .4 * torch.rand(n,1,1,1,generator=generator)
    channels = .9 + .2 * torch.rand(n,3,1,1,generator=generator)
    center = inputs.mean((-2,-1),keepdim=True)
    inputs = (((inputs-center)*contrast+center)*brightness*channels).clamp(0,1)
    mean=inputs.new_tensor((.485,.456,.406))[None,:,None,None]
    std=inputs.new_tensor((.229,.224,.225))[None,:,None,None]
    return ((inputs-mean)/std).contiguous()
