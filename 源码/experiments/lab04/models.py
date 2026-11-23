"""A CPU-sized ResNet12 and differentiable, full-network MAML.

Twelve 3x3 convolutions in four residual blocks. 1x1 shortcut projections are
not counted, following common ResNet12 naming. GroupNorm avoids sharing query
batch statistics with the support set. Widths are recorded in every result.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        layers = []
        for index in range(3):
            layers.extend([
                nn.Conv2d(input_channels if index == 0 else output_channels,
                          output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(4, output_channels),
            ])
            if index != 2:
                layers.append(nn.LeakyReLU(0.1, inplace=False))
        self.convolutions = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.GroupNorm(4, output_channels),
        )

    def forward(self, inputs):
        output = F.leaky_relu(self.convolutions(inputs) + self.shortcut(inputs),
                              negative_slope=0.1)
        return F.max_pool2d(output, 2)


class ResNet12(nn.Module):
    def __init__(self, classes: int = 64, widths=(8, 16, 24, 32)):
        super().__init__()
        self.widths = tuple(int(value) for value in widths)
        if len(self.widths) != 4 or any(value % 4 for value in self.widths):
            raise ValueError("Four widths divisible by four are required")
        self.encoder = nn.Sequential(*[
            ResidualBlock(3 if index == 0 else self.widths[index - 1], width)
            for index, width in enumerate(self.widths)
        ])
        self.classifier = nn.Linear(self.widths[-1], classes)

    def features(self, inputs):
        return self.encoder(inputs).mean(dim=(-2, -1))

    def forward(self, inputs, return_features=False):
        features = self.features(inputs)
        scores = self.classifier(features)
        return (scores, features) if return_features else scores


def maml_adapt(model, support, labels, *, steps=1, learning_rate=0.1,
               second_order=True):
    """Adapt every named parameter; never consume query images or labels.

    second_order=True preserves the inner gradient graph for exact MAML.
    False is an explicit first-order approximation, reported as FOMAML.
    """
    parameters = OrderedDict(model.named_parameters())
    for _ in range(steps):
        logits = torch.func.functional_call(model, parameters, (support,))
        loss = F.cross_entropy(logits, labels)
        gradients = torch.autograd.grad(loss, tuple(parameters.values()),
                                        create_graph=second_order)
        parameters = OrderedDict(
            (name, weight - learning_rate * gradient)
            for (name, weight), gradient in zip(parameters.items(), gradients)
        )
    return parameters


def maml_logits(model, support, labels, query, *, steps=3, learning_rate=0.1):
    # Test-time adaptation needs gradients on the support set, but no
    # second-order graph and no gradients on the held-out query set.
    with torch.enable_grad():
        parameters = maml_adapt(model, support, labels, steps=steps,
                                learning_rate=learning_rate, second_order=False)
    with torch.no_grad():
        return torch.func.functional_call(model, parameters, (query,))


def distribution_alignment(first, second):
    """Align two augmentations using feature mean and variance moments.

    This is a minibatch regularizer over training images only, with no use of
    validation/test labels or images. It discourages view-dependent drift.
    """
    mean_loss = (first.mean(0) - second.mean(0)).square().mean()
    variance_loss = (first.var(0, unbiased=False) -
                     second.var(0, unbiased=False)).square().mean()
    return mean_loss + variance_loss
