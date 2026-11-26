"""Inference-only adapter for the cited FEAT ResNet12 research checkpoint.

This is a separate external reference, not the course's self-trained light
backbone. BatchNorm is frozen in eval mode; no query batch statistics adapt it.
Architecture/state names follow Sha-Lab/FEAT (MIT), attribution bundled nearby.
Only its encoder is used, NOT the FEAT attention adaptation module. The input
images here are the existing 84px mini-ImageNet images, and tensor resize may
differ slightly from upstream PIL resize; no exact-paper-score claim is made.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class ReferenceBlock(nn.Module):
    def __init__(self,input_channels,channels):
        super().__init__()
        self.conv1=nn.Conv2d(input_channels,channels,3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(channels)
        self.conv2=nn.Conv2d(channels,channels,3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(channels)
        self.conv3=nn.Conv2d(channels,channels,3,padding=1,bias=False)
        self.bn3=nn.BatchNorm2d(channels)
        self.downsample=nn.Sequential(nn.Conv2d(input_channels,channels,1,bias=False),nn.BatchNorm2d(channels))

    def forward(self,x):
        residual=self.downsample(x)
        out=F.leaky_relu(self.bn1(self.conv1(x)),.1)
        out=F.leaky_relu(self.bn2(self.conv2(out)),.1)
        out=self.bn3(self.conv3(out))
        return F.max_pool2d(F.leaky_relu(out+residual,.1),2)


class ReferenceResNet12(nn.Module):
    def __init__(self):
        super().__init__()
        for index,(input_channels,channels) in enumerate(zip((3,64,160,320),(64,160,320,640)),1):
            setattr(self,f"layer{index}",nn.Sequential(ReferenceBlock(input_channels,channels)))

    def forward(self,x):
        for name in ('layer1','layer2','layer3','layer4'):x=getattr(self,name)(x)
        return x.mean((-2,-1))


def load_reference(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    state=checkpoint.get('params',checkpoint.get('state_dict',checkpoint))
    prefix='encoder.'
    encoder={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
    if not encoder:raise ValueError('Unexpected FEAT encoder state keys')
    model=ReferenceResNet12();model.load_state_dict(encoder,strict=True);model.eval()
    return model


def normalized_reference(images):
    # Author preprocessing: resize to 92 then center crop 84, dataset-specific
    # train statistics, no semantic labels or external ImageNet classifier.
    x=torch.from_numpy(images).permute(0,3,1,2).float()/255
    x=F.interpolate(x,size=(92,92),mode='bilinear',align_corners=False,antialias=True)[:,:,4:88,4:88]
    mean=x.new_tensor((120.39586422,115.59361427,104.54012653))[None,:,None,None]/255
    std=x.new_tensor((70.68188272,68.27635443,72.54505529))[None,:,None,None]/255
    return (x-mean)/std
