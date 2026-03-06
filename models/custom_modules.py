from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def _autopad(k: int, p: int | None = None) -> int:
    return k // 2 if p is None else p


class EMA(nn.Module):
    """Efficient Multi-scale Attention using spatial pooling + grouped 1x1 convolutions.

    Parameters are built in __init__ (not lazily) so they are captured by the
    optimizer from the very first training step.  Grouped convolutions implement
    the multi-scale channel interaction described in the original EMA paper.
    """

    def __init__(self, channels: int, factor: int = 8):
        super().__init__()
        factor = factor if channels % factor == 0 else 1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, channels, 1, groups=factor, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 1, groups=factor, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        attn = self.sigmoid(self.conv1(x_h) + self.conv2(x_w).permute(0, 1, 3, 2))
        return x * attn


class DCNv3(nn.Module):
    """Depthwise Separable Convolution with BN+SiLU after both DW and PW stages.

    Replaces deformable conv (deform_conv2d unsupported on sm_120/RTX5090).
    DW-sep conv achieves similar multi-scale efficiency with fewer params:
      - depthwise conv captures spatial context channel-independently
      - pointwise conv mixes channels
    BN is applied after both stages for training stability.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1,
                 p: int | None = None, g: int = 1, act: bool = True):
        super().__init__()
        p = _autopad(k, p)
        self.dw = nn.Conv2d(c1, c1, k, stride=s, padding=p, groups=c1, bias=False)
        self.bn_dw = nn.BatchNorm2d(c1)
        self.act_dw = nn.SiLU()
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.act_dw(self.bn_dw(self.dw(x))))))


class WeightedSum(nn.Module):
    def __init__(self, n: int = 2):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n))

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        w = torch.relu(self.weights)
        w = w / (w.sum() + 1e-4)
        return sum(w[i] * x for i, x in enumerate(inputs))


class WConcat(nn.Module):
    def __init__(self, n: int = 2, dimension: int = 1, weight_init: float = 1.0):
        super().__init__()
        self.d = dimension
        self.weights = nn.Parameter(torch.ones(n) * weight_init)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        w = torch.softmax(self.weights, dim=0)
        scaled = [x * w[i] for i, x in enumerate(inputs)]
        return torch.cat(scaled, dim=self.d)
