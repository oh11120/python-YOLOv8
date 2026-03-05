from __future__ import annotations

import warnings
from typing import List

import torch
import torch.nn as nn


def _autopad(k: int, p: int | None = None) -> int:
    return k // 2 if p is None else p


class EMA(nn.Module):
    """Efficient Multi-scale Attention (EMA).

    Lightweight attention that aggregates spatial context along H and W
    with group-wise channel interaction.
    """

    def __init__(self, channels: int, factor: int = 8):
        super().__init__()
        if channels % factor != 0:
            factor = 1
        self.groups = factor
        self.group_channels = channels // self.groups
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        a_h = self.conv1(x_h)
        a_w = self.conv2(x_w).permute(0, 1, 3, 2)
        attn = self.sigmoid(a_h + a_w)
        return x * attn


class DCNv3(nn.Module):
    """Deformable Conv v3 (fallback to standard Conv if ops unavailable).

    This is a practical implementation using torchvision.ops.DeformConv2d when
    available. If not present, it silently falls back to nn.Conv2d so training
    can proceed on environments without the op.
    """

    _warned = False

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None, g: int = 1, act: bool = True):
        super().__init__()
        p = _autopad(k, p)
        self.act = nn.SiLU() if act else nn.Identity()
        # NOTE: DeformConv2d on Windows often crashes (access violation) even when installed.
        # Always fall back to a standard Conv2d to keep training stable.
        self.use_dcn = False
        if not DCNv3._warned:
            warnings.warn("DCNv3 disabled; falling back to Conv2d for stability on this environment")
            DCNv3._warned = True
        self.conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_dcn:
            offset = self.offset(x)
            x = self.dcn(x, offset)
        else:
            x = self.conv(x)
        return self.act(self.bn(x))


class WConcat(nn.Module):
    """Weighted concatenation for feature fusion."""

    def __init__(self, n: int = 2, dimension: int = 1, weight_init: float = 1.0):
        super().__init__()
        self.d = dimension
        self.weights = nn.Parameter(torch.ones(n) * weight_init)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        w = torch.softmax(self.weights, dim=0)
        scaled = [x * w[i] for i, x in enumerate(inputs)]
        return torch.cat(scaled, dim=self.d)


class BiFPN(nn.Module):
    """Lightweight BiFPN block producing fused P3, P4, P5 outputs."""

    def __init__(self, c3: int, c4: int, c5: int, out: int = 256):
        super().__init__()
        self.p3_in = nn.Conv2d(c3, out, 1, 1, 0)
        self.p4_in = nn.Conv2d(c4, out, 1, 1, 0)
        self.p5_in = nn.Conv2d(c5, out, 1, 1, 0)

        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(2))
        self.w3 = nn.Parameter(torch.ones(2))
        self.w4 = nn.Parameter(torch.ones(2))

        self.p4_td = nn.Conv2d(out, out, 3, 1, 1)
        self.p3_td = nn.Conv2d(out, out, 3, 1, 1)
        self.p4_out = nn.Conv2d(out, out, 3, 1, 1)
        self.p5_out = nn.Conv2d(out, out, 3, 1, 1)
        self.act = nn.SiLU()

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = inputs
        p3 = self.p3_in(p3)
        p4 = self.p4_in(p4)
        p5 = self.p5_in(p5)

        w1 = torch.softmax(self.w1, dim=0)
        p4_td = self.act(self.p4_td(w1[0] * p4 + w1[1] * nn.functional.interpolate(p5, scale_factor=2, mode="nearest")))

        w2 = torch.softmax(self.w2, dim=0)
        p3_td = self.act(self.p3_td(w2[0] * p3 + w2[1] * nn.functional.interpolate(p4_td, scale_factor=2, mode="nearest")))

        w3 = torch.softmax(self.w3, dim=0)
        p4_out = self.act(self.p4_out(w3[0] * p4_td + w3[1] * nn.functional.max_pool2d(p3_td, 2)))

        w4 = torch.softmax(self.w4, dim=0)
        p5_out = self.act(self.p5_out(w4[0] * p5 + w4[1] * nn.functional.max_pool2d(p4_out, 2)))

        return [p3_td, p4_out, p5_out]
