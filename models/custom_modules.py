from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torchvision.ops as tvops


def _autopad(k: int, p: int | None = None) -> int:
    return k // 2 if p is None else p


class EMA(nn.Module):
    """Efficient Multi-scale Attention (EMA).

    Lightweight attention that aggregates spatial context along H and W
    with group-wise channel interaction.

    parse_model calls this as EMA(channels) where `channels` comes from the
    YAML and may not match the actual (width-scaled) input channels.
    Conv layers are therefore built lazily on the first forward pass.
    """

    def __init__(self, channels: int = 0, factor: int = 8):
        super().__init__()
        self._factor = factor
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.sigmoid = nn.Sigmoid()
        # conv1/conv2 are built lazily once the actual channel count is known.
        self.conv1 = None
        self.conv2 = None

    def _build(self, c: int, device: torch.device) -> None:
        factor = self._factor if c % self._factor == 0 else 1
        self.groups = factor
        self.group_channels = c // factor
        self.conv1 = nn.Conv2d(c, c, kernel_size=1, bias=True).to(device)
        self.conv2 = nn.Conv2d(c, c, kernel_size=1, bias=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv1 is None:
            self._build(x.shape[1], x.device)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        a_h = self.conv1(x_h)
        a_w = self.conv2(x_w).permute(0, 1, 3, 2)
        attn = self.sigmoid(a_h + a_w)
        return x * attn


class DCNv3(nn.Module):
    """Deformable Convolution v3 using torchvision.ops.deform_conv2d.

    Offset and mask generators share the same stride as the main conv so
    their output spatial size matches the expected deform_conv2d input.
    Offsets are zero-initialised so the module starts as a regular Conv,
    giving stable early training before the offsets learn to be useful.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1,
                 p: int | None = None, g: int = 1, act: bool = True):
        super().__init__()
        p = _autopad(k, p)
        self.k, self.s, self.p, self.g = k, s, p, g

        # Lightweight offset and mask predictors (stride=s → correct spatial)
        self.offset_conv = nn.Conv2d(c1, 2 * k * k, k, stride=s, padding=p, bias=True)
        self.mask_conv   = nn.Conv2d(c1, k * k,     k, stride=s, padding=p, bias=True)

        # Main conv weight; bias absorbed by BN
        self.weight = nn.Parameter(torch.empty(c2, c1 // g, k, k))
        self.bn  = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

        # Zero-init offsets → identity start (same as regular Conv)
        # Zero-init mask bias → sigmoid(0) = 0.5 (uniform modulation)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask   = torch.sigmoid(self.mask_conv(x))
        out = tvops.deform_conv2d(
            x, offset, self.weight, mask=mask,
            stride=self.s, padding=self.p, groups=self.g,
        )
        return self.act(self.bn(out))


class WeightedSum(nn.Module):
    """BiFPN-style learnable weighted sum (fast-normalise variant).

    Fuses N feature maps of identical shape into one without changing
    channels — a drop-in replacement for Concat followed by a channel-
    reduction Conv.  Uses ReLU + normalise (original BiFPN paper) for
    numerical stability.
    """

    def __init__(self, n: int = 2):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n))

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        w = torch.relu(self.weights)
        w = w / (w.sum() + 1e-4)
        return sum(w[i] * x for i, x in enumerate(inputs))


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
