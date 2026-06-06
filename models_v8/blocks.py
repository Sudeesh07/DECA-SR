"""Spatial building blocks for StereoSR V8.

Reused unchanged from V6/V7. These are the load-bearing pieces that V6 A1
validated empirically (23.70 dB val / 23.81 dB test on Flickr1024 ×4).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for 2D feature maps.

    Operates directly on ``[B, C, H, W]``. Per-channel learned affine.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """NAFNet's SimpleGate: split tensor channel-wise into (a, b), return ``a*b``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """Non-linear activation-free block from NAFNet (Chen et al., ECCV 2022).

    Per-view spatial processor. β, γ zero-init scalars so the block starts as
    identity. Training lifts them off zero as the SR loss creates gradient
    through the residual paths.
    """

    def __init__(self, channels: int, dw_expansion: float = 2.0, ffn_expansion: float = 2.0):
        super().__init__()
        c = channels
        dw_c = int(c * dw_expansion)
        ffn_c = int(c * ffn_expansion)

        # Spatial path
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, 2 * dw_c, 1)
        self.dwconv = nn.Conv2d(2 * dw_c, 2 * dw_c, 3, padding=1, groups=2 * dw_c)
        self.gate1 = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_c, dw_c, 1),
        )
        self.conv2 = nn.Conv2d(dw_c, c, 1)

        # FFN path
        self.norm2 = LayerNorm2d(c)
        self.conv3 = nn.Conv2d(c, 2 * ffn_c, 1)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_c, c, 1)

        # Zero-init residual scales for stable training (NAFNet trick).
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial block
        y = self.conv1(self.norm1(x))
        y = self.dwconv(y)
        y = self.gate1(y)
        y = self.sca(y) * y
        y = self.conv2(y)
        x = x + self.beta * y

        # FFN block
        y = self.conv3(self.norm2(x))
        y = self.gate2(y)
        y = self.conv4(y)
        x = x + self.gamma * y
        return x


def icnr_init(conv: nn.Conv2d, scale: int) -> None:
    """ICNR initialization for sub-pixel convolution (Aitken et al., 2017).

    Without ICNR, PixelShuffle produces a checkerboard pattern at init.
    ICNR initializes one sub-pixel kernel and repeats it scale² times,
    producing a smooth nearest-neighbour upscale at init.
    """
    out_c, in_c, kh, kw = conv.weight.shape
    sub_c = out_c // (scale ** 2)
    if sub_c * (scale ** 2) != out_c:
        raise ValueError(
            f"ICNR requires out_channels ({out_c}) divisible by scale^2 ({scale ** 2})"
        )
    kernel = torch.empty(sub_c, in_c, kh, kw)
    nn.init.kaiming_normal_(kernel, mode="fan_out", nonlinearity="relu")
    kernel = kernel.repeat_interleave(scale ** 2, dim=0)
    with torch.no_grad():
        conv.weight.copy_(kernel)
        if conv.bias is not None:
            conv.bias.zero_()


class TailUpsample(nn.Module):
    """Single PixelShuffle upsample at the tail.

    ICNR-initialised then scaled by ``init_scale`` (default 0.01) so the
    initial residual is small vs the bicubic baseline.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int,
        init_scale: float = 0.01,
    ):
        super().__init__()
        if scale not in (2, 3, 4):
            raise ValueError(f"scale must be 2, 3, or 4 (got {scale})")
        self.scale = scale
        self.conv = nn.Conv2d(in_channels, out_channels * (scale ** 2), 3, padding=1)
        icnr_init(self.conv, scale=scale)
        with torch.no_grad():
            self.conv.weight.mul_(init_scale)
        nn.init.zeros_(self.conv.bias)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))
