"""Spatial building blocks for StereoSR V6 (SpecFuse).

All computation stays at LR resolution in the trunk; the only upsample is a
single PixelShuffle at the tail. This avoids the cascaded-upsample checkerboard
problem that plagues U-Net-style SR designs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for 2D feature maps."""

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
    """SimpleGate from NAFNet: x1 * x2 where x = [x1, x2] split along channels."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """Non-linear activation-free block from NAFNet (Chen et al., ECCV 2022).

    Backbone of NAFSSR and NAFNet; used here as the per-view spatial processor.
    Uses two gated residual paths (spatial-mixing + channel FFN). SimpleGate
    replaces activations; SCA (simplified channel attention) gates per-channel.
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

        # Zero-init residual scales for stable training (NAFNet trick)
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

    Without this, PixelShuffle produces a checkerboard pattern at init because
    each of the scale×scale sub-pixels in a block receives independently
    random weights. ICNR shares init across sub-pixels, producing a smooth
    nearest-neighbour upscale at init.
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
    """Single PixelShuffle upsample at the network tail.

    Initialisation
    --------------
    We apply ICNR (Aitken et al., 2017) — sub-pixel-sharing init that avoids
    checkerboard — and then scale the weights by ``init_scale`` (default 0.01).

    The small scale is critical for networks with a **global bicubic residual**
    (``SR = tail(trunk(LR)) + bicubic(LR)``). Without it, the tail output has
    std ≈ 1.4 per pixel at init, which dominates the bicubic baseline and
    produces nonsense initial SR (PSNR ~8 dB, pixel loss ~10). With it, the
    initial residual is ~0.014 per pixel, so ``SR_init ≈ bicubic(LR)`` and the
    network trains *up* from the bicubic baseline (~24 dB at step 0).

    Zero-init the weights outright would break gradient flow to the trunk
    (∂loss/∂trunk = tail.weightᵀ · loss_grad vanishes). A small non-zero scale
    keeps gradients flowing while starting close to the identity-to-bicubic map.
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
