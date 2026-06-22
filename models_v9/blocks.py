"""Spatial building blocks for StereoSR V9 (DECA-SR).

Self-contained (does not import from earlier model packages so V9 cannot be
broken by edits elsewhere). Primitives LayerNorm2d / SimpleGate / TailUpsample /
icnr_init are the V6-validated versions, copied verbatim. The NEW piece is
``RepLKNAFBlock``: a NAFBlock whose 3x3 depthwise conv is replaced by a
structurally-reparameterizable large-kernel depthwise conv (train as a
multi-branch {LK, 5, 3, identity} sum; merge to a single LK depthwise conv for
inference at zero added deploy cost). Large-kernel intra-view receptive field is
the cheapest way to free parameter budget for the cross-view fusion (CVHSSR
lesson); reparameterization gives the receptive field + edge-oriented gradients
for free at inference (ECBSR / RepLKNet / RepSR).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for ``[B, C, H, W]`` tensors."""

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
    """NAFNet SimpleGate: split channel-wise into (a, b), return ``a * b``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class RepLargeKernelDW(nn.Module):
    """Reparameterizable large-kernel depthwise conv.

    Training: parallel depthwise branches {LK x LK, then each smaller kernel}
    plus an identity branch, summed. Inference (after ``reparametrize()``): a
    single ``LK x LK`` depthwise conv with the algebraically-merged kernel — bit
    -for-bit equivalent, zero added deploy params vs one large-kernel DW conv.

    BN-free by design so the merge is an exact kernel-pad-and-add (no folding).
    """

    def __init__(self, channels: int, large_kernel: int = 9, small_kernels=(5, 3)):
        super().__init__()
        if large_kernel % 2 == 0:
            raise ValueError(f"large_kernel must be odd (got {large_kernel})")
        self.channels = channels
        self.lk = large_kernel
        self.small_kernels = tuple(k for k in small_kernels if k < large_kernel)

        pad = large_kernel // 2
        self.lk_conv = nn.Conv2d(channels, channels, large_kernel, padding=pad, groups=channels)
        self.small_convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels) for k in self.small_kernels]
        )
        # Merged inference conv (created lazily by reparametrize()).
        self.merged: nn.Conv2d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged is not None:
            return self.merged(x)
        out = self.lk_conv(x)
        for conv in self.small_convs:
            out = out + conv(x)
        out = out + x  # identity branch
        return out

    @torch.no_grad()
    def reparametrize(self) -> None:
        """Fuse all branches into a single ``LK x LK`` depthwise conv."""
        lk, c = self.lk, self.channels
        weight = self.lk_conv.weight.detach().clone()             # [C, 1, lk, lk]
        bias = self.lk_conv.bias.detach().clone() if self.lk_conv.bias is not None else torch.zeros(c)

        def _pad_to_lk(w: torch.Tensor) -> torch.Tensor:
            k = w.shape[-1]
            p = (lk - k) // 2
            return F.pad(w, (p, p, p, p))

        for conv in self.small_convs:
            weight = weight + _pad_to_lk(conv.weight.detach())
            if conv.bias is not None:
                bias = bias + conv.bias.detach()

        # Identity branch = unit impulse at the kernel center.
        identity = torch.zeros(c, 1, lk, lk, device=weight.device, dtype=weight.dtype)
        identity[:, 0, lk // 2, lk // 2] = 1.0
        weight = weight + identity

        merged = nn.Conv2d(c, c, lk, padding=lk // 2, groups=c).to(weight.device, weight.dtype)
        merged.weight.copy_(weight)
        merged.bias.copy_(bias)
        self.merged = merged
        # Free the training-time branches so the deployed module is a single DW conv.
        self.lk_conv = None
        self.small_convs = nn.ModuleList()


class RepLKNAFBlock(nn.Module):
    """NAFBlock with a reparameterizable large-kernel depthwise conv.

    Identical to NAFNet's NAFBlock except the single 3x3 depthwise conv on the
    expanded features is replaced by :class:`RepLargeKernelDW`. Zero-init beta/
    gamma residual scales so the block starts as identity.
    """

    def __init__(
        self,
        channels: int,
        dw_expansion: float = 2.0,
        ffn_expansion: float = 2.0,
        large_kernel: int = 9,
    ):
        super().__init__()
        c = channels
        dw_c = int(c * dw_expansion)
        ffn_c = int(c * ffn_expansion)

        # Spatial path
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, 2 * dw_c, 1)
        self.dwconv = RepLargeKernelDW(2 * dw_c, large_kernel=large_kernel)
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

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.norm1(x))
        y = self.dwconv(y)
        y = self.gate1(y)
        y = self.sca(y) * y
        y = self.conv2(y)
        x = x + self.beta * y

        y = self.conv3(self.norm2(x))
        y = self.gate2(y)
        y = self.conv4(y)
        x = x + self.gamma * y
        return x


def icnr_init(conv: nn.Conv2d, scale: int) -> None:
    """ICNR initialization for sub-pixel conv (Aitken et al., 2017)."""
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
    """Single PixelShuffle upsample at the tail (ICNR-init, scaled by init_scale)."""

    def __init__(self, in_channels: int, out_channels: int, scale: int, init_scale: float = 0.01):
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
