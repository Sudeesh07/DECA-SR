"""Test-time Local Statistics Correction (TLSC).

Implements the inference-time correction from Chu et al., "Revisiting Global
Statistics Aggregation for Improving Image Restoration" (ECCV-W 2022), as
applied to NAFSSR.

Background
----------
During training, the model sees small LR patches (e.g. 30×90). Inside each
NAFBlock the SCA branch performs ``AdaptiveAvgPool2d(1)`` followed by a
1×1 conv — i.e. global average pool over the *training patch*. The
subsequent 1×1 conv is therefore trained to map "stats of a 30×90 patch"
to "channel gating values."

At inference the model receives full LR images (e.g. 200×500). The same
``AdaptiveAvgPool2d(1)`` now pools over ~10× the area, giving channel
statistics with very different scale and variance than the model has ever
seen. The downstream 1×1 conv's mapping breaks, and PSNR suffers.

Fix
---
Replace each SCA's global pool with a **local windowed average pool** sized
to match the training patch. Each spatial position now sees the same
statistical aggregation regime it would have during training, restoring the
distribution the channel-attention conv was trained for. As a side effect
the gating becomes position-dependent (a per-pixel modulation) rather than
a single global scalar.

This is applied only at inference; training is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAvgPool2d(nn.Module):
    """Local-window average pool with output size equal to input size.

    Drop-in replacement for ``nn.AdaptiveAvgPool2d(1)`` at inference under TLSC.
    The kernel matches the training-time feature spatial shape; reflection
    padding preserves output dimensions so downstream layers (1×1 conv,
    elementwise multiply with the residual stream) see the original shape.
    """

    def __init__(self, base_size):
        super().__init__()
        if isinstance(base_size, int):
            base_size = (base_size, base_size)
        self.base_size = tuple(int(s) for s in base_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        kh = min(self.base_size[0], h)
        kw = min(self.base_size[1], w)

        # Fall through to true global pool when the input is smaller than or
        # equal to the training patch (e.g. tiny test images / single-patch
        # paths). The original AdaptiveAvgPool2d(1) is the correct behaviour
        # in that regime.
        if kh == h and kw == w:
            return x.mean(dim=(-2, -1), keepdim=True)

        # Reflection pad so the output shape matches the input shape.
        pad_top = (kh - 1) // 2
        pad_bottom = kh - 1 - pad_top
        pad_left = (kw - 1) // 2
        pad_right = kw - 1 - pad_left
        x_pad = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], mode="reflect")
        return F.avg_pool2d(x_pad, kernel_size=(kh, kw), stride=1)

    def extra_repr(self) -> str:
        return f"base_size={self.base_size}"


def enable_tlsc(model: nn.Module, base_size) -> int:
    """Swap ``AdaptiveAvgPool2d(1)`` for :class:`LocalAvgPool2d` in every
    ``NAFBlock``-style SCA branch in ``model``.

    ``base_size`` is the LR feature-map spatial shape during training — for
    V6 it equals (args.patch_h, args.patch_w) since the trunk preserves
    LR resolution throughout.

    Returns the number of pool layers replaced.
    """
    from .blocks import NAFBlock  # local import to avoid circular

    if isinstance(base_size, int):
        base_size = (base_size, base_size)

    replaced = 0
    for module in model.modules():
        if not isinstance(module, NAFBlock):
            continue
        sca = getattr(module, "sca", None)
        if not isinstance(sca, nn.Sequential):
            continue
        if len(sca) == 0 or not isinstance(sca[0], nn.AdaptiveAvgPool2d):
            continue
        sca[0] = LocalAvgPool2d(base_size)
        replaced += 1
    return replaced


def disable_tlsc(model: nn.Module) -> int:
    """Revert TLSC: replace each :class:`LocalAvgPool2d` back to
    ``nn.AdaptiveAvgPool2d(1)``. Returns the number of layers reverted."""
    from .blocks import NAFBlock

    reverted = 0
    for module in model.modules():
        if not isinstance(module, NAFBlock):
            continue
        sca = getattr(module, "sca", None)
        if not isinstance(sca, nn.Sequential):
            continue
        if len(sca) == 0 or not isinstance(sca[0], LocalAvgPool2d):
            continue
        sca[0] = nn.AdaptiveAvgPool2d(1)
        reverted += 1
    return reverted
