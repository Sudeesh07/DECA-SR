"""Test-time Local Statistics Correction (TLSC) — reused from V6 unchanged.

Implements the inference-time correction from Chu et al., "Revisiting Global
Statistics Aggregation for Improving Image Restoration" (ECCV-W 2022), as
applied to NAFSSR. Same code path V6 used; documented at length in V6_REPORT.md
§5.2.

Key idea: at training the SCA branch's ``AdaptiveAvgPool2d(1)`` pools over the
training-patch shape (V7 default: 48×144); at inference on full LR images
(often 200×500+) that pool covers 10× the area, and the downstream 1×1 conv
breaks. Replace the global pool with a local windowed pool sized to the
training patch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAvgPool2d(nn.Module):
    """Local-window average pool with output size equal to input size.

    Drop-in replacement for ``nn.AdaptiveAvgPool2d(1)`` at inference under TLSC.
    Reflection padding preserves output dimensions.
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

        # When input is smaller than the training window, true global pool is
        # the correct behaviour — fall through.
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
    NAFBlock-style SCA branch in ``model``. Returns count replaced.

    ``base_size`` should equal ``(patch_h, patch_w)`` from training.
    Applied AFTER loading weights, BEFORE ``model.eval()``. Idempotent.
    """
    from .blocks import NAFBlock

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
    """Revert TLSC: swap each :class:`LocalAvgPool2d` back to
    ``nn.AdaptiveAvgPool2d(1)``. Returns count reverted."""
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
