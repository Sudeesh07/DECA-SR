"""Test-time Local Statistics Correction (TLSC) — reused from V6/V7 unchanged.

Implements the inference-time correction from Chu et al., "Revisiting Global
Statistics Aggregation for Improving Image Restoration" (ECCV-W 2022).

Key idea: at training the SCA branch's ``AdaptiveAvgPool2d(1)`` pools over the
training-patch shape; at inference on full LR images that pool covers ~10×+ the
area, and the downstream 1×1 conv breaks because it never saw those channel
statistics. Replace the global pool with a local windowed pool sized to the
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
        # When input is smaller than the training window, true global pool is correct.
        if kh == h and kw == w:
            return x.mean(dim=(-2, -1), keepdim=True)
        pad_top = (kh - 1) // 2
        pad_bottom = kh - 1 - pad_top
        pad_left = (kw - 1) // 2
        pad_right = kw - 1 - pad_left
        x_pad = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], mode="reflect")
        return F.avg_pool2d(x_pad, kernel_size=(kh, kw), stride=1)

    def extra_repr(self) -> str:
        return f"base_size={self.base_size}"


def enable_tlsc(model: nn.Module, base_size) -> int:
    """Swap ``AdaptiveAvgPool2d(1)`` for ``LocalAvgPool2d`` in every NAFBlock SCA.
    Returns count replaced. Idempotent. Call AFTER loading weights, BEFORE eval.
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
    """Revert TLSC: swap each LocalAvgPool2d back to AdaptiveAvgPool2d(1)."""
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
