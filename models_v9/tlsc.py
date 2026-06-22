"""Test-time Local Statistics Correction (TLSC) for V9.

Swaps each trunk block's SCA ``AdaptiveAvgPool2d(1)`` for a local windowed pool
sized to the training patch, so the channel statistics seen at full-image
inference match training. Targets any module exposing a ``.sca`` Sequential
whose first element is an AdaptiveAvgPool2d (covers RepLKNAFBlock). Call AFTER
loading weights, BEFORE eval. Idempotent.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAvgPool2d(nn.Module):
    """Local-window average pool with output size equal to input size."""

    def __init__(self, base_size):
        super().__init__()
        if isinstance(base_size, int):
            base_size = (base_size, base_size)
        self.base_size = tuple(int(s) for s in base_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        kh = min(self.base_size[0], h)
        kw = min(self.base_size[1], w)
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


def _iter_sca(model: nn.Module):
    for module in model.modules():
        sca = getattr(module, "sca", None)
        if isinstance(sca, nn.Sequential) and len(sca) > 0:
            yield sca


def enable_tlsc(model: nn.Module, base_size) -> int:
    if isinstance(base_size, int):
        base_size = (base_size, base_size)
    replaced = 0
    for sca in _iter_sca(model):
        if isinstance(sca[0], nn.AdaptiveAvgPool2d):
            sca[0] = LocalAvgPool2d(base_size)
            replaced += 1
    return replaced


def disable_tlsc(model: nn.Module) -> int:
    reverted = 0
    for sca in _iter_sca(model):
        if isinstance(sca[0], LocalAvgPool2d):
            sca[0] = nn.AdaptiveAvgPool2d(1)
            reverted += 1
    return reverted
