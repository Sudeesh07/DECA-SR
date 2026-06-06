"""Losses for V8.

V8 uses Charbonnier pixel loss + iPASSR-style stereo photometric consistency,
nothing else. Specifically: V8 deliberately drops the spectral consistency
loss (derived from V1's falsified magnitude-invariance theorem) and the
disparity-aware loss family (V4/V7 showed the pixel-disparity prior does not
generalize across train/eval scales). The architecture novelty lives in
CT-CVA, not in the loss.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth-L1-like loss: ``sqrt((pred − target)² + ε²)``.

    Robust to outliers (vs L2) but smooth at 0 (unlike L1). The default
    ε = 1e-3 matches NAFSSR / iPASSR / SwinFSR practice.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class StereoPhotometricConsistency(nn.Module):
    """iPASSR-style disparity-free stereo consistency.

    For each candidate horizontal shift in {0, max_shift/4, max_shift/2,
    max_shift}, compute L1 between SR_L and shifted SR_R. Take the per-pixel
    minimum across shifts. Approximates "best match" over a small disparity
    range without an explicit disparity prediction.
    """

    def __init__(self, max_shift: int = 48):
        super().__init__()
        self.max_shift = int(max_shift)

    def forward(self, sr_l: torch.Tensor, sr_r: torch.Tensor) -> torch.Tensor:
        shifts = (0, self.max_shift // 4, self.max_shift // 2, self.max_shift)
        best = None
        for s in shifts:
            if s == 0:
                diff = (sr_l - sr_r).abs()
            else:
                diff = (sr_l[..., :, s:] - sr_r[..., :, :-s]).abs()
                diff = F.pad(diff, (0, s, 0, 0))
            diff = diff.mean(dim=1, keepdim=True)  # avg over RGB
            best = diff if best is None else torch.minimum(best, diff)
        return best.mean()


class StereoSRV8Loss(nn.Module):
    """Composite loss for V8: pixel + stereo_weight · stereo."""

    def __init__(
        self,
        charbonnier_eps: float = 1e-3,
        stereo_weight: float = 0.02,
        stereo_max_shift: int = 48,
    ):
        super().__init__()
        self.pixel = CharbonnierLoss(charbonnier_eps)
        self.stereo_weight = float(stereo_weight)
        self.stereo = (
            StereoPhotometricConsistency(stereo_max_shift) if stereo_weight > 0 else None
        )

    def forward(
        self,
        sr_left: torch.Tensor,
        sr_right: torch.Tensor,
        hr_left: torch.Tensor,
        hr_right: torch.Tensor,
        aux: dict | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses: Dict[str, torch.Tensor] = {}
        pixel = 0.5 * (self.pixel(sr_left, hr_left) + self.pixel(sr_right, hr_right))
        losses["pixel"] = pixel
        total = pixel
        if self.stereo is not None:
            stereo = self.stereo(sr_left, sr_right)
            total = total + self.stereo_weight * stereo
            losses["stereo"] = stereo
        losses["total"] = total
        return total, losses
