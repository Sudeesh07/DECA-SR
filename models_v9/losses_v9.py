"""Losses for V9 (DECA-SR).

Charbonnier pixel + iPASSR-style stereo photometric consistency + a per-view
Focal Frequency Loss (intra-view SR-vs-own-HR, NOT the falsified cross-view
|F_L|=|F_R| term). A terminal phase switches the pixel term to pure MSE and
drops the auxiliary terms, aligning the objective with the PSNR metric for the
last iterations (the HTCAN +0.04 dB lever).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class StereoPhotometricConsistency(nn.Module):
    """iPASSR-style disparity-free stereo consistency (min over candidate shifts)."""

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
            diff = diff.mean(dim=1, keepdim=True)
            best = diff if best is None else torch.minimum(best, diff)
        return best.mean()


class FocalFrequencyLoss(nn.Module):
    """Per-image focal frequency loss (Jiang et al., ICCV 2021).

    Penalizes the squared spectral distance, focally re-weighted toward the
    hard (large-error) frequencies. Computed in fp32, orthonormal FFT for
    scale-stability across image sizes.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            fp = torch.fft.rfft2(pred.float(), norm="ortho")
            ft = torch.fft.rfft2(target.float(), norm="ortho")
            d2 = (fp - ft).abs() ** 2                       # [B,C,H,Wf]
            w = d2.detach() ** self.alpha
            w = w / (w.amax(dim=(-2, -1), keepdim=True) + 1e-8)
            return (w * d2).mean()


class StereoSRV9Loss(nn.Module):
    def __init__(
        self,
        charbonnier_eps: float = 1e-3,
        stereo_weight: float = 0.02,
        stereo_max_shift: int = 48,
        ffl_weight: float = 0.1,
    ):
        super().__init__()
        self.pixel = CharbonnierLoss(charbonnier_eps)
        self.stereo_weight = float(stereo_weight)
        self.stereo = StereoPhotometricConsistency(stereo_max_shift) if stereo_weight > 0 else None
        self.ffl_weight = float(ffl_weight)
        self.ffl = FocalFrequencyLoss() if ffl_weight > 0 else None
        self.terminal = False

    def set_terminal(self, on: bool = True) -> None:
        """Terminal phase: pixel -> MSE, drop stereo/FFL (align with PSNR)."""
        self.terminal = bool(on)

    def forward(self, sr_left, sr_right, hr_left, hr_right, aux: dict | None = None):
        losses: Dict[str, torch.Tensor] = {}
        if self.terminal:
            pixel = 0.5 * (F.mse_loss(sr_left, hr_left) + F.mse_loss(sr_right, hr_right))
            losses["pixel"] = pixel
            losses["total"] = pixel
            return pixel, losses

        pixel = 0.5 * (self.pixel(sr_left, hr_left) + self.pixel(sr_right, hr_right))
        losses["pixel"] = pixel
        total = pixel
        if self.stereo is not None:
            stereo = self.stereo(sr_left, sr_right)
            total = total + self.stereo_weight * stereo
            losses["stereo"] = stereo
        if self.ffl is not None:
            ffl = 0.5 * (self.ffl(sr_left, hr_left) + self.ffl(sr_right, hr_right))
            total = total + self.ffl_weight * ffl
            losses["ffl"] = ffl
        losses["total"] = total
        return total, losses
