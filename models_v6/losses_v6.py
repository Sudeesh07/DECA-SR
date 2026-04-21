"""Losses for V6.

* **Pixel reconstruction** (Charbonnier): robust main supervision.
* **Stereo photometric consistency**: view-symmetric flip-equivariance loss —
  cheap, disparity-free, effective for stereo SR (see iPASSR).
* **Spectral consistency (novel)**: L1 on the magnitude spectra of SR_L and
  SR_R. Derived directly from Proposition 1 in ARCHITECTURE.md; penalises
  deviations from magnitude-spectrum equality. Small weight so occluded
  regions (where the assumption is violated) are not over-regularised.
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
    """Disparity-free stereo consistency: encourage symmetric HR reconstructions.

    Uses a small search-window photometric L1 between SR_L and horizontal shifts
    of SR_R. Cheap approximation to warped consistency when no disparity is
    predicted. Kept as a lightweight prior; CVSFM handles the heavy lifting.
    """

    def __init__(self, max_shift: int = 48):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, sr_l: torch.Tensor, sr_r: torch.Tensor) -> torch.Tensor:
        # Minimum over a handful of candidate shifts — approximates taking the
        # best match per-pixel without a disparity network.
        shifts = (0, self.max_shift // 4, self.max_shift // 2, self.max_shift)
        best = None
        for s in shifts:
            if s == 0:
                diff = (sr_l - sr_r).abs()
            else:
                diff = (sr_l[..., :, s:] - sr_r[..., :, :-s]).abs()
                diff = F.pad(diff, (0, s, 0, 0))  # pad right edge
            diff = diff.mean(dim=1, keepdim=True)  # average over RGB
            best = diff if best is None else torch.minimum(best, diff)
        return best.mean()


class SpectralConsistencyLoss(nn.Module):
    """L1 on magnitude spectra of SR_L and SR_R. Novel stereo loss derived from
    the magnitude-invariance property (Proposition 1, ARCHITECTURE.md)."""

    def __init__(self, hf_emphasis: bool = True):
        super().__init__()
        self.hf_emphasis = hf_emphasis

    def forward(self, sr_l: torch.Tensor, sr_r: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            Fl = torch.fft.rfft2(sr_l.float(), norm="ortho")
            Fr = torch.fft.rfft2(sr_r.float(), norm="ortho")
            diff = (Fl.abs() - Fr.abs()).abs()  # [B, C, H, W//2+1]
            if self.hf_emphasis:
                # Radial high-pass weight. Low frequencies encode global lighting
                # / illumination differences which legitimately differ across
                # views; high frequencies carry shared textural content.
                H, Wf = diff.shape[-2:]
                fy = torch.fft.fftfreq(H, device=diff.device).abs()    # [H]
                fx = torch.linspace(0, 0.5, Wf, device=diff.device)    # [W//2+1]
                # Normalise to [0, 1] radial distance.
                radial = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()
                radial = radial / radial.max().clamp_min(1e-8)
                diff = diff * radial[None, None]
        return diff.mean()


class StereoSRV6Loss(nn.Module):
    """Composite loss for V6."""

    def __init__(
        self,
        charbonnier_eps: float = 1e-3,
        stereo_weight: float = 0.02,
        spectral_weight: float = 0.01,
        stereo_max_shift: int = 48,
    ):
        super().__init__()
        self.pixel = CharbonnierLoss(charbonnier_eps)
        self.stereo_weight = stereo_weight
        self.spectral_weight = spectral_weight
        self.stereo = StereoPhotometricConsistency(stereo_max_shift) if stereo_weight > 0 else None
        self.spectral = SpectralConsistencyLoss(hf_emphasis=True) if spectral_weight > 0 else None

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
        total = pixel
        losses["pixel"] = pixel

        if self.stereo is not None:
            stereo = self.stereo(sr_left, sr_right)
            total = total + self.stereo_weight * stereo
            losses["stereo"] = stereo

        if self.spectral is not None:
            spec = self.spectral(sr_left, sr_right)
            total = total + self.spectral_weight * spec
            losses["spectral"] = spec

        losses["total"] = total
        return total, losses
