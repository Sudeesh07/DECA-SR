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


class StereoShiftConsistencyLoss(nn.Module):
    """
    Optional, very small stereo regularizer.

    Left OFF by default for benchmark PSNR training.
    """

    def __init__(self, disparities: tuple[int, ...] = (1, 2, 4, 8, 12, 16)):
        super().__init__()
        self.disparities = disparities

    def _one_direction(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        _, _, _, w = src.shape
        losses = []
        for d in self.disparities:
            if d >= w:
                continue
            losses.append((src[:, :, :, d:] - tgt[:, :, :, : w - d]).abs().mean())
        if not losses:
            return src.new_tensor(0.0)
        return torch.stack(losses).mean()

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self._one_direction(left, right) + self._one_direction(right, left))


class StereoSRPSNRLoss(nn.Module):
    def __init__(
        self,
        pixel_loss: str = "charbonnier",
        charbonnier_eps: float = 1e-3,
        mse_weight: float = 0.05,
        stereo_weight: float = 0.0,
    ):
        super().__init__()
        self.pixel_loss_name = pixel_loss
        self.mse_weight = mse_weight
        self.stereo_weight = stereo_weight

        if pixel_loss == "charbonnier":
            self.pixel = CharbonnierLoss(charbonnier_eps)
        elif pixel_loss == "l1":
            self.pixel = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported pixel loss: {pixel_loss}")

        self.stereo = StereoShiftConsistencyLoss()

    def forward(
        self,
        sr_left: torch.Tensor,
        sr_right: torch.Tensor,
        hr_left: torch.Tensor,
        hr_right: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses: Dict[str, torch.Tensor] = {}
        pixel = 0.5 * (self.pixel(sr_left, hr_left) + self.pixel(sr_right, hr_right))
        mse = 0.5 * (F.mse_loss(sr_left, hr_left) + F.mse_loss(sr_right, hr_right))

        total = pixel + self.mse_weight * mse
        losses["pixel"] = pixel
        losses["mse"] = mse

        if self.stereo_weight > 0:
            stereo = self.stereo(sr_left, sr_right)
            total = total + self.stereo_weight * stereo
            losses["stereo"] = stereo

        losses["total"] = total
        return total, losses
