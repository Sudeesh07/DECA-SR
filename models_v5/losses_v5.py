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


# ---------------------------------------------------------------------------
# FIX 3: Disparity-warped stereo consistency (replaces naive shift loss)
# ---------------------------------------------------------------------------

def _warp_with_disparity(
    src: torch.Tensor,
    disparity: torch.Tensor,
    reverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp src image using disparity map. Returns (warped, valid_mask).

    disparity: [B, 1, H, W] in pixel units at the disparity map resolution.
    src: [B, C, H_sr, W_sr] at SR resolution.
    The disparity is upsampled to SR resolution and scaled accordingly.
    """
    b, c, h_sr, w_sr = src.shape
    _, _, h_d, w_d = disparity.shape

    # Upsample disparity to SR resolution and scale by the ratio
    if h_d != h_sr or w_d != w_sr:
        scale_w = w_sr / w_d
        disp_up = F.interpolate(disparity, size=(h_sr, w_sr), mode="bilinear", align_corners=False)
        disp_up = disp_up * scale_w  # Scale disparity values to SR pixel space
    else:
        disp_up = disparity

    # Build sampling grid
    base_y = torch.linspace(-1.0, 1.0, h_sr, device=src.device, dtype=src.dtype)
    base_x = torch.linspace(-1.0, 1.0, w_sr, device=src.device, dtype=src.dtype)
    grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")
    grid_x = grid_x.unsqueeze(0).expand(b, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(b, -1, -1)

    sign = 1.0 if reverse else -1.0
    px_to_grid = 2.0 / max(w_sr - 1, 1)
    sample_x = grid_x + sign * disp_up.squeeze(1) * px_to_grid

    valid = (sample_x >= -1.0) & (sample_x <= 1.0)
    grid = torch.stack([sample_x, grid_y], dim=-1)

    warped = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid_mask = valid.unsqueeze(1).float()  # [B, 1, H, W]
    return warped, valid_mask


class WarpedStereoConsistencyLoss(nn.Module):
    """FIX 3: Disparity-warped stereo consistency loss.

    Uses predicted disparity to warp right view to left (and vice versa),
    then computes L1 photometric error on valid (non-occluded) regions.
    """

    def forward(
        self,
        sr_left: torch.Tensor,
        sr_right: torch.Tensor,
        disp_left: torch.Tensor,
        disp_right: torch.Tensor,
        conf_left: torch.Tensor | None = None,
        conf_right: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Warp right→left using left disparity
        warped_r, valid_r = _warp_with_disparity(sr_right, disp_left, reverse=False)
        # Warp left→right using right disparity
        warped_l, valid_l = _warp_with_disparity(sr_left, disp_right, reverse=True)

        # FIX 4: Confidence-weighted — if confidence available, upscale and use
        if conf_left is not None:
            conf_l_up = F.interpolate(conf_left, size=sr_left.shape[2:], mode="bilinear", align_corners=False)
            weight_r = valid_r * conf_l_up
        else:
            weight_r = valid_r

        if conf_right is not None:
            conf_r_up = F.interpolate(conf_right, size=sr_right.shape[2:], mode="bilinear", align_corners=False)
            weight_l = valid_l * conf_r_up
        else:
            weight_l = valid_l

        # Weighted photometric L1
        loss_l2r = (weight_r * (warped_r - sr_left).abs()).sum() / weight_r.sum().clamp_min(1.0)
        loss_r2l = (weight_l * (warped_l - sr_right).abs()).sum() / weight_l.sum().clamp_min(1.0)

        return 0.5 * (loss_l2r + loss_r2l)


# ---------------------------------------------------------------------------
# FIX 5: Edge-aware disparity smoothness
# ---------------------------------------------------------------------------

class DisparitySmoothness(nn.Module):
    """Edge-aware smoothness: |grad_d| * exp(-|grad_I|)."""

    def _compute(self, disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        # Resize image to disparity resolution if needed
        if img.shape[2:] != disp.shape[2:]:
            img = F.interpolate(img, size=disp.shape[2:], mode="bilinear", align_corners=False)

        dx_d = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        dy_d = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        dx_i = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
        dy_i = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)

        return (dx_d * torch.exp(-dx_i)).mean() + (dy_d * torch.exp(-dy_i)).mean()

    def forward(
        self,
        disp_left: torch.Tensor,
        disp_right: torch.Tensor,
        img_left: torch.Tensor,
        img_right: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * (self._compute(disp_left, img_left) + self._compute(disp_right, img_right))


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class StereoSRV5Loss(nn.Module):
    def __init__(
        self,
        pixel_loss: str = "charbonnier",
        charbonnier_eps: float = 1e-3,
        mse_weight: float = 0.05,
        stereo_weight: float = 0.05,
        smooth_weight: float = 0.01,
    ):
        super().__init__()
        self.pixel_loss_name = pixel_loss
        self.mse_weight = mse_weight
        self.stereo_weight = stereo_weight
        self.smooth_weight = smooth_weight

        if pixel_loss == "charbonnier":
            self.pixel = CharbonnierLoss(charbonnier_eps)
        elif pixel_loss == "l1":
            self.pixel = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported pixel loss: {pixel_loss}")

        self.stereo = WarpedStereoConsistencyLoss()
        self.smooth = DisparitySmoothness()

    def forward(
        self,
        sr_left: torch.Tensor,
        sr_right: torch.Tensor,
        hr_left: torch.Tensor,
        hr_right: torch.Tensor,
        aux: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses: Dict[str, torch.Tensor] = {}

        # Pixel reconstruction loss
        pixel = 0.5 * (self.pixel(sr_left, hr_left) + self.pixel(sr_right, hr_right))
        mse = 0.5 * (F.mse_loss(sr_left, hr_left) + F.mse_loss(sr_right, hr_right))
        total = pixel + self.mse_weight * mse
        losses["pixel"] = pixel
        losses["mse"] = mse

        # FIX 3+4: Warped stereo consistency with confidence weighting
        if self.stereo_weight > 0 and aux is not None:
            disp_l = aux.get("refined_disp_left", aux.get("deep_disp_left"))
            disp_r = aux.get("refined_disp_right", aux.get("deep_disp_right"))
            conf_l = aux.get("bottleneck_conf_left", aux.get("deep_conf_left"))
            conf_r = aux.get("bottleneck_conf_right", aux.get("deep_conf_right"))

            if disp_l is not None and disp_r is not None:
                stereo = self.stereo(sr_left, sr_right, disp_l, disp_r, conf_l, conf_r)
                total = total + self.stereo_weight * stereo
                losses["stereo"] = stereo

        # FIX 5: Edge-aware disparity smoothness
        if self.smooth_weight > 0 and aux is not None:
            disp_l = aux.get("refined_disp_left", aux.get("deep_disp_left"))
            disp_r = aux.get("refined_disp_right", aux.get("deep_disp_right"))

            if disp_l is not None and disp_r is not None:
                # Use LR input (available via hr downsampled) or sr output as image guide
                smooth = self.smooth(disp_l, disp_r, sr_left, sr_right)
                total = total + self.smooth_weight * smooth
                losses["smooth"] = smooth

        losses["total"] = total
        return total, losses
