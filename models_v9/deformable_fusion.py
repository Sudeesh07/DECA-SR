"""Deformable Epipolar Cost-view-Aware fusion (DECVF) — V9 (DECA-SR) novelty.

The cross-view operator. For each pixel in the self view it:
  1. builds a lightweight cost volume along the SAME epipolar row (D shifted
     dot-products on a reduced key channel) -> a soft disparity prior d0 and a
     per-pixel match confidence (peak sharpness);
  2. uses d0 to INITIALIZE + BOUND learned deformable offsets (K sample points,
     x-offset centred on d0 with a small learnable correction, y-offset clamped
     small to respect epipolar geometry while tolerating rectification residual);
  3. sub-pixel-samples the other view's value map at those K points (grid_sample,
     bilinear) and aggregates by per-point weights * match confidence;
  4. fuses through an SFAM-style global L/R channel gate and a zero-init
     per-channel gamma residual.

Why this beats SCAM's dense soft row-attention:
  * sub-pixel resampling (vs softmax-averaging over the whole row) places the
    correct other-view texture WITHOUT low-pass blur -> sharper HR detail;
  * the cost-volume prior fixes the cold-start that makes pure learned-offset
    deformable attention unstable (the failure of the earlier V4 deformable try);
  * the small 2D (mostly-horizontal) offsets absorb the vertical-rectification
    bleed that strict-1D SCAM / phase-correlation provably cannot represent
    (the documented KITTI failure mode);
  * confidence gating suppresses fusion in occluded / textureless regions.

Numerics: cost-volume softmax, soft-argmax, offset head and all grid_sample
calls run in fp32 under ``autocast(enabled=False)`` (the project's bf16-stability
contract, same as SCAM). Identity at init: gamma_l = gamma_r = 0, and the offset
head's last layer is zero-init so offsets start exactly at the cost-volume
disparity d0 (y-offset 0).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm2d


class DeformableEpipolarFusion(nn.Module):
    """DECVF cross-view fusion block (bidirectional, weight-shared)."""

    def __init__(
        self,
        channels: int,
        cost_volume_d: int = 24,
        num_points: int = 8,
        y_offset_range: float = 1.5,
        x_correction_range: float = 8.0,
        key_ratio: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.D = int(cost_volume_d)
        self.K = int(num_points)
        self.y_range = float(y_offset_range)
        self.x_range = float(x_correction_range)
        cr = max(channels // key_ratio, 8)
        self.cr = cr

        # Shared reduce (per-view LayerNorm + 1x1) for the cost-volume keys.
        self.reduce = nn.Sequential(LayerNorm2d(channels), nn.Conv2d(channels, cr, 1))
        # Value projection of the other view (the fusion payload).
        self.v_proj = nn.Conv2d(channels, channels, 1)
        # Offset / weight head: input = concat(g_self, warped_g_other) = 2*cr.
        self.offset_head = nn.Sequential(
            nn.Conv2d(2 * cr, 2 * cr, 3, padding=1, groups=2 * cr),
            nn.Conv2d(2 * cr, 3 * self.K, 1),  # K * (dx_raw, dy_raw, logit)
        )
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        # Output projection of the aggregated other-view features.
        self.out_proj = nn.Conv2d(channels, channels, 1)
        # SFAM-style global L/R channel gate on the fusion residual (bottlenecked).
        gate_hidden = max(channels // 4, 8)
        self.sfam_gate = nn.Sequential(
            nn.Conv2d(2 * channels, gate_hidden, 1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, channels, 1),
        )
        # Learnable cost-volume softmax sharpness.
        self.cv_scale = nn.Parameter(torch.ones(1))
        # Zero-init residual scales -> identity at step 0.
        self.gamma_l = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma_r = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Diagnostics (Python attrs; DDP-safe, not in state_dict).
        self._diag_gamma_abs: torch.Tensor | None = None
        self._diag_conf: torch.Tensor | None = None
        self._diag_disp: torch.Tensor | None = None
        self._diag_dy_abs: torch.Tensor | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def _shifted_corr(g_self: torch.Tensor, g_other: torch.Tensor, s: int, sign: int) -> torch.Tensor:
        """sum_c g_self(x) * g_other(x + sign*s) over channels -> [B,1,H,W]."""
        W = g_self.shape[-1]
        if s == 0:
            go = g_other
        elif sign < 0:  # self queries other at x - s  -> shift other right
            go = F.pad(g_other, (s, 0))[..., :W]
        else:           # self queries other at x + s  -> shift other left
            go = F.pad(g_other, (0, s))[..., s:]
        return (g_self * go).sum(dim=1, keepdim=True)

    def _sample(self, value: torch.Tensor, disp_x: torch.Tensor, disp_y: torch.Tensor, sign: int) -> torch.Tensor:
        """Bilinear-sample ``value`` at (x + sign*disp_x, y + disp_y). All fp32.

        ``disp_x``/``disp_y`` broadcast to [B,H,W]; returns [B,C,H,W].
        """
        B, C, H, W = value.shape
        device = value.device
        ys = torch.arange(H, device=device, dtype=value.dtype).view(1, H, 1)
        xs = torch.arange(W, device=device, dtype=value.dtype).view(1, 1, W)
        sample_x = xs + sign * disp_x
        sample_y = ys + disp_y
        # Normalize to [-1, 1] for grid_sample (align_corners=True).
        gx = 2.0 * sample_x / max(W - 1, 1) - 1.0
        gy = 2.0 * sample_y / max(H - 1, 1) - 1.0
        grid = torch.stack(
            [gx.expand(B, H, W), gy.expand(B, H, W)], dim=-1
        )  # [B,H,W,2]
        return F.grid_sample(value, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def _fuse(self, x_self: torch.Tensor, x_other: torch.Tensor, gamma: torch.Tensor, sign: int):
        """One fusion direction. sign=-1 for L<-R (match at x-d), +1 for R<-L."""
        g_self = self.reduce(x_self)
        g_other = self.reduce(x_other)
        v_other = self.v_proj(x_other)

        with torch.amp.autocast("cuda", enabled=False):
            g_self_f = g_self.float()
            g_other_f = g_other.float()
            v_other_f = v_other.float()
            scale = self.cr ** -0.5

            # Cost volume over D candidate disparities -> soft prior + confidence.
            corr = torch.cat(
                [self._shifted_corr(g_self_f, g_other_f, s, sign) * scale for s in range(self.D)],
                dim=1,
            )  # [B, D, H, W]
            p = F.softmax(corr * self.cv_scale.float(), dim=1)
            idx = torch.arange(self.D, device=corr.device, dtype=corr.dtype).view(1, self.D, 1, 1)
            d0 = (p * idx).sum(dim=1, keepdim=True)            # [B,1,H,W] soft disparity
            conf = p.max(dim=1, keepdim=True).values           # [B,1,H,W] peak sharpness

            warped_g = self._sample(g_other_f, d0[:, 0], torch.zeros_like(d0[:, 0]), sign)
            head_in = torch.cat([g_self_f, warped_g], dim=1)   # [B, 2cr, H, W]
            raw = self.offset_head(head_in)                    # [B, 3K, H, W]
            dx_raw, dy_raw, logit = torch.split(raw, self.K, dim=1)
            disp_x = d0 + self.x_range * torch.tanh(dx_raw)     # [B,K,H,W], centred on d0
            disp_y = self.y_range * torch.tanh(dy_raw)          # [B,K,H,W], small
            w = F.softmax(logit, dim=1)                         # [B,K,H,W] per-point weights

            agg = torch.zeros_like(v_other_f)
            for k in range(self.K):
                sampled = self._sample(v_other_f, disp_x[:, k], disp_y[:, k], sign)
                agg = agg + w[:, k:k + 1] * sampled
            agg = agg * conf                                    # gate by match confidence

        # SFAM-style global channel gate (cheap, complements per-pixel conf).
        desc = torch.cat([x_self.mean(dim=(-2, -1), keepdim=True),
                          x_other.mean(dim=(-2, -1), keepdim=True)], dim=1)
        sg = torch.sigmoid(self.sfam_gate(desc))
        residual = sg * self.out_proj(agg.to(x_self.dtype))
        out = x_self + gamma * residual

        # Diagnostics
        with torch.no_grad():
            self._diag_gamma_abs = gamma.detach().abs().mean()
            self._diag_conf = conf.detach().mean()
            self._diag_disp = d0.detach().mean()
            self._diag_dy_abs = disp_y.detach().abs().mean()
        return out

    def forward(self, x_l: torch.Tensor, x_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_l = self._fuse(x_l, x_r, self.gamma_l, sign=-1)   # L borrows from R at x-d
        out_r = self._fuse(x_r, x_l, self.gamma_r, sign=+1)   # R borrows from L at x+d
        return out_l, out_r
