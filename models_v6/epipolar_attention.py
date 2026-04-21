"""Stereo Cross-Attention Module (SCAM) — spatial-domain epipolar fusion.

This is the row-wise cross-view attention introduced in NAFSSR (Chu et al.,
CVPRW 2022). For each row y, left queries attend over right keys along the
full width (and symmetrically for right→left). This yields a strong spatial
baseline; CVSFM complements it by operating in the spectral domain.

Both residual scales β, γ are zero-initialised so that SCAM, like every other
V6 block, is the identity at initialisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm2d


class StereoCrossAttention(nn.Module):
    """Row-wise (epipolar) cross-view attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.scale = channels ** -0.5
        self.norm_l = LayerNorm2d(channels)
        self.norm_r = LayerNorm2d(channels)
        self.q_l = nn.Conv2d(channels, channels, 1)
        self.q_r = nn.Conv2d(channels, channels, 1)
        self.v_l = nn.Conv2d(channels, channels, 1)
        self.v_r = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x_l: torch.Tensor, x_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x_l.shape
        # Projections
        Ql = self.q_l(self.norm_l(x_l)).permute(0, 2, 3, 1)        # [B, H, W, C]
        Qr_T = self.q_r(self.norm_r(x_r)).permute(0, 2, 1, 3)      # [B, H, C, W]
        Vl = self.v_l(x_l).permute(0, 2, 3, 1)                    # [B, H, W, C]
        Vr = self.v_r(x_r).permute(0, 2, 3, 1)                    # [B, H, W, C]

        # Attention along the epipolar (width) axis, per row.
        attn = torch.matmul(Ql, Qr_T) * self.scale                # [B, H, W, W]

        # Right → Left direction (left queries, right values)
        F_r2l = torch.matmul(F.softmax(attn, dim=-1), Vr)         # [B, H, W, C]
        # Left → Right direction (right queries, left values)
        F_l2r = torch.matmul(F.softmax(attn.permute(0, 1, 3, 2), dim=-1), Vl)

        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta             # [B, C, H, W]
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma

        return x_l + F_r2l, x_r + F_l2r
