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
        # Compute attention scores + softmax in fp32 even under AMP. fp16's
        # ~65k dynamic range overflows on long-tail Q·Kᵀ values; once exp()
        # produces inf, softmax → NaN and corrupts weights via backward.
        with torch.amp.autocast("cuda", enabled=False):
            Ql_f = Ql.float()
            QrT_f = Qr_T.float()
            attn = torch.matmul(Ql_f, QrT_f) * self.scale          # [B, H, W, W]
            attn_r2l = F.softmax(attn, dim=-1)
            attn_l2r = F.softmax(attn.transpose(-2, -1), dim=-1)

        attn_r2l = attn_r2l.to(Vr.dtype)
        attn_l2r = attn_l2r.to(Vl.dtype)

        F_r2l = torch.matmul(attn_r2l, Vr)                        # [B, H, W, C]
        F_l2r = torch.matmul(attn_l2r, Vl)

        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta             # [B, C, H, W]
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma

        return x_l + F_r2l, x_r + F_l2r
