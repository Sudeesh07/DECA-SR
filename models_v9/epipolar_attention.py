"""Stereo Cross-Attention Module (SCAM) — V6/NAFSSR, reused verbatim in V9.

SCAM is the *only* cross-view mechanism in this project's history that beat the
no-fusion baseline (A0 23.41 -> A1 23.70). In V9 (DECA-SR) it is the cheap dense
epipolar-row complement that runs in every block; the novel DECVF adds sub-pixel
deformable + 2D-tolerant fusion on top. Both are zero-init so the block starts
as identity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm2d


class StereoCrossAttention(nn.Module):
    """Row-wise epipolar cross-view attention. Identical to V6's SCAM."""

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
        Ql = self.q_l(self.norm_l(x_l)).permute(0, 2, 3, 1)        # [B, H, W, C]
        Qr_T = self.q_r(self.norm_r(x_r)).permute(0, 2, 1, 3)      # [B, H, C, W]
        Vl = self.v_l(x_l).permute(0, 2, 3, 1)                    # [B, H, W, C]
        Vr = self.v_r(x_r).permute(0, 2, 3, 1)                    # [B, H, W, C]

        # Attention + softmax in fp32 even under AMP (overflow -> NaN otherwise).
        with torch.amp.autocast("cuda", enabled=False):
            attn = torch.matmul(Ql.float(), Qr_T.float()) * self.scale   # [B, H, W, W]
            attn_r2l = F.softmax(attn, dim=-1)
            attn_l2r = F.softmax(attn.transpose(-2, -1), dim=-1)

        F_r2l = torch.matmul(attn_r2l.to(Vr.dtype), Vr).permute(0, 3, 1, 2) * self.beta
        F_l2r = torch.matmul(attn_l2r.to(Vl.dtype), Vl).permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r
