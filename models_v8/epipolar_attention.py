"""Stereo Cross-Attention Module (SCAM) — V6-style, V8 keeps this UNCHANGED.

V7's disparity-prior bias on the SCAM logits saturated at the parametrisation
boundary and dropped val PSNR to 22.12 dB. V8 reverts SCAM to the clean
V6 design (NAFSSR, Chu et al. CVPRW 2022) — the *only* cross-view mechanism in
this project's history that beat the no-fusion baseline (A0 = 23.41 → A1 = 23.70).

What SCAM does
--------------
For each row y independently:
  Q_L = q_l(norm_l(x_L)), Q_R = q_r(norm_r(x_R))     [B, H, W, C]
  V_L = v_l(x_L),         V_R = v_r(x_R)
  attn = softmax(Q_L · Q_R^T / sqrt(C))               [B, H, W, W]
  F_r2l = attn · V_R,     F_l2r = attn^T · V_L
  out_L = x_L + β · F_r2l,  out_R = x_R + γ · F_l2r

β, γ are zero-init per-channel scalars → identity at step 0.

V8 adds the *channel-transposed* cross-view attention (CT-CVA) in parallel —
documented in channel_cross_view.py. SCAM and CT-CVA combine in
StereoSpatialChannelBlock (network_v8.py).
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
        B, C, H, W = x_l.shape

        Ql = self.q_l(self.norm_l(x_l)).permute(0, 2, 3, 1)        # [B, H, W, C]
        Qr_T = self.q_r(self.norm_r(x_r)).permute(0, 2, 1, 3)      # [B, H, C, W]
        Vl = self.v_l(x_l).permute(0, 2, 3, 1)                    # [B, H, W, C]
        Vr = self.v_r(x_r).permute(0, 2, 3, 1)                    # [B, H, W, C]

        # Attention + softmax in fp32 even under AMP (long-tail Q·K^T overflows
        # fp16, → softmax NaN, → weights corrupted via backward).
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
