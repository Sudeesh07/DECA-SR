"""Core blocks for DSRPSNRNet (models_v3)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nafblock import LayerNorm2d
from models_v3.depa_adapter import StableDEPAAdapter


class DLFEBlock(nn.Module):
    """
    Deformable-like Local Feature Extraction block.

    This block is SR-oriented and lightweight:
    - LayerNorm2d pre-norm
    - local adaptive aggregation (identity + 2 depthwise kernels)
    - channel FFN branch
    - learnable residual scaling for stable training
    """

    def __init__(self, channels: int, ffn_expand: int = 2):
        super().__init__()
        hidden = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.pre = nn.Conv2d(channels, channels, 1)

        # Local adaptive branch with deformable-like dynamic mixing.
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.dw5d2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels)
        self.mix_logits = nn.Conv2d(channels, 3, 3, padding=1)
        self.local_proj = nn.Conv2d(channels, channels, 1)

        self.norm2 = LayerNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        base = self.pre(y)
        b1 = self.dw3(base)
        b2 = self.dw5d2(base)

        logits = self.mix_logits(y)
        weights = torch.softmax(logits, dim=1)
        local = (
            weights[:, 0:1] * base
            + weights[:, 1:2] * b1
            + weights[:, 2:3] * b2
        )
        local = self.local_proj(local)
        x = x + local * self.beta

        z = self.norm2(x)
        z = self.ffn(z)
        x = x + z * self.gamma
        return x


class HybridStereoFusion(nn.Module):
    """
    Controlled stereo fusion using stable DEPA residual + gated residual fusion.

    Input: target feature, reference feature
    Output: updated target feature
    """

    def __init__(
        self,
        channels: int,
        depa_k: int = 8,
        num_heads: int = 4,
        max_disparity: int = 32,
        sample_span: float = 4.0,
        residual_range: float = 3.0,
        gate_hidden_ratio: float = 1.0,
    ):
        super().__init__()
        hidden = max(channels, int(channels * gate_hidden_ratio))

        self.norm_t = LayerNorm2d(channels)
        self.norm_r = LayerNorm2d(channels)

        self.stereo = StableDEPAAdapter(
            channels=channels,
            k=depa_k,
            num_heads=num_heads,
            max_disparity=max_disparity,
            sample_span=sample_span,
            residual_range=residual_range,
        )

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        t = self.norm_t(target)
        r = self.norm_r(reference)

        stereo_res = self.stereo(t, r)
        gate = self.gate(torch.cat([t, stereo_res], dim=1))
        fused = self.fuse(torch.cat([t, gate * stereo_res], dim=1))

        return target + fused * self.alpha


class ReconstructionBlock(nn.Module):
    """Lightweight decoder refinement block for high-resolution reconstruction."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.conv1(y)
        y = self.act(y)
        y = self.conv2(y)
        return x + y * self.scale
