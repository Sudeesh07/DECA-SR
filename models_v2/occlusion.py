"""
Hybrid Occlusion-Aware Fusion for Stereo Image Super-Resolution

Uses a full disparity-aware correlation volume to drive both confidence
prediction and cross-view attention:

1. GroupwiseCorrelation builds a cost volume across D disparity candidates
2. CorrelationEncoder compresses the 3D volume into 2D spatial features
3. Confidence predictor uses L, R, |L-R|, AND correlation features
4. Cross-attention queries are modulated by correlation (warm-start matching)
5. Self-attention fallback for occluded regions (never zeros out)

Output = C * cross_out + (1-C) * self_out
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from models.nafblock import NAFBlock, LayerNorm2d


# ── Group-wise Disparity Correlation ────────────────────────────────────


class GroupwiseCorrelation(nn.Module):
    """
    Bidirectional group-wise correlation at D disparity candidates.

    Searches BOTH positive and negative disparities so the module
    works correctly regardless of which view is target vs reference.

    Output: [B, G, 2*max_disp-1, H, W] correlation volume.
    """

    def __init__(self, channels: int, groups: int = 8, max_disp: int = 16):
        super().__init__()
        assert channels % groups == 0
        self.groups = groups
        self.max_disp = max_disp
        self.group_channels = channels // groups
        self.num_disps = 2 * max_disp - 1  # [-max_disp+1, ..., 0, ..., max_disp-1]

    def forward(
        self, f_left: torch.Tensor, f_right: torch.Tensor
    ) -> torch.Tensor:
        B, _, H, W = f_left.shape
        G, Cg = self.groups, self.group_channels

        fL = F.normalize(f_left.view(B, G, Cg, H, W), dim=2)
        fR = F.normalize(f_right.view(B, G, Cg, H, W), dim=2)

        corrs = []

        # Negative disparities: left position x correlates with right position x+d
        for d in range(self.max_disp - 1, 0, -1):
            corr = (fL[:, :, :, :, :-d] * fR[:, :, :, :, d:]).mean(dim=2)
            corr = F.pad(corr, (0, d), mode="constant", value=0)
            corrs.append(corr)

        # Zero disparity
        corrs.append((fL * fR).mean(dim=2))

        # Positive disparities: left position x correlates with right position x-d
        for d in range(1, self.max_disp):
            corr = (fL[:, :, :, :, d:] * fR[:, :, :, :, :W - d]).mean(dim=2)
            corr = F.pad(corr, (d, 0), mode="constant", value=0)
            corrs.append(corr)

        return torch.stack(corrs, dim=2)  # [B, G, 2*max_disp-1, H, W]


# ── Correlation Encoder ─────────────────────────────────────────────────


class CorrelationEncoder(nn.Module):
    """
    Encodes [B, G, D, H, W] correlation volume into [B, C, H, W] features
    using 3D convolutions (process across disparity + spatial jointly),
    then collapses the disparity dimension.
    """

    def __init__(self, groups: int, out_channels: int):
        super().__init__()
        mid = max(groups, out_channels // 2)
        self.conv = nn.Sequential(
            nn.Conv3d(groups, mid, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(mid, out_channels, 3, padding=1),
        )

    def forward(self, corr_vol: torch.Tensor) -> torch.Tensor:
        x = self.conv(corr_vol)   # [B, C, D, H, W]
        return x.mean(dim=2)      # [B, C, H, W]


# ── Correlation-Guided Epipolar Attention ───────────────────────────────


class GuidedEpipolarAttention(nn.Module):
    """
    Row-wise cross-attention with correlation-guided queries.

    The correlation features encode disparity information — adding them
    to the query shifts each query's representation towards matching the
    correct key position, producing higher attention scores at the right
    disparities. This gives attention a warm start from stereo matching.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.scale = channels ** -0.5

        self.norm_l = LayerNorm2d(channels)
        self.norm_r = LayerNorm2d(channels)

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)

        # Correlation modulates queries (tells them WHERE to look)
        self.corr_to_q = nn.Conv2d(channels, channels, 1)

        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.alpha = nn.Parameter(torch.zeros(1))  # starts at 0 → safe init


    def forward(
        self,
        f_left: torch.Tensor,
        f_right: torch.Tensor,
        corr_feat: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = f_left.shape

        fl = self.norm_l(f_left)
        fr = self.norm_r(f_right)

        # Query incorporates correlation prior
        q = self.q_proj(fl) + self.alpha * self.corr_to_q(corr_feat)        
        k = self.k_proj(fr)
        v = self.v_proj(fr)

        # Row-wise attention: [B, C, H, W] -> [B*H, W, C]
        q = q.permute(0, 2, 3, 1).reshape(B * H, W, C)
        k = k.permute(0, 2, 3, 1).reshape(B * H, W, C)
        v = v.permute(0, 2, 3, 1).reshape(B * H, W, C)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B*H, W, W]
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v)  # [B*H, W, C]

        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        out = self.out_proj(out)

        return f_left + out * self.beta


# ── Confidence Predictor ────────────────────────────────────────────────


class HybridConfidencePredictor(nn.Module):
    """
    Predicts occlusion confidence from four cues:
    - f_left, f_right: both view features (symmetric)
    - gradient: |f_left - f_right| (occlusion boundary signal)
    - corr_feat: encoded correlation volume (disparity-aware matching quality)

    A position with low correlation at ALL disparities is likely occluded.
    This is strictly more informative than zero-disparity correlation alone.

    Output: C in [0, 1] where 1 = visible, 0 = occluded.
    """

    def __init__(self, channels: int):
        super().__init__()
        reduced = channels // 4

        # 4C input: f_left + f_right + gradient + corr_feat
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, reduced, 1),
        )

        self.head = nn.Sequential(
            nn.Conv2d(reduced, reduced, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(reduced, 1, 3, padding=1),
            # nn.Sigmoid(),
        )
        self.temperature = nn.Parameter(torch.ones(1))



    def forward(
        self,
        f_left: torch.Tensor,
        f_right: torch.Tensor,
        corr_feat: torch.Tensor,
    ) -> torch.Tensor:
        grad = (f_left - f_right).abs()
        x = torch.cat([f_left, f_right, grad, corr_feat], dim=1)
        x = self.reduce(x)
        logits = self.head(x)
        return torch.sigmoid(logits / (self.temperature + 1e-6))

# ── Hybrid Occlusion Fusion ────────────────────────────────────────────


class HybridOcclusionFusion(nn.Module):
    """
    Disparity-aware occlusion fusion.

    Flow:
        1. Build correlation volume across D disparities
        2. Encode into spatial features via 3D conv
        3. Predict confidence using L, R, gradient, and correlation
        4. Cross path: correlation-guided epipolar attention + NAFBlock
        5. Self path: NAFBlock x 2 (intra-view fallback)
        6. Blend: C * cross_out + (1-C) * self_out

    Both paths have internal residuals from f_left, so:
    - At init (beta/gamma=0): output = f_left (identity)
    - Occluded pixels (C≈0): fall back to self-attention, never to zero
    """

    def __init__(
        self,
        channels: int,
        corr_groups: int = 8,
        corr_max_disp: int = 16,
    ):
        super().__init__()

        # Correlation volume
        self.corr = GroupwiseCorrelation(channels, corr_groups, corr_max_disp)
        self.corr_encoder = CorrelationEncoder(corr_groups, channels)

        # Confidence prediction
        self.confidence = HybridConfidencePredictor(channels)

        # Cross path: guided attention + refinement
        self.cross_attn = GuidedEpipolarAttention(channels)
        self.cross_refine = NAFBlock(channels)

        # Self path: intra-view only
        self.self_path = nn.Sequential(
            NAFBlock(channels),
            NAFBlock(channels),
        )

    def forward(
        self, f_left: torch.Tensor, f_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused:      [B, C, H, W] occlusion-aware fused features
            confidence: [B, 1, H, W] confidence map
        """
        # Disparity-aware correlation
        corr_vol = self.corr(f_left, f_right)       # [B, G, D, H, W]
        corr_feat = self.corr_encoder(corr_vol)      # [B, C, H, W]

        # Confidence
        C = self.confidence(f_left, f_right, corr_feat)

        # Cross path (correlation-guided)
        cross_out = self.cross_attn(f_left, f_right, corr_feat)
        cross_out = self.cross_refine(cross_out)

        # Self path
        self_out = self.self_path(f_left)

        # Blend: both paths contain f_left via internal residuals
        fused = C * cross_out + (1 - C) * self_out

        return fused, C
