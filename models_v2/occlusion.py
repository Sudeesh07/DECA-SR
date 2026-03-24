"""
Hybrid Occlusion-Aware Fusion for Stereo Image Super-Resolution

Uses a full disparity-aware correlation volume to drive both confidence
prediction and cross-view attention:

1. GroupwiseCorrelation builds a cost volume across D disparity candidates
2. CorrelationEncoder compresses the 3D volume into 2D spatial features
3. Confidence predictor uses L, R, |L-R|, AND correlation features
4. Cross-attention queries are modulated by correlation (warm-start matching)
5. Self-attention fallback for occluded regions (never zeros out)

Output = f_left + beta * (C * cross_out + (1-C) * self_out)

Key fix: Both paths produce RAW outputs (no internal residual from f_left).
This ensures C * cross_out != (1-C) * self_out at init, giving the
confidence map gradient signal from the very first iteration.
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
        self.num_disps = 2 * max_disp - 1

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
    correct key position. This gives attention a warm start from stereo matching.

    Returns RAW attention output (no internal f_left residual).
    The external residual is handled by HybridOcclusionFusion.
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
        # Start partially active so correlation influences queries from the start
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(
        self,
        f_left: torch.Tensor,
        f_right: torch.Tensor,
        corr_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Returns raw cross-attention output [B, C, H, W]. No residual."""
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
        return self.out_proj(out)


# ── Self-View Refinement Path ───────────────────────────────────────────


class SelfViewPath(nn.Module):
    """
    Intra-view refinement for occluded regions.

    Uses plain convolutions (NOT NAFBlocks with zero-init residual scaling)
    to ensure the output is different from the input at initialization.
    This is critical: if self_out == f_left at init, then C has no gradient
    because C * cross_out + (1-C) * f_left = f_left when cross_out = f_left too.

    With plain convolutions and random init, self_out != cross_out at init,
    giving C gradient signal to learn which regions should use which path.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw self-refined features [B, C, H, W]. No residual."""
        return self.net(x)


# ── Cross-View Refinement Path ──────────────────────────────────────────


class CrossViewPath(nn.Module):
    """
    Cross-view attention + refinement.

    Guided epipolar attention pulls info from the other view,
    then a small conv refines it. Returns raw output (no residual).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.attn = GuidedEpipolarAttention(channels)
        self.refine = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(
        self, f_left: torch.Tensor, f_right: torch.Tensor, corr_feat: torch.Tensor
    ) -> torch.Tensor:
        """Returns raw cross-view features [B, C, H, W]. No residual."""
        x = self.attn(f_left, f_right, corr_feat)
        return self.refine(x)


# ── Confidence Predictor ────────────────────────────────────────────────


class HybridConfidencePredictor(nn.Module):
    """
    Predicts occlusion confidence from four cues:
    - f_left, f_right: both view features
    - gradient: |f_left - f_right| (occlusion boundary signal)
    - corr_feat: encoded correlation volume (disparity-aware matching quality)

    A position with low correlation at ALL disparities is likely occluded.

    Output: C in [0, 1] where 1 = visible, 0 = occluded.

    Initialization: final conv bias = 0 so sigmoid(0) = 0.5.
    This means confidence starts at 0.5 everywhere, giving equal weight
    to both paths and allowing gradients to flow through C from step 1.
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
        )

        self.temperature = nn.Parameter(torch.ones(1))

        # Initialize final conv to output ~0 pre-sigmoid → sigmoid(0) = 0.5
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

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
        4. Cross path: correlation-guided epipolar attention (raw output)
        5. Self path: conv refinement (raw output)
        6. Blend: f_left + beta * (C * cross_out + (1-C) * self_out)

    Critical design for gradient flow:
        - Both paths produce RAW outputs (not f_left + something)
        - At init, cross_out != self_out (different random weights)
        - So C has gradient: d/dC [C*a + (1-C)*b] = a - b != 0
        - beta starts small (0.01) so the module doesn't perturb f_left much
          at init, but is nonzero so gradients flow through all components
        - Confidence starts at 0.5 (sigmoid of zero-init bias)
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

        # Confidence prediction (initialized to output 0.5)
        self.confidence = HybridConfidencePredictor(channels)

        # Cross path: guided attention + refinement (raw output, no residual)
        self.cross_path = CrossViewPath(channels)

        # Self path: intra-view refinement (raw output, no residual)
        self.self_path = SelfViewPath(channels)

        # External residual scaling
        # Starts small but nonzero so gradients flow through everything
        self.beta = nn.Parameter(torch.tensor(0.01))

    def forward(
        self, f_left: torch.Tensor, f_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused:      [B, C, H, W] occlusion-aware fused features
            confidence: [B, 1, H, W] confidence map (1=visible, 0=occluded)
        """
        # Disparity-aware correlation
        corr_vol = self.corr(f_left, f_right)       # [B, G, D, H, W]
        corr_feat = self.corr_encoder(corr_vol)      # [B, C, H, W]

        # Confidence (starts at 0.5 everywhere)
        C = self.confidence(f_left, f_right, corr_feat)  # [B, 1, H, W]

        # Cross path: raw cross-view output (NOT f_left + ...)
        cross_out = self.cross_path(f_left, f_right, corr_feat)  # [B, C, H, W]

        # Self path: raw self-view output (NOT f_left + ...)
        self_out = self.self_path(f_left)  # [B, C, H, W]

        # Blend with external residual
        # At init: beta=0.01, C=0.5, cross_out and self_out are random
        # Output ≈ f_left + 0.01 * (0.5 * random_a + 0.5 * random_b) ≈ f_left
        # But crucially, d(output)/dC = beta * (cross_out - self_out) != 0
        # So C receives gradient and can learn from step 1
        blended = C * cross_out + (1 - C) * self_out
        fused = f_left + self.beta * blended

        return fused, C