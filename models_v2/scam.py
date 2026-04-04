"""
Lightweight Stereo Cross-Attention Module (SCAM) for frequent cross-view
interaction between NAFBlocks within encoder stages.

Based on NAFSSR's SCAM design: element-wise correlation → softmax → weighted sum.
Much cheaper than full DEPA (no offset prediction, no ECA, no multi-head attention).
Used every 2 NAFBlocks for dense cross-view information flow, with full DEPA
reserved for end-of-stage deep fusion.

Reference: NAFSSR (Chu et al., CVPRW 2022) — 32 SCAMs after every NAFBlock
achieved best results, with each SCAM adding only marginal parameter cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nafblock import LayerNorm2d


class LightweightSCAM(nn.Module):
    """
    Stereo Cross-Attention Module (lightweight).

    For each spatial position, computes channel-wise correlation between
    left and right views, applies softmax to get attention weights,
    and uses them to aggregate cross-view information. The entire
    operation is O(H*W*C) — no quadratic attention, no grid sampling.

    This is intentionally simple. The heavy lifting (deformable spatial
    attention, channel attention, offset prediction) is done by DEPA
    at the end of each stage. SCAM's job is just to keep cross-view
    information flowing between NAFBlocks so they can condition their
    intra-view processing on the other view.

    Args:
        channels: Number of feature channels
    """

    def __init__(self, channels: int):
        super().__init__()

        self.norm_l = LayerNorm2d(channels)
        self.norm_r = LayerNorm2d(channels)

        # Project both views to a shared correlation space
        # 1x1 conv is cheap and gives each view a learned representation
        self.proj_l = nn.Conv2d(channels, channels, 1)
        self.proj_r = nn.Conv2d(channels, channels, 1)

        # Scale factor for correlation (prevents softmax saturation)
        self.scale = channels ** -0.5

        # Learnable residual scaling — starts at zero (identity at init)
        self.beta_l = nn.Parameter(torch.ones(1, channels, 1, 1)*0.05)
        self.beta_r = nn.Parameter(torch.ones(1, channels, 1, 1)*0.05)

    def forward(
        self, f_left: torch.Tensor, f_right: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_left:  [B, C, H, W]
            f_right: [B, C, H, W]

        Returns:
            (enhanced_left, enhanced_right)
        """
        # Normalize
        fl = self.norm_l(f_left)
        fr = self.norm_r(f_right)

        # Project
        ql = self.proj_l(fl)  # [B, C, H, W]
        qr = self.proj_r(fr)  # [B, C, H, W]

        # Channel-wise correlation at each spatial position
        # corr[b, c, h, w] = ql[b, c, h, w] * qr[b, c, h, w]
        # This is element-wise — O(BCHW), no quadratic cost
        corr = ql * qr * self.scale  # [B, C, H, W]

        # Attention weights via sigmoid (per-channel, per-position gate)
        # sigmoid is better than softmax here because we're gating
        # individual channel-position pairs, not competing across positions
        attn = torch.sigmoid(corr)  # [B, C, H, W]

        # Cross-view aggregation:
        # High attention → this position+channel has good correspondence
        # → pull information from the other view
        # Low attention → poor correspondence (occlusion, texture-less)
        # → keep original features
        enhanced_l = f_left + self.beta_l * (attn * fr)
        enhanced_r = f_right + self.beta_r * (attn * fl)

        return enhanced_l, enhanced_r


class InterleavedEncoderStage(nn.Module):
    """
    Encoder stage with interleaved NAFBlocks and lightweight SCAMs.

    Instead of: [NAFBlock x N] → BidirectionalDEPA → downsample
    This does:  [NAFBlock → NAFBlock → SCAM] x (N//2) → BidirectionalDEPA → downsample

    Cross-view information flows every 2 NAFBlocks via cheap SCAM,
    with full DEPA fusion at the end of the stage.

    Args:
        channels: Feature channels
        num_blocks: Total NAFBlocks in this stage
        scam_interval: Insert SCAM every this many NAFBlocks (default: 2)
        depa_k: Sampling points for end-of-stage DEPA
        depa_max_disparity: Max disparity for DEPA
        num_heads: Attention heads for DEPA
        downsample: Whether to downsample at the end
        drop_path_rates: Per-block drop path rates
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        scam_interval: int = 2,
        depa_k: int = 8,
        depa_max_disparity: int = 32,
        num_heads: int = 4,
        downsample: bool = True,
        drop_path_rates: list[float] | None = None,
    ):
        super().__init__()

        from .depa import BidirectionalDEPA
        from models.nafblock import NAFBlock

        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks

        # Build interleaved sequence: NAFBlocks with SCAMs inserted
        self.naf_blocks = nn.ModuleList()
        self.scam_positions = set()
        self.scams = nn.ModuleDict()

        for i in range(num_blocks):
            self.naf_blocks.append(
                NAFBlock(channels, expansion=2, drop_path=drop_path_rates[i])
            )
            # Insert SCAM after every scam_interval blocks (but not after the last block,
            # since DEPA follows immediately)
            if (i + 1) % scam_interval == 0 and (i + 1) < num_blocks:
                self.scam_positions.add(i)
                self.scams[str(i)] = LightweightSCAM(channels)

        # Full DEPA at end of stage for deep cross-view fusion
        self.cross_view = BidirectionalDEPA(
            channels, k=depa_k, num_heads=num_heads, max_disparity=depa_max_disparity,
        )

        # Optional downsample
        if downsample:
            self.down = nn.Sequential(
                LayerNorm2d(channels),
                nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            )
        else:
            self.down = None

    def forward(
        self, f_left: torch.Tensor, f_right: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor,  # post-DEPA (skip connections)
        torch.Tensor, torch.Tensor,  # next-stage input (possibly downsampled)
        torch.Tensor, torch.Tensor,  # pre-DEPA (for bottleneck)
    ]:
        fl, fr = f_left, f_right

        # Interleaved NAFBlocks + SCAMs
        for i, block in enumerate(self.naf_blocks):
            # Shared-weight NAFBlock applied independently to each view
            fl = block(fl)
            fr = block(fr)

            # Lightweight cross-view interaction at SCAM positions
            if i in self.scam_positions:
                fl, fr = self.scams[str(i)](fl, fr)

        # Save pre-DEPA features (for bottleneck at deepest stage)
        fl_pre, fr_pre = fl, fr

        # Full DEPA cross-view fusion
        fl, fr = self.cross_view(fl, fr)

        # Save post-DEPA features (for skip connections)
        fl_skip, fr_skip = fl, fr

        # Optional downsample
        if self.down is not None:
            fl_next = self.down(fl)
            fr_next = self.down(fr)
        else:
            fl_next, fr_next = fl, fr

        return fl_skip, fr_skip, fl_next, fr_next, fl_pre, fr_pre