"""
Stable DEPA for stereo super-resolution.

This module addresses common instability issues in epipolar attention:
1) Adds an explicit disparity prior via soft-argmax cost volume.
2) Uses zero padding in grid_sample (no border hallucination).
3) Removes hard coordinate clamping before sampling.
4) Adds validity masking for out-of-bound sampled positions.
5) Uses bounded residual offsets (tanh).
6) Keeps epipolar search horizontal only.

Key classes:
- DisparityPriorEstimator
- StableEpipolarSpatialAttention
- StableEpipolarChannelAttention
- StableDEPA
- BidirectionalStableDEPA

Also includes integration-ready components:
- StableEncoderStage
- StableBottleneckCorrelationPrior
- StableDecoderStage
- SimplePixelShuffleHead
- StereoSRNetStableIntegration
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.nafblock import LayerNorm2d, NAFBlock
except ModuleNotFoundError:
    # Allow running this file directly: python models_v2/depa_stable.py
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from models.nafblock import LayerNorm2d, NAFBlock
from models_v2.frequency import LightweightFrequencyRefinement
from models_v2.occlusion import HybridOcclusionFusion


# -----------------------------------------------------------------------------
# Helper tensor functions (einops-free)
# -----------------------------------------------------------------------------


def _to_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[B, C, H, W] -> [B, num_heads, head_dim, N]."""
    b, c, h, w = x.shape
    if c % num_heads != 0:
        raise ValueError(f"channels={c} must be divisible by num_heads={num_heads}")
    d = c // num_heads
    return x.view(b, num_heads, d, h * w)


def _from_heads(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """[B, num_heads, head_dim, N] -> [B, C, H, W]."""
    b, nh, d, _ = x.shape
    return x.reshape(b, nh * d, h, w)


def _make_base_grid(b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns base normalized grids x,y with shape [B, H, W]."""
    gy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    gx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    by, bx = torch.meshgrid(gy, gx, indexing="ij")
    bx = bx.unsqueeze(0).expand(b, -1, -1)
    by = by.unsqueeze(0).expand(b, -1, -1)
    return bx, by


def _masked_softmax(logits: torch.Tensor, valid_mask: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Numerically stable masked softmax.

    logits: [..., K]
    valid_mask: broadcastable bool/float mask with same shape on dim K
    """
    mask = valid_mask.to(dtype=logits.dtype)

    # Large negative value for invalid entries before softmax.
    neg = torch.finfo(logits.dtype).min / 2.0
    masked_logits = torch.where(valid_mask, logits, torch.full_like(logits, neg))

    # Standard softmax, then explicitly zero invalid and renormalize.
    attn = torch.softmax(masked_logits, dim=dim)
    attn = attn * mask
    denom = attn.sum(dim=dim, keepdim=True).clamp_min(eps)
    return attn / denom


# -----------------------------------------------------------------------------
# A) Disparity Prior Estimation
# -----------------------------------------------------------------------------


class DisparityPriorEstimator(nn.Module):
    """
    Builds a 1D horizontal disparity cost volume using group-wise normalized correlation,
    then estimates disparity via soft-argmax.

    Stereo convention used here:
    - Positive disparity d means left(x) matches right(x - d).
    - Search is LEFT -> RIGHT along horizontal epipolar lines only.

    Inputs:
    - f_left:  [B, C, H, W]
    - f_right: [B, C, H, W]

    Outputs:
    - disp_prior: [B, 1, H, W] in pixel units
    - confidence: [B, 1, H, W]
    """

    def __init__(
        self,
        channels: int,
        max_disparity: int = 32,
        num_groups: int = 8,
        refine_cost: bool = True,
        cost_hidden: int = 64,
        softmax_temperature: float = 1.0,
    ):
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError(f"channels={channels} must be divisible by num_groups={num_groups}")

        self.max_disparity = max_disparity
        self.num_groups = num_groups
        self.group_channels = channels // num_groups
        self.refine_cost = refine_cost
        self.softmax_temperature = softmax_temperature

        num_disp = max_disparity + 1

        if refine_cost:
            self.refine = nn.Sequential(
                nn.Conv2d(num_disp, cost_hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(cost_hidden, num_disp, 3, padding=1),
            )
        else:
            self.refine = None

        disp_values = torch.arange(0, num_disp, dtype=torch.float32).view(1, num_disp, 1, 1)
        self.register_buffer("disp_values", disp_values, persistent=False)

    def _build_cost_volume(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        """
        Returns cost volume [B, D, H, W] where D=max_disparity+1.
        Cost is average group-wise cosine correlation.
        """
        b, c, h, w = f_left.shape
        g, cg = self.num_groups, self.group_channels
        dmax = min(self.max_disparity, w - 1)

        fl = F.normalize(f_left.view(b, g, cg, h, w), dim=2)
        fr = F.normalize(f_right.view(b, g, cg, h, w), dim=2)

        # [B, G, D, H, W]
        corr = f_left.new_zeros(b, g, self.max_disparity + 1, h, w)

        # d = 0 (no shift)
        corr[:, :, 0] = (fl * fr).mean(dim=2)

        # d > 0: compare left(x) with right(x-d)
        for d in range(1, dmax + 1):
            # Shift right feature to the right by d: shifted[..., x] = fr[..., x-d]
            fr_shift = F.pad(fr[..., :-d], (d, 0), mode="constant", value=0.0)
            corr[:, :, d] = (fl * fr_shift).mean(dim=2)

        # Average over groups -> [B, D, H, W]
        cost = corr.mean(dim=1)
        return cost

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cost = self._build_cost_volume(f_left, f_right)

        if self.refine is not None:
            cost = cost + self.refine(cost)

        logits = cost / max(self.softmax_temperature, 1e-6)
        prob = torch.softmax(logits, dim=1)

        disp = (prob * self.disp_values).sum(dim=1, keepdim=True)
        conf = prob.max(dim=1, keepdim=True)[0]

        return disp, conf


# -----------------------------------------------------------------------------
# B) Stable Epipolar Spatial Attention
# -----------------------------------------------------------------------------


class StableEpipolarSpatialAttention(nn.Module):
    """
    Spatial cross-view attention with disparity prior + residual offset refinement.

    Critical stability fixes implemented:
    - no hard clamp before grid_sample
    - padding_mode="zeros" (no border hallucination)
    - validity mask for sampled coordinates
    - invalid logits masked before softmax
    - attention renormalized after masking
    - residual offsets bounded via tanh
    """

    def __init__(
        self,
        channels: int,
        k: int = 8,
        num_heads: int = 4,
        max_disparity: int = 32,
        sample_span: float = 4.0,
        residual_range: float = 3.0,
        prior_groups: int = 8,
        refine_prior_cost: bool = True,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")

        self.k = k
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.sample_span = sample_span
        self.residual_range = residual_range

        self.prior_estimator = DisparityPriorEstimator(
            channels=channels,
            max_disparity=max_disparity,
            num_groups=prior_groups,
            refine_cost=refine_prior_cost,
        )

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)

        # Predict residual offsets around prior for each sample point.
        self.offset_residual = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, k, 3, padding=1),
        )

        # Stable scaling parameter.
        self.logit_scale = nn.Parameter(torch.tensor(self.head_dim ** -0.5).log())

        # Fixed symmetric sample anchors around prior disparity.
        if k <= 1:
            anchors = torch.zeros(1, 1, 1, 1, 1)
        else:
            anchors = torch.linspace(-sample_span, sample_span, k).view(1, k, 1, 1, 1)
        self.register_buffer("sample_anchors", anchors, persistent=False)

        # Debug outputs.
        self.last_disp_prior: Optional[torch.Tensor] = None
        self.last_prior_conf: Optional[torch.Tensor] = None

    def _sample_right_features(
        self,
        feat: torch.Tensor,
        disp_samples_px: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample right-view features at x - disparity for each of K candidates.

        feat: [B, C, H, W]
        disp_samples_px: [B, K, H, W] disparity candidates in pixels

        Returns:
        sampled: [B, C, H, W, K]
        valid:   [B, 1, H, W, K] bool
        """
        b, c, h, w = feat.shape
        k = disp_samples_px.shape[1]

        bx, by = _make_base_grid(b, h, w, feat.device, feat.dtype)
        px_to_norm = 2.0 / (w - 1) if w > 1 else 0.0

        # Stereo convention: sample from right at x - d.
        sx = bx.unsqueeze(1) - disp_samples_px * px_to_norm
        sy = by.unsqueeze(1).expand(b, k, h, w)

        # NO hard clamping here by design.
        valid = (sx >= -1.0) & (sx <= 1.0)

        # [B*K, H, W, 2]
        grid = torch.stack([sx, sy], dim=-1).reshape(b * k, h, w, 2)

        # Repeat features across K in batch dimension.
        feat_rep = feat.unsqueeze(1).expand(b, k, c, h, w).reshape(b * k, c, h, w)

        sampled = F.grid_sample(
            feat_rep,
            grid,
            mode="bilinear",
            padding_mode="zeros",  # critical fix
            align_corners=True,
        )

        sampled = sampled.view(b, k, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
        valid = valid.unsqueeze(2).permute(0, 2, 3, 4, 1).contiguous()  # [B,1,H,W,K]
        return sampled, valid

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f_left.shape

        disp_prior, prior_conf = self.prior_estimator(f_left, f_right)
        self.last_disp_prior = disp_prior.detach()
        self.last_prior_conf = prior_conf.detach()

        q = self.q_proj(f_left)
        kr = self.k_proj(f_right)
        vr = self.v_proj(f_right)

        residual = torch.tanh(self.offset_residual(torch.cat([f_left, f_right], dim=1)))
        residual = residual * self.residual_range

        # [B,K,H,W]
        disp_samples = disp_prior + self.sample_anchors.squeeze(-1) + residual

        k_samp, valid = self._sample_right_features(kr, disp_samples)
        v_samp, _ = self._sample_right_features(vr, disp_samples)

        # reshape for attention
        # q:      [B, Hh, D, N]
        # k_samp: [B, Hh, D, N, K]
        # v_samp: [B, Hh, D, N, K]
        qh = _to_heads(q, self.num_heads)
        kh = k_samp.view(b, self.num_heads, self.head_dim, h * w, self.k)
        vh = v_samp.view(b, self.num_heads, self.head_dim, h * w, self.k)

        scale = self.logit_scale.exp().clamp(1e-4, 1e4)
        logits = torch.einsum("bhdn,bhdnk->bhnk", qh, kh) * scale

        valid_nk = valid.view(b, 1, h * w, self.k)
        attn = _masked_softmax(logits, valid_nk, dim=-1)

        out = torch.einsum("bhnk,bhdnk->bhdn", attn, vh)
        return _from_heads(out, h, w)


# -----------------------------------------------------------------------------
# C) Stable Epipolar Channel Attention
# -----------------------------------------------------------------------------


class StableEpipolarChannelAttention(nn.Module):
    """
    Stable cross-view channel attention (multi-head, channel-channel attention).
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)

        # Learnable positive temperature per head.
        self.log_temp = nn.Parameter(torch.zeros(num_heads, 1, 1))

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f_left.shape
        n = h * w

        q = self.q_proj(f_left).view(b, self.num_heads, self.head_dim, n)
        k = self.k_proj(f_right).view(b, self.num_heads, self.head_dim, n)
        v = self.v_proj(f_right).view(b, self.num_heads, self.head_dim, n)

        # Normalize along spatial dimension.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # [B, Hh, D, D]
        temp = self.log_temp.exp().clamp(1e-4, 1e4)
        logits = torch.matmul(q, k.transpose(-2, -1)) / temp
        logits = logits - logits.amax(dim=-1, keepdim=True)
        attn = torch.softmax(logits, dim=-1)

        out = torch.matmul(attn, v)
        return out.view(b, c, h, w)


# -----------------------------------------------------------------------------
# D) Combined DEPA block
# -----------------------------------------------------------------------------


class StableDEPA(nn.Module):
    """
    Stable DEPA block (single direction): target <- reference cross-view transfer.
    """

    def __init__(
        self,
        channels: int,
        k: int = 8,
        num_heads: int = 4,
        max_disparity: int = 32,
        sample_span: float = 4.0,
        residual_range: float = 3.0,
        prior_groups: int = 8,
        refine_prior_cost: bool = True,
    ):
        super().__init__()

        self.norm_t = LayerNorm2d(channels)
        self.norm_r = LayerNorm2d(channels)

        self.spatial = StableEpipolarSpatialAttention(
            channels=channels,
            k=k,
            num_heads=num_heads,
            max_disparity=max_disparity,
            sample_span=sample_span,
            residual_range=residual_range,
            prior_groups=prior_groups,
            refine_prior_cost=refine_prior_cost,
        )
        self.channel = StableEpipolarChannelAttention(channels=channels, num_heads=num_heads)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.05)

    def forward(self, f_target: torch.Tensor, f_reference: torch.Tensor) -> torch.Tensor:
        ft = self.norm_t(f_target)
        fr = self.norm_r(f_reference)

        s = self.spatial(ft, fr)
        c = self.channel(ft, fr)
        fused = self.fuse(torch.cat([s, c], dim=1))

        return f_target + fused * self.beta


class BidirectionalStableDEPA(nn.Module):
    """Separate L->R and R->L StableDEPA blocks."""

    def __init__(
        self,
        channels: int,
        k: int = 8,
        num_heads: int = 4,
        max_disparity: int = 32,
        sample_span: float = 4.0,
        residual_range: float = 3.0,
        prior_groups: int = 8,
        refine_prior_cost: bool = True,
    ):
        super().__init__()

        self.l2r = StableDEPA(
            channels=channels,
            k=k,
            num_heads=num_heads,
            max_disparity=max_disparity,
            sample_span=sample_span,
            residual_range=residual_range,
            prior_groups=prior_groups,
            refine_prior_cost=refine_prior_cost,
        )
        self.r2l = StableDEPA(
            channels=channels,
            k=k,
            num_heads=num_heads,
            max_disparity=max_disparity,
            sample_span=sample_span,
            residual_range=residual_range,
            prior_groups=prior_groups,
            refine_prior_cost=refine_prior_cost,
        )

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.l2r(f_left, f_right), self.r2l(f_right, f_left)


# -----------------------------------------------------------------------------
# E) Integration components for stereo SR network
# -----------------------------------------------------------------------------


class StableEncoderStage(nn.Module):
    """
    NAF blocks + BidirectionalStableDEPA + optional downsample.

    Integration note:
    stage_disp should be set per stage as:
    stage_disp = depa_max_disparity // (2 ** i)
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        depa_k: int,
        depa_max_disparity: int,
        num_heads: int,
        downsample: bool,
        drop_path_rates: Optional[List[float]] = None,
        sample_span: float = 4.0,
        residual_range: float = 3.0,
    ):
        super().__init__()

        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks

        self.blocks = nn.Sequential(
            *[NAFBlock(channels, expansion=2, drop_path=drop_path_rates[i]) for i in range(num_blocks)]
        )

        self.cross_view = BidirectionalStableDEPA(
            channels=channels,
            k=depa_k,
            num_heads=num_heads,
            max_disparity=depa_max_disparity,
            sample_span=sample_span,
            residual_range=residual_range,
        )

        if downsample:
            self.down = nn.Sequential(
                LayerNorm2d(channels),
                nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            )
        else:
            self.down = None

    def forward(
        self, f_left: torch.Tensor, f_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fl = self.blocks(f_left)
        fr = self.blocks(f_right)
        fl_pre, fr_pre = fl, fr

        fl, fr = self.cross_view(fl, fr)
        fl_skip, fr_skip = fl, fr

        if self.down is None:
            fl_next, fr_next = fl, fr
        else:
            fl_next = self.down(fl)
            fr_next = self.down(fr)

        return fl_skip, fr_skip, fl_next, fr_next, fl_pre, fr_pre


class StableBottleneckCorrelationPrior(nn.Module):
    """
    Keep bottleneck correlation prior as in original architecture.
    """

    def __init__(self, channels: int, num_groups: int = 8, max_disp: int = 12):
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError(f"channels={channels} must be divisible by num_groups={num_groups}")

        self.num_groups = num_groups
        self.max_disp = max_disp
        self.group_channels = channels // num_groups

        self.aggregate = nn.Sequential(
            nn.Conv2d(num_groups * max_disp, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.05)

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f_left.shape
        g, cg = self.num_groups, self.group_channels
        max_d = min(self.max_disp, w)

        fl = F.normalize(f_left.view(b, g, cg, h, w), dim=2)
        fr = F.normalize(f_right.view(b, g, cg, h, w), dim=2)

        corr_vol = f_left.new_zeros(b, g * self.max_disp, h, w)

        for d in range(max_d):
            if d == 0:
                corr = (fl * fr).mean(dim=2)
            else:
                corr = (fl[:, :, :, :, d:] * fr[:, :, :, :, : w - d]).mean(dim=2)
                corr = F.pad(corr, (d, 0), mode="constant", value=0)
            corr_vol[:, d * g : (d + 1) * g] = corr

        out = self.aggregate(corr_vol)
        return f_left + out * self.beta


class StableBottleneck(nn.Module):
    """Bottleneck correlation prior + StableDEPA."""

    def __init__(
        self,
        channels: int,
        depa_k: int,
        num_heads: int,
        max_disparity: int,
        corr_max_disp: int,
        sample_span: float = 4.0,
        residual_range: float = 3.0,
    ):
        super().__init__()
        self.corr_prior = StableBottleneckCorrelationPrior(channels, max_disp=corr_max_disp)
        self.depa = StableDEPA(
            channels=channels,
            k=depa_k,
            num_heads=num_heads,
            max_disparity=max_disparity,
            sample_span=sample_span,
            residual_range=residual_range,
        )

    def forward(self, f_target: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
        x = self.corr_prior(f_target, f_ref)
        return self.depa(x, f_ref)


class StableDecoderStage(nn.Module):
    """PixelShuffle upsample -> skip add -> NAF refinement."""

    def __init__(self, channels: int, num_blocks: int = 4):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.blocks = nn.Sequential(*[NAFBlock(channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.blocks(x + skip)


class SimplePixelShuffleHead(nn.Module):
    """Simple reconstruction head retained in integration."""

    def __init__(self, channels: int, scale: int, out_channels: int = 3):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, out_channels * (scale**2), 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class StereoSRNetStableIntegration(nn.Module):
    """
    Integration example showing where to place Stable DEPA in encoder stages and
    how to preserve bottleneck correlation prior, occlusion fusion, and PixelShuffle head.
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        embed_dim: int = 48,
        enc_blocks: Optional[List[int]] = None,
        dec_blocks: Optional[List[int]] = None,
        depa_k: int = 8,
        # depa_max_disparity: int = 32,
        depa_max_disparity: Optional[int] = None,  # set automatically based on scale if None   
        num_heads: int = 4,
        corr_max_disp: Optional[int] = None,  # set automatically based on depa_max_disparity if None
        drop_path_rate: float = 0.1,
        sample_span: float = 4.0,
        residual_range: float = 3.0,
    ):
        super().__init__()
        if depa_max_disparity is None:
            depa_max_disparity = 64 if scale == 2 else 32

        if corr_max_disp is None:
            corr_max_disp = depa_max_disparity // 2
        if enc_blocks is None:
            enc_blocks = [4, 6, 6]
        if dec_blocks is None:
            dec_blocks = [4, 4]

        self.scale = scale
        num_enc = len(enc_blocks)
        num_down = num_enc - 1

        if len(dec_blocks) != num_down:
            raise ValueError(
                f"dec_blocks length ({len(dec_blocks)}) must equal number of downsamples ({num_down})"
            )

        self.shallow = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 3, padding=1),
            LayerNorm2d(embed_dim),
        )

        total_enc = sum(enc_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_enc)]

        self.encoder_stages = nn.ModuleList()
        block_idx = 0
        for i, nb in enumerate(enc_blocks):
            rates = dpr[block_idx : block_idx + nb]
            block_idx += nb

            # Requested stage-wise disparity scaling.
            stage_disp = depa_max_disparity // (2 ** i)

            self.encoder_stages.append(
                StableEncoderStage(
                    channels=embed_dim,
                    num_blocks=nb,
                    depa_k=depa_k,
                    depa_max_disparity=stage_disp,
                    num_heads=num_heads,
                    downsample=(i < num_down),
                    drop_path_rates=rates,
                    sample_span=sample_span,
                    residual_range=residual_range,
                )
            )

        bottleneck_disp = depa_max_disparity // (2 ** num_down)
        self.bottleneck = StableBottleneck(
            channels=embed_dim,
            depa_k=depa_k,
            num_heads=num_heads,
            max_disparity=bottleneck_disp,
            corr_max_disp=corr_max_disp,
            sample_span=sample_span,
            residual_range=residual_range,
        )

        self.decoder_stages = nn.ModuleList([StableDecoderStage(embed_dim, nb) for nb in dec_blocks])

        self.freq_refine = LightweightFrequencyRefinement(embed_dim)
        self.occlusion = HybridOcclusionFusion(embed_dim)
        self.upsampler = SimplePixelShuffleHead(embed_dim, scale, out_channels=in_channels)

    def _encode(
        self, fl: torch.Tensor, fr: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        skips_l: List[torch.Tensor] = []
        skips_r: List[torch.Tensor] = []
        pre_l, pre_r = fl, fr

        for i, stage in enumerate(self.encoder_stages):
            fl_skip, fr_skip, fl_next, fr_next, fl_pre, fr_pre = stage(fl, fr)
            skips_l.append(fl_skip)
            skips_r.append(fr_skip)
            if i == len(self.encoder_stages) - 1:
                pre_l, pre_r = fl_pre, fr_pre
            fl, fr = fl_next, fr_next

        return skips_l, skips_r, pre_l, pre_r

    def _decode(self, pre_target: torch.Tensor, pre_ref: torch.Tensor, skips_target: List[torch.Tensor]) -> torch.Tensor:
        x = self.bottleneck(pre_target, pre_ref)
        for i, dec in enumerate(self.decoder_stages):
            skip_idx = len(self.encoder_stages) - 2 - i
            x = dec(x, skips_target[skip_idx])
        return self.freq_refine(x)

    def forward(
        self, img_left: torch.Tensor, img_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fl = self.shallow(img_left)
        fr = self.shallow(img_right)

        skips_l, skips_r, pre_l, pre_r = self._encode(fl, fr)

        xl = self._decode(pre_l, pre_r, skips_l)
        xr = self._decode(pre_r, pre_l, skips_r)

        # Keep occlusion fusion.
        fused_l, conf_l = self.occlusion(xl, xr)
        fused_r, conf_r = self.occlusion(xr, xl)

        # Keep simple PixelShuffle reconstruction head.
        sr_left = self.upsampler(fused_l)
        sr_right = self.upsampler(fused_r)

        # Keep global residual learning.
        lr_up_l = F.interpolate(img_left, scale_factor=self.scale, mode="bilinear", align_corners=False)
        lr_up_r = F.interpolate(img_right, scale_factor=self.scale, mode="bilinear", align_corners=False)
        sr_left = sr_left + lr_up_l
        sr_right = sr_right + lr_up_r

        conf_l = F.interpolate(conf_l, size=sr_left.shape[-2:], mode="bilinear", align_corners=False)
        conf_r = F.interpolate(conf_r, size=sr_right.shape[-2:], mode="bilinear", align_corners=False)

        return sr_left, sr_right, conf_l, conf_r


# -----------------------------------------------------------------------------
# Debug utilities / recommended defaults
# -----------------------------------------------------------------------------


@dataclass
class StableDEPARecommendedDefaults:
    k: int = 8
    max_disparity: int = 32
    sample_span: float = 4.0
    residual_range: float = 3.0


def save_disparity_debug(disp: torch.Tensor, conf: torch.Tensor, save_prefix: str) -> None:
    """
    Save disparity prior and confidence for debugging.
    Requires matplotlib; if unavailable, function exits silently.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    d = disp[0, 0].detach().cpu().numpy()
    c = conf[0, 0].detach().cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.imshow(d, cmap="plasma")
    plt.colorbar()
    plt.title("Disparity Prior")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_disp.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.imshow(c, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.title("Disparity Prior Confidence")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_conf.png", dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b, c, h, w = 1, 48, 30, 90
    left = torch.randn(b, c, h, w, device=device)
    right = torch.randn(b, c, h, w, device=device)

    print("Running DisparityPriorEstimator smoke test...")
    prior = DisparityPriorEstimator(channels=c, max_disparity=32).to(device)
    disp, conf = prior(left, right)
    print("  disparity:", tuple(disp.shape), "confidence:", tuple(conf.shape))

    print("Running StableDEPA smoke test...")
    depa = StableDEPA(channels=c, k=8, num_heads=4, max_disparity=32).to(device)
    out = depa(left, right)
    print("  depa output:", tuple(out.shape))

    print("Running BidirectionalStableDEPA smoke test...")
    bi = BidirectionalStableDEPA(channels=c, k=8, num_heads=4, max_disparity=32).to(device)
    l2r, r2l = bi(left, right)
    print("  l2r:", tuple(l2r.shape), "r2l:", tuple(r2l.shape))

    print("Running StereoSRNetStableIntegration smoke test...")
    net = StereoSRNetStableIntegration(scale=4, embed_dim=48).to(device)
    in_l = torch.randn(1, 3, 30, 90, device=device)
    in_r = torch.randn(1, 3, 30, 90, device=device)
    with torch.no_grad():
        sr_l, sr_r, conf_l, conf_r = net(in_l, in_r)
    print("  sr_l:", tuple(sr_l.shape), "sr_r:", tuple(sr_r.shape))
    print("  conf_l:", tuple(conf_l.shape), "conf_r:", tuple(conf_r.shape))

    defaults = StableDEPARecommendedDefaults()
    print("Recommended defaults:", defaults)
