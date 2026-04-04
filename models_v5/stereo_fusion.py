from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm2d


@dataclass
class StereoFusionStats:
    disparity: torch.Tensor
    confidence: torch.Tensor


def resolve_default_stereo_ranges(scale: int) -> tuple[int, int]:
    if scale == 2:
        return 64, 16
    if scale == 4:
        return 32, 12
    raise ValueError(f"Unsupported scale: {scale}")


class GroupwiseDisparityPrior(nn.Module):
    """Soft-argmax disparity prior on deep features."""

    def __init__(self, channels: int, max_disp: int, num_groups: int = 8):
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError(f"channels={channels} must be divisible by num_groups={num_groups}")
        self.channels = channels
        self.max_disp = max_disp
        self.num_groups = num_groups
        self.group_channels = channels // num_groups
        self.refine = nn.Sequential(
            nn.Conv2d(max_disp + 1, max(32, max_disp + 1), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(max(32, max_disp + 1), max_disp + 1, 3, padding=1),
        )
        self.register_buffer(
            "disp_values",
            torch.arange(0, max_disp + 1, dtype=torch.float32).view(1, max_disp + 1, 1, 1),
            persistent=False,
        )

    def forward(self, target: torch.Tensor, reference: torch.Tensor, reverse: bool = False) -> StereoFusionStats:
        b, _, h, w = target.shape
        group_target = F.normalize(target.view(b, self.num_groups, self.group_channels, h, w), dim=2)
        group_ref = F.normalize(reference.view(b, self.num_groups, self.group_channels, h, w), dim=2)

        cost = target.new_zeros(b, self.max_disp + 1, h, w)
        cost[:, 0] = (group_target * group_ref).mean(dim=2).mean(dim=1)

        max_valid = min(self.max_disp, w - 1)
        for d in range(1, max_valid + 1):
            if reverse:
                shifted = F.pad(group_ref[..., d:], (0, d), mode="constant", value=0.0)
            else:
                shifted = F.pad(group_ref[..., :-d], (d, 0), mode="constant", value=0.0)
            corr = (group_target * shifted).mean(dim=2).mean(dim=1)
            cost[:, d] = corr

        logits = cost + self.refine(cost)
        prob = logits.softmax(dim=1)
        disparity = (prob * self.disp_values).sum(dim=1, keepdim=True)
        confidence = prob.max(dim=1, keepdim=True)[0]
        return StereoFusionStats(disparity=disparity, confidence=confidence)


class GuidedEpipolarAttention(nn.Module):
    """Disparity-prior-guided epipolar attention with sigmoid gating (DCNv4-style)."""

    def __init__(
        self,
        channels: int,
        max_disp: int,
        num_heads: int = 4,
        num_points: int = 5,
        sample_span: float = 2.0,
        residual_range: float = 1.5,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")

        self.channels = channels
        self.max_disp = max_disp
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.num_points = num_points
        self.residual_range = residual_range

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.offset_refine = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, num_points, 3, padding=1),
        )
        # DCNv4: per-point modulation weights (sigmoid, not softmax)
        self.mod_weights = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, num_heads * num_points, 1),
        )
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.logit_scale = nn.Parameter(torch.tensor(self.head_dim ** -0.5).log())
        self.register_buffer(
            "anchors",
            torch.linspace(-sample_span, sample_span, num_points).view(1, num_points, 1, 1),
            persistent=False,
        )

    def _sample_reference(
        self,
        feat: torch.Tensor,
        disparity_samples: torch.Tensor,
        reverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = feat.shape
        base_y = torch.linspace(-1.0, 1.0, h, device=feat.device, dtype=feat.dtype)
        base_x = torch.linspace(-1.0, 1.0, w, device=feat.device, dtype=feat.dtype)
        grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")
        grid_x = grid_x.view(1, 1, h, w).expand(b, self.num_points, -1, -1)
        grid_y = grid_y.view(1, 1, h, w).expand(b, self.num_points, -1, -1)

        sign = 1.0 if reverse else -1.0
        px_to_grid = 2.0 / max(w - 1, 1)
        sample_x = grid_x + sign * disparity_samples * px_to_grid
        valid = (sample_x >= -1.0) & (sample_x <= 1.0)
        grid = torch.stack([sample_x, grid_y], dim=-1)

        feat_rep = feat.unsqueeze(1).expand(-1, self.num_points, -1, -1, -1).reshape(b * self.num_points, c, h, w)
        grid = grid.reshape(b * self.num_points, h, w, 2)
        sampled = F.grid_sample(
            feat_rep,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.view(b, self.num_points, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
        valid = valid.view(b, 1, h, w, self.num_points)
        return sampled, valid

    def forward(
        self,
        target: torch.Tensor,
        reference: torch.Tensor,
        prior: StereoFusionStats,
        reverse: bool = False,
    ) -> torch.Tensor:
        b, _, h, w = target.shape
        q = self.q_proj(target).view(b, self.num_heads, self.head_dim, h * w)
        k = self.k_proj(reference)
        v = self.v_proj(reference)

        offset_input = torch.cat([target, reference, prior.confidence], dim=1)
        residual = torch.tanh(self.offset_refine(offset_input)) * self.residual_range
        disparity_samples = (prior.disparity + self.anchors + residual).clamp(0.0, float(self.max_disp))

        k_samp, valid = self._sample_reference(k, disparity_samples, reverse=reverse)
        v_samp, _ = self._sample_reference(v, disparity_samples, reverse=reverse)

        k_samp = k_samp.view(b, self.num_heads, self.head_dim, h * w, self.num_points)
        v_samp = v_samp.view(b, self.num_heads, self.head_dim, h * w, self.num_points)

        # DCNv4-style: sigmoid modulation instead of softmax attention
        # Compute content-aware attention logits
        logits = torch.einsum("bhdn,bhdnk->bhnk", q, k_samp) * self.logit_scale.exp()
        valid_mask = valid.view(b, 1, h * w, self.num_points)
        logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)

        # Sigmoid gating: each point independently weighted (DCNv4 insight)
        mod = torch.sigmoid(self.mod_weights(offset_input))  # [B, heads*points, H, W]
        mod = mod.view(b, self.num_heads, self.num_points, h, w)
        mod = mod.permute(0, 1, 3, 4, 2).reshape(b, self.num_heads, h * w, self.num_points)
        mod = mod * valid_mask.to(mod.dtype)

        # Combine: sigmoid-modulated attention
        attn = torch.sigmoid(logits) * mod
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        out = torch.einsum("bhnk,bhdnk->bhdn", attn, v_samp).reshape(b, self.channels, h, w)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# NEW: Iterative Disparity Refinement (RAFT-Stereo inspired)
# ---------------------------------------------------------------------------

class ConvGRUCell(nn.Module):
    """Convolutional GRU cell for iterative refinement."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_conv = nn.Conv2d(input_dim + hidden_dim, hidden_dim * 2, 3, padding=1)
        self.candidate_conv = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h], dim=1)
        gates = torch.sigmoid(self.input_conv(combined))
        reset_gate, update_gate = gates.chunk(2, dim=1)
        candidate = torch.tanh(self.candidate_conv(torch.cat([x, reset_gate * h], dim=1)))
        return (1 - update_gate) * h + update_gate * candidate


class IterativeDisparityRefinement(nn.Module):
    """RAFT-style iterative disparity refinement at bottleneck resolution.

    Uses ConvGRU to iteratively refine a coarse disparity estimate using
    correlation feedback from warped features.
    """

    def __init__(self, channels: int, num_iters: int = 2, max_delta: float = 4.0):
        super().__init__()
        self.num_iters = num_iters
        self.max_delta = max_delta
        # Context encoder: target features -> GRU hidden state init
        self.context_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        # GRU input: [disparity(1), correlation(1), context_features(channels)]
        gru_input_dim = 1 + 1 + channels
        self.gru = ConvGRUCell(gru_input_dim, channels)
        self.delta_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 3, padding=1),
        )

    def forward(
        self,
        d_init: torch.Tensor,
        feat_target: torch.Tensor,
        feat_ref: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        b, _, h, w = feat_target.shape
        context = self.context_net(feat_target)
        hidden = torch.tanh(context)
        d = d_init.detach()  # Don't backprop through initial disparity

        # Normalize features for correlation
        target_norm = F.normalize(feat_target, dim=1)

        for _ in range(self.num_iters):
            # Warp reference features by current disparity estimate
            base_y = torch.linspace(-1.0, 1.0, h, device=d.device, dtype=d.dtype)
            base_x = torch.linspace(-1.0, 1.0, w, device=d.device, dtype=d.dtype)
            grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")
            grid_x = grid_x.view(1, 1, h, w).expand(b, 1, -1, -1)
            grid_y = grid_y.view(1, 1, h, w).expand(b, 1, -1, -1)

            sign = 1.0 if reverse else -1.0
            px_to_grid = 2.0 / max(w - 1, 1)
            sample_x = grid_x + sign * d * px_to_grid
            grid = torch.stack([sample_x.squeeze(1), grid_y.squeeze(1)], dim=-1)

            warped_ref = F.grid_sample(feat_ref, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            ref_norm = F.normalize(warped_ref, dim=1)

            # Correlation: [B, 1, H, W]
            corr = (target_norm * ref_norm).sum(dim=1, keepdim=True)

            # GRU update
            gru_input = torch.cat([d, corr, context], dim=1)
            hidden = self.gru(gru_input, hidden)

            # Predict bounded delta
            delta = torch.tanh(self.delta_head(hidden)) * self.max_delta
            d = d + delta

        return d


class StereoFusionBlock(nn.Module):
    """Targeted stereo fusion used only in the deep encoder and bottleneck.

    Supports optional pre-computed disparity to skip prior estimation.
    """

    def __init__(
        self,
        channels: int,
        max_disp: int,
        corr_max_disp: int,
        num_heads: int = 4,
        num_points: int = 5,
        sample_span: float = 2.0,
        residual_range: float = 1.5,
    ):
        super().__init__()
        self.norm_t = LayerNorm2d(channels)
        self.norm_r = LayerNorm2d(channels)
        self.prior = GroupwiseDisparityPrior(channels, max_disp=corr_max_disp, num_groups=8)
        self.guided = GuidedEpipolarAttention(
            channels=channels,
            max_disp=max_disp,
            num_heads=num_heads,
            num_points=num_points,
            sample_span=sample_span,
            residual_range=residual_range,
        )
        self.corr_max_disp = corr_max_disp
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(
        self,
        target: torch.Tensor,
        reference: torch.Tensor,
        reverse: bool = False,
        disp_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, StereoFusionStats]:
        target_n = self.norm_t(target)
        reference_n = self.norm_r(reference)
        if disp_init is not None:
            # Use pre-computed disparity, still compute confidence from prior
            stats_raw = self.prior(target_n, reference_n, reverse=reverse)
            stats = StereoFusionStats(disparity=disp_init, confidence=stats_raw.confidence)
        else:
            stats = self.prior(target_n, reference_n, reverse=reverse)
        transfer = self.guided(target_n, reference_n, stats, reverse=reverse)
        fused = self.fuse(torch.cat([target_n, transfer, stats.confidence], dim=1))
        return target + fused * self.alpha, stats
