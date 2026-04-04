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
    """Disparity-prior-guided epipolar attention around a small candidate set."""

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

        logits = torch.einsum("bhdn,bhdnk->bhnk", q, k_samp) * self.logit_scale.exp()
        valid = valid.view(b, 1, h * w, self.num_points)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        attn = logits.softmax(dim=-1)
        attn = attn * valid.to(attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        out = torch.einsum("bhnk,bhdnk->bhdn", attn, v_samp).reshape(b, self.channels, h, w)
        return self.out_proj(out)


class StereoFusionBlock(nn.Module):
    """Targeted stereo fusion used only in the deep encoder and bottleneck."""

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
    ) -> tuple[torch.Tensor, StereoFusionStats]:
        target_n = self.norm_t(target)
        reference_n = self.norm_r(reference)
        stats = self.prior(target_n, reference_n, reverse=reverse)
        transfer = self.guided(target_n, reference_n, stats, reverse=reverse)
        fused = self.fuse(torch.cat([target_n, transfer, stats.confidence], dim=1))
        return target + fused * self.alpha, stats
