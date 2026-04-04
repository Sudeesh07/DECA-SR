from __future__ import annotations

from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DropPath, GatedConvFFN, LayerNorm2d


def _make_base_offsets(num_points: int) -> torch.Tensor:
    if num_points <= 0:
        raise ValueError("num_points must be positive")

    candidates = [(dy, dx) for dy, dx in product((-1.0, 0.0, 1.0), repeat=2)]
    if num_points <= len(candidates):
        center_first = [
            (0.0, 0.0),
            (0.0, -1.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (1.0, 0.0),
            (-1.0, -1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (1.0, 1.0),
        ]
        coords = center_first[:num_points]
    else:
        coords = candidates
        extra = num_points - len(candidates)
        for i in range(extra):
            coords.append((0.0, float(i + 2)))

    tensor = torch.tensor([(dx, dy) for dy, dx in coords], dtype=torch.float32)
    return tensor.view(1, num_points, 1, 1, 2)


class AdaptiveSamplingAttention(nn.Module):
    """
    Real local adaptive sampling attention.

    A query pixel predicts offsets around a local base pattern, samples keys/values
    from those positions, and aggregates them with attention weights.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_points: int = 9,
        max_residual_offset: float = 1.5,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.num_points = num_points
        self.max_residual_offset = max_residual_offset

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, num_points * 2, 1),
        )
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.logit_scale = nn.Parameter(torch.tensor(self.head_dim ** -0.5).log())
        self.register_buffer("base_offsets", _make_base_offsets(num_points), persistent=False)

    def _sample(self, feat: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        base_y = torch.linspace(-1.0, 1.0, h, device=feat.device, dtype=feat.dtype)
        base_x = torch.linspace(-1.0, 1.0, w, device=feat.device, dtype=feat.dtype)
        grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")
        grid_x = grid_x.view(1, 1, h, w).expand(b, self.num_points, -1, -1)
        grid_y = grid_y.view(1, 1, h, w).expand(b, self.num_points, -1, -1)

        offsets_x = offsets[..., 0]
        offsets_y = offsets[..., 1]
        px_to_grid_x = 2.0 / max(w - 1, 1)
        px_to_grid_y = 2.0 / max(h - 1, 1)

        sample_x = grid_x + offsets_x * px_to_grid_x
        sample_y = grid_y + offsets_y * px_to_grid_y
        grid = torch.stack([sample_x, sample_y], dim=-1)

        feat_rep = feat.unsqueeze(1).expand(-1, self.num_points, -1, -1, -1).reshape(b * self.num_points, c, h, w)
        grid = grid.reshape(b * self.num_points, h, w, 2)
        sampled = F.grid_sample(
            feat_rep,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.view(b, self.num_points, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        q = self.q_proj(x).view(b, self.num_heads, self.head_dim, h * w)
        k = self.k_proj(x)
        v = self.v_proj(x)

        residual = torch.tanh(self.offset_net(x))
        residual = residual.view(b, self.num_points, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()
        offsets = self.base_offsets + residual * self.max_residual_offset

        k_samp = self._sample(k, offsets)
        v_samp = self._sample(v, offsets)

        k_samp = k_samp.view(b, self.num_heads, self.head_dim, h * w, self.num_points)
        v_samp = v_samp.view(b, self.num_heads, self.head_dim, h * w, self.num_points)

        logits = torch.einsum("bhdn,bhdnk->bhnk", q, k_samp) * self.logit_scale.exp()
        attn = logits.softmax(dim=-1)
        out = torch.einsum("bhnk,bhdnk->bhdn", attn, v_samp).reshape(b, self.channels, h, w)
        return self.out_proj(out)


class DeformableLocalFeatureBlock(nn.Module):
    """Encoder block built around adaptive local sampling."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_points: int = 9,
        ffn_expand: float = 2.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attn = AdaptiveSamplingAttention(
            channels=channels,
            num_heads=num_heads,
            num_points=num_points,
        )
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GatedConvFFN(channels, expansion=ffn_expand)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x))) * self.beta
        x = x + self.drop_path(self.ffn(self.norm2(x))) * self.gamma
        return x
