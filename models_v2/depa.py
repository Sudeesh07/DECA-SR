"""
DEPA: Deformable Epipolar Paired-Attention for Stereo Image Super-Resolution

Combines two complementary cross-view attention mechanisms:
- ESA (Epipolar Spatial Attention): Deformable sampling along horizontal epipolar lines
  Answers WHERE the correspondence is for each query position.
- ECA (Epipolar Channel Attention): Transposed channel-wise cross-view attention
  Answers WHAT information to transfer between views.

Fused via concat + conv, with learnable residual scaling for stable training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from models.nafblock import LayerNorm2d
from models.deca import DECAOffsetPredictor


# ── Tensor reshaping helpers (einops-free) ──────────────────────────────


def _to_multihead(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """[B, C, H, W] -> [B, num_heads, head_dim, H*W]"""
    B, C, H, W = x.shape
    return x.view(B, num_heads, head_dim, H * W)


def _sampled_to_multihead(
    x: torch.Tensor, B: int, K: int, num_heads: int, head_dim: int
) -> torch.Tensor:
    """[K, B, C, H, W] -> [B, num_heads, head_dim, H*W, K]"""
    x = x.permute(1, 0, 2, 3, 4).contiguous()
    H, W = x.shape[3], x.shape[4]
    x = x.view(B, K, num_heads, head_dim, H, W)
    x = x.permute(0, 2, 3, 4, 5, 1).contiguous()
    return x.reshape(B, num_heads, head_dim, H * W, K)


def _from_multihead(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """[B, num_heads, head_dim, H*W] -> [B, C, H, W]"""
    B, nh, hd, N = x.shape
    return x.reshape(B, nh * hd, H, W)


# ── ESA: Epipolar Spatial Attention ─────────────────────────────────────


class EpipolarSpatialAttention(nn.Module):
    """
    Deformable sampling + multi-head attention along horizontal epipolar lines.

    For each query position in the left view, predicts K horizontal offsets,
    samples keys/values from the right view at those positions, and computes
    attention-weighted aggregation. Complexity: O(H*W*K) per head.
    """

    def __init__(
        self,
        channels: int,
        k: int = 8,
        num_heads: int = 4,
        max_disparity: int = 32,
        num_corr_shifts: int = 9,
    ):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.offset_predictor = DECAOffsetPredictor(
            channels, k, max_disparity, num_corr_shifts
        )
        self.log_scale = nn.Parameter(
            torch.tensor(self.head_dim ** -0.5).log()
        )

    def forward(self, fl: torch.Tensor, fr: torch.Tensor) -> torch.Tensor:
        """Returns raw attention output [B, C, H, W] (no residual)."""
        B, C, H, W = fl.shape

        offsets = self.offset_predictor(fl, fr)  # [B, K, H, W]
        q, k, v = self.q_proj(fl), self.k_proj(fr), self.v_proj(fr)

        # Base grid (created once, reused for all K samples)
        gy = torch.linspace(-1, 1, H, device=fl.device)
        gx = torch.linspace(-1, 1, W, device=fl.device)
        by, bx = torch.meshgrid(gy, gx, indexing="ij")
        bx = bx.unsqueeze(0).expand(B, -1, -1)
        by = by.unsqueeze(0).expand(B, -1, -1)
        px2g = 2.0 / (W - 1) if W > 1 else 0.0

        # Sample K positions from right view
        ks_list, vs_list = [], []
        for i in range(self.k):
            sx = (bx + offsets[:, i] * px2g).clamp(-1, 1)
            grid = torch.stack([sx, by], dim=-1)
            ks_list.append(
                F.grid_sample(k, grid, mode="bilinear", padding_mode="border", align_corners=True)
            )
            vs_list.append(
                F.grid_sample(v, grid, mode="bilinear", padding_mode="border", align_corners=True)
            )

        # Multi-head attention over K samples
        ks = _sampled_to_multihead(
            torch.stack(ks_list), B, self.k, self.num_heads, self.head_dim
        )
        vs = _sampled_to_multihead(
            torch.stack(vs_list), B, self.k, self.num_heads, self.head_dim
        )
        q_mh = _to_multihead(q, self.num_heads, self.head_dim)

        scale = self.log_scale.exp()
        attn = torch.einsum("bhdn,bhdnk->bhnk", q_mh, ks) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnk,bhdnk->bhdn", attn, vs)

        return _from_multihead(out, H, W)


# ── ECA: Epipolar Channel Attention ────────────────────────────────────


class EpipolarChannelAttention(nn.Module):
    """
    Transposed channel-wise cross-view attention (einops-free).

    Computes d x d (channel-channel) attention between views.
    Q/K are L2-normalized along spatial dim for cosine similarity.
    Complexity: O(d^2) per head — efficient at any spatial resolution.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.temperature = nn.Parameter(
            torch.ones(num_heads, 1, 1) * (self.head_dim ** 0.5)
        )

    def forward(self, fl: torch.Tensor, fr: torch.Tensor) -> torch.Tensor:
        """Returns raw attention output [B, C, H, W] (no residual)."""
        B, C, H, W = fl.shape
        h, d, N = self.num_heads, self.head_dim, H * W

        q = self.q_proj(fl).view(B, h, d, N)
        k = self.k_proj(fr).view(B, h, d, N)
        v = self.v_proj(fr).view(B, h, d, N)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature  # [B, h, d, d]
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # [B, h, d, N]

        return out.reshape(B, C, H, W)


# ── DEPA: Combined Block ───────────────────────────────────────────────


class DEPA(nn.Module):
    """
    Deformable Epipolar Paired-Attention.

    Combines ESA (spatial) + ECA (channel) with learned fusion.
    Each branch produces raw cross-view features; they are concatenated,
    fused via 1x1 conv, and added to the input as a scaled residual.
    """

    def __init__(
        self,
        channels: int,
        k: int = 8,
        num_heads: int = 4,
        max_disparity: int = 32,
        num_corr_shifts: int = 9,
    ):
        super().__init__()
        self.norm_l = LayerNorm2d(channels)
        self.norm_r = LayerNorm2d(channels)

        self.esa = EpipolarSpatialAttention(
            channels, k, num_heads, max_disparity, num_corr_shifts
        )
        self.eca = EpipolarChannelAttention(channels, num_heads)

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1)*0.05)  # Learnable residual scaling

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        fl = self.norm_l(f_left)
        fr = self.norm_r(f_right)

        esa_out = self.esa(fl, fr)
        eca_out = self.eca(fl, fr)
        fused = self.fusion(torch.cat([esa_out, eca_out], dim=1))

        return f_left + fused * self.beta


class BidirectionalDEPA(nn.Module):
    """Separate L->R and R->L DEPA for asymmetric disparity patterns."""

    def __init__(
        self,
        channels: int,
        k: int = 8,
        num_heads: int = 4,
        max_disparity: int = 32,
        num_corr_shifts: int = 9,
    ):
        super().__init__()
        self.l2r = DEPA(channels, k, num_heads, max_disparity, num_corr_shifts)
        self.r2l = DEPA(channels, k, num_heads, max_disparity, num_corr_shifts)

    def forward(
        self, f_left: torch.Tensor, f_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.l2r(f_left, f_right), self.r2l(f_right, f_left)
