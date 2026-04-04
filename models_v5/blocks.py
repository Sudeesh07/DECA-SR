from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for image restoration."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class GatedConvFFN(nn.Module):
    """Lightweight SR-oriented FFN."""

    def __init__(self, channels: int, expansion: float = 2.0):
        super().__init__()
        hidden = int(channels * expansion)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden * 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2),
            nn.GLU(dim=1),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualConvBlock(nn.Module):
    """Refinement block used in decoder and HR reconstruction."""

    def __init__(self, channels: int, expansion: float = 2.0, drop_path: float = 0.0):
        super().__init__()
        hidden = int(channels * expansion)
        self.norm1 = LayerNorm2d(channels)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=max(1, hidden // 8)),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GatedConvFFN(channels, expansion=expansion)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.conv(self.norm1(x))) * self.beta
        x = x + self.drop_path(self.ffn(self.norm2(x))) * self.gamma
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpsampleX2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_drop_path_schedule(depths: Iterable[int], drop_path_rate: float) -> list[list[float]]:
    total = sum(depths)
    if total <= 0:
        return [[] for _ in depths]
    values = torch.linspace(0, drop_path_rate, total).tolist()
    schedules: list[list[float]] = []
    cursor = 0
    for depth in depths:
        schedules.append(values[cursor:cursor + depth])
        cursor += depth
    return schedules


# ---------------------------------------------------------------------------
# NEW: Haar DWT modules for lossless downsampling
# ---------------------------------------------------------------------------

class HaarDWT(nn.Module):
    """Fixed (non-learnable) Haar wavelet forward transform.

    Returns (LL, LH, HL, HH) each [B, C, H/2, W/2].
    """

    def __init__(self, channels: int):
        super().__init__()
        # Haar analysis filters (4 filters, each 2x2)
        # LL = 0.5*(a+b+c+d), LH = 0.5*(a+b-c-d)
        # HL = 0.5*(a-b+c-d), HH = 0.5*(a-b-c+d)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        # Shape: [4*C, 1, 2, 2] for groups=C
        filters = torch.stack([ll, lh, hl, hh], dim=0)  # [4, 2, 2]
        filters = filters.unsqueeze(1)  # [4, 1, 2, 2]
        filters = filters.repeat(channels, 1, 1, 1)  # [4*C, 1, 2, 2]
        self.register_buffer("filters", filters, persistent=False)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        # Pad to even dimensions if needed
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        out = F.conv2d(x, self.filters, stride=2, groups=c)  # [B, 4*C, H/2, W/2]
        # Deinterleave: filters are arranged as [ll_c0, lh_c0, hl_c0, hh_c0, ll_c1, ...]
        out = out.view(b, c, 4, out.shape[2], out.shape[3])
        ll = out[:, :, 0]
        lh = out[:, :, 1]
        hl = out[:, :, 2]
        hh = out[:, :, 3]
        return ll, lh, hl, hh


class InverseHaarDWT(nn.Module):
    """Fixed (non-learnable) Haar wavelet inverse transform.

    Input: (LL, LH, HL, HH) each [B, C, H/2, W/2].
    Output: [B, C, H, W].
    """

    def __init__(self, channels: int):
        super().__init__()
        # Haar synthesis filters (transpose of analysis)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        filters = torch.stack([ll, lh, hl, hh], dim=0)  # [4, 2, 2]
        filters = filters.unsqueeze(1)  # [4, 1, 2, 2]
        filters = filters.repeat(channels, 1, 1, 1)  # [4*C, 1, 2, 2]
        self.register_buffer("filters", filters, persistent=False)
        self.channels = channels

    def forward(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        b, c, h, w = ll.shape
        # Interleave subbands: [B, C, 4, H/2, W/2] -> [B, 4*C, H/2, W/2]
        x = torch.stack([ll, lh, hl, hh], dim=2).view(b, c * 4, h, w)
        return F.conv_transpose2d(x, self.filters, stride=2, groups=c)


class ChannelPackDWT(nn.Module):
    """Lossless 2x downsampling: Haar DWT + 1x1 conv to project channels.

    Replaces DownsampleBlock (strided conv) with a lossless alternative.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = LayerNorm2d(in_channels)
        self.dwt = HaarDWT(in_channels)
        self.proj = nn.Conv2d(in_channels * 4, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        ll, lh, hl, hh = self.dwt(x)
        packed = torch.cat([ll, lh, hl, hh], dim=1)  # [B, 4*C, H/2, W/2]
        return self.proj(packed)


class SubbandAttentionGate(nn.Module):
    """Frequency-aware skip connection gating via per-subband channel attention.

    Decomposes skip features into wavelet subbands, applies independent
    channel attention to each, then reconstructs. Very lightweight.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dwt = HaarDWT(channels)
        self.idwt = InverseHaarDWT(channels)
        # Per-subband channel attention: squeeze -> FC -> sigmoid
        self.gate_ll = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, channels), nn.Sigmoid())
        self.gate_lh = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, channels), nn.Sigmoid())
        self.gate_hl = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, channels), nn.Sigmoid())
        self.gate_hh = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[2], x.shape[3]
        ll, lh, hl, hh = self.dwt(x)
        ll = ll * self.gate_ll(ll).unsqueeze(-1).unsqueeze(-1)
        lh = lh * self.gate_lh(lh).unsqueeze(-1).unsqueeze(-1)
        hl = hl * self.gate_hl(hl).unsqueeze(-1).unsqueeze(-1)
        hh = hh * self.gate_hh(hh).unsqueeze(-1).unsqueeze(-1)
        out = self.idwt(ll, lh, hl, hh)
        # Trim to original size (needed when input has odd dimensions)
        return out[:, :, :orig_h, :orig_w]
