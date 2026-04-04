from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


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
