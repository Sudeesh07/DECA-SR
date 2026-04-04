from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualConvBlock, UpsampleX2


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        self.up = UpsampleX2(in_channels, out_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 1),
            nn.GELU(),
        )
        self.refine = nn.Sequential(*[ResidualConvBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat([x, skip], dim=1))
        return self.refine(x)


class HRStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        self.up = UpsampleX2(in_channels, out_channels)
        self.refine = nn.Sequential(*[ResidualConvBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.up(x))


class ProgressiveReconstructionDecoder(nn.Module):
    """Stronger progressive decoder for x2/x4 benchmark-oriented SR."""

    def __init__(
        self,
        stage_channels: tuple[int, int, int],
        scale: int,
        out_channels: int = 3,
        decoder_blocks: tuple[int, int] = (4, 4),
        hr_blocks: tuple[int, int] = (3, 3),
        hr_channels: int = 64,
    ):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"Unsupported scale: {scale}")

        c1, c2, c3 = stage_channels
        self.scale = scale

        self.decode_2x = DecoderStage(c3, c2, c2, decoder_blocks[0])
        self.decode_1x = DecoderStage(c2, c1, c1, decoder_blocks[1])

        self.to_hr = nn.Conv2d(c1, hr_channels, 1)
        self.hr_x2 = HRStage(hr_channels, hr_channels, hr_blocks[0])
        self.to_rgb_x2 = nn.Conv2d(hr_channels, out_channels, 3, padding=1)

        if scale == 4:
            self.hr_x4 = HRStage(hr_channels, hr_channels // 2, hr_blocks[1])
            self.to_rgb_x4 = nn.Conv2d(hr_channels // 2, out_channels, 3, padding=1)
        else:
            self.hr_x4 = None
            self.to_rgb_x4 = None

    def forward(self, bottleneck: torch.Tensor, skip_stage2: torch.Tensor, skip_stage1: torch.Tensor) -> torch.Tensor:
        x = self.decode_2x(bottleneck, skip_stage2)
        x = self.decode_1x(x, skip_stage1)

        hr = self.to_hr(x)
        hr = self.hr_x2(hr)
        if self.scale == 2:
            return self.to_rgb_x2(hr)

        assert self.hr_x4 is not None
        assert self.to_rgb_x4 is not None
        hr = self.hr_x4(hr)
        return self.to_rgb_x4(hr)
