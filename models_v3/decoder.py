"""Progressive decoder for DSRPSNRNet."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models_v3.blocks import ReconstructionBlock


class ProgressiveDecoder(nn.Module):
    """
    Progressive decoder with concat skip fusion.

    For scale=4:
    - x2 upsample + skip2 concat + refine
    - x4 upsample + skip1 concat + refine

    For scale=2:
    - uses the first x2 branch only.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int = 3,
        scale: int = 4,
        refine_blocks_per_stage: int = 2,
        sr_channels: int | None = None,
    ):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"scale must be 2 or 4, got {scale}")

        self.scale = scale

        self.up1 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.fuse1 = nn.Conv2d(channels * 2, channels, 1)
        self.refine1 = nn.Sequential(*[ReconstructionBlock(channels) for _ in range(refine_blocks_per_stage)])

        self.up2 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.fuse2 = nn.Conv2d(channels * 2, channels, 1)
        self.refine2 = nn.Sequential(*[ReconstructionBlock(channels) for _ in range(refine_blocks_per_stage)])

        if sr_channels is None:
            sr_channels = max(32, channels // 2)

        self.sr_in = nn.Conv2d(channels, sr_channels, 1)
        self.sr_up1 = nn.Sequential(
            nn.Conv2d(sr_channels, sr_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.sr_refine1 = nn.Sequential(*[ReconstructionBlock(sr_channels) for _ in range(refine_blocks_per_stage)])

        self.sr_up2 = nn.Sequential(
            nn.Conv2d(sr_channels, sr_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.sr_refine2 = nn.Sequential(*[ReconstructionBlock(sr_channels) for _ in range(refine_blocks_per_stage)])

        self.to_rgb_x2 = nn.Conv2d(sr_channels, out_channels, 3, padding=1)
        self.to_rgb_x4 = nn.Conv2d(sr_channels, out_channels, 3, padding=1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skip_stage2: torch.Tensor,
        skip_stage1: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up1(bottleneck)
        if x.shape[-2:] != skip_stage2.shape[-2:]:
            x = F.interpolate(x, size=skip_stage2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse1(torch.cat([x, skip_stage2], dim=1))
        x = self.refine1(x)

        if self.scale == 2:
            y = self.sr_in(self.up2(x))
            y = self.sr_up1(y)
            y = self.sr_refine1(y)
            return self.to_rgb_x2(y)

        x = self.up2(x)
        if x.shape[-2:] != skip_stage1.shape[-2:]:
            x = F.interpolate(x, size=skip_stage1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse2(torch.cat([x, skip_stage1], dim=1))
        x = self.refine2(x)

        # Progressive SR head from LR features: x2 then x4.
        y = self.sr_in(x)
        y = self.sr_up1(y)
        y = self.sr_refine1(y)
        y = self.sr_up2(y)
        y = self.sr_refine2(y)
        return self.to_rgb_x4(y)
